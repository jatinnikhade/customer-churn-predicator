"""Customer Churn Prediction System - Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg  # noqa: E402
from src import data_processing as dp  # noqa: E402
from src import explain as ex  # noqa: E402
from src import model_training as mt  # noqa: E402
from src import predict as pr  # noqa: E402
from src import recommendations as rec  # noqa: E402
from src import visuals as vz  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px;}
      .hero {background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#DB2777 100%);
             color:#fff; padding: 26px 30px; border-radius: 18px; margin-bottom: 22px;
             box-shadow: 0 8px 24px rgba(79,70,229,.25);}
      .hero h1 {margin:0; font-size: 2rem; font-weight: 800; color:#fff; padding:0;}
      .hero p {margin:6px 0 0 0; opacity:.92; font-size: 1.02rem;}
      .kpi {border-radius:14px; padding:16px 18px; color:#fff; box-shadow:0 4px 14px rgba(0,0,0,.10);}
      .kpi .l {font-size:.74rem; opacity:.88; text-transform:uppercase; letter-spacing:.07em;}
      .kpi .v {font-size:1.8rem; font-weight:800; line-height:1.25;}
      .kpi .s {font-size:.78rem; opacity:.88;}
      .rec {padding:12px 16px; border-radius:10px; margin-bottom:10px; background:rgba(127,127,127,.09);}
      .rec .t {font-weight:700; font-size:1.0rem;}
      .rec .a {margin-top:2px;}
      .rec .r {margin-top:4px; font-size:.86rem; opacity:.75;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem;
             font-weight:700; color:#fff; margin-right:8px; vertical-align:middle;}
      .badge {display:inline-block; padding:8px 20px; border-radius:999px; font-weight:800;
              color:#fff; font-size:1.15rem;}
      .insight {padding:10px 14px; margin-bottom:8px; border-radius:10px;
                background:rgba(79,70,229,.08); border-left:4px solid #4F46E5;}
      div[data-testid="stSidebar"] .stRadio label {font-size: 1.02rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

KPI_GRADIENTS = {
    "indigo": "linear-gradient(135deg,#4F46E5,#6366F1)",
    "red": "linear-gradient(135deg,#DC2626,#F87171)",
    "green": "linear-gradient(135deg,#059669,#34D399)",
    "amber": "linear-gradient(135deg,#D97706,#FBBF24)",
    "violet": "linear-gradient(135deg,#7C3AED,#A78BFA)",
}


def kpi(label: str, value: str, sub: str = "", colour: str = "indigo") -> str:
    return (
        f'<div class="kpi" style="background:{KPI_GRADIENTS[colour]}">'
        f'<div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}&nbsp;</div></div>'
    )


def show_chart(fig, key: str | None = None) -> None:
    """Render a Plotly figure full-width (works across Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_raw() -> pd.DataFrame:
    return dp.load_raw_data()


@st.cache_data(show_spinner=False)
def get_clean() -> pd.DataFrame:
    return dp.clean_data(get_raw())


@st.cache_resource(show_spinner=False)
def get_models():
    return pr.load_models(), pr.load_metrics()


def reset_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def build_scored(frame: pd.DataFrame, pipe) -> pd.DataFrame:
    """Add churn probability, risk band and the top retention action to each customer."""
    probs = pr.predict_proba(pipe, frame)
    out = frame.copy()
    out["Churn Probability"] = probs
    out["Risk Level"] = [pr.risk_level(p) for p in probs]
    records = out.to_dict("records")
    out["Suggested Action"] = [rec.primary_action(r, p) for r, p in zip(records, probs)]
    return out


@st.cache_data(show_spinner="Scoring customers ...")
def score_demo(model_name: str, trained_at: str) -> pd.DataFrame:  # noqa: ARG001 (args form the cache key)
    models, _ = get_models()
    return build_scored(get_clean(), models[model_name])


# --------------------------------------------------------------------------- #
# Bootstrap: make sure data + models exist
# --------------------------------------------------------------------------- #
if not cfg.DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{cfg.DATA_PATH}`. Download the **Telco Customer Churn** CSV from Kaggle "
        "(blastchar/telco-customer-churn) and save it as `data/Telco-Customer-Churn.csv`."
    )
    st.stop()

if not pr.models_available():
    with st.spinner("First launch - training the models (about 20 seconds) ..."):
        mt.train_models(tune=False)
    reset_caches()

raw_df = get_raw()
df = get_clean()
models, metrics = get_models()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
PAGES = [
    "📊 Dashboard",
    "🔍 Data Explorer",
    "🤖 Model Performance",
    "🎯 Predict a Customer",
    "📋 Batch Scoring",
]
with st.sidebar:
    st.markdown("## 📉 Churn Predictor")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    model_names = list(models)
    selected_model = st.selectbox(
        "Prediction model",
        model_names,
        index=model_names.index(metrics["best_model"]),
        help="Used on the 'Predict a Customer' and 'Batch Scoring' pages. The best model by ROC-AUC is pre-selected.",
    )
    st.caption(
        f"**Dataset:** {len(df):,} customers · {df[cfg.TARGET].mean():.1%} churn\n\n"
        f"**Best model:** {metrics['best_model']} (AUC {metrics['models'][metrics['best_model']]['roc_auc']:.3f})"
    )


# =========================================================================== #
# PAGE 1 - DASHBOARD
# =========================================================================== #
def churn_insights(d: pd.DataFrame) -> list[str]:
    def rate(mask) -> float:
        sub = d[mask]
        return float(sub[cfg.TARGET].mean()) if len(sub) else float("nan")

    out = []
    m2m, two = rate(d["Contract"] == "Month-to-month"), rate(d["Contract"] == "Two year")
    if not (np.isnan(m2m) or np.isnan(two)):
        out.append(f"**Contract type is the biggest lever.** Month-to-month customers churn at "
                   f"**{m2m:.0%}**, versus **{two:.0%}** on two-year contracts.")
    new, old = rate(d["tenure"] <= 12), rate(d["tenure"] > 12)
    if not (np.isnan(new) or np.isnan(old)):
        out.append(f"**The first year is critical.** Customers in their first 12 months churn at "
                   f"**{new:.0%}**, versus **{old:.0%}** afterwards.")
    fiber, dsl = rate(d["InternetService"] == "Fiber optic"), rate(d["InternetService"] == "DSL")
    if not (np.isnan(fiber) or np.isnan(dsl)):
        out.append(f"**Fibre customers are at higher risk.** Fibre churn is **{fiber:.0%}** vs **{dsl:.0%}** for DSL, "
                   f"which points to price or service-quality issues.")
    echeck, auto = rate(d["PaymentMethod"] == "Electronic check"), rate(d["PaymentMethod"].str.contains("automatic"))
    if not (np.isnan(echeck) or np.isnan(auto)):
        out.append(f"**Payment method matters.** Electronic-check payers churn at **{echeck:.0%}**, "
                   f"compared with **{auto:.0%}** for automatic payers.")
    internet = d["InternetService"] != "No"
    no_ts, ts = rate(internet & (d["TechSupport"] == "No")), rate(internet & (d["TechSupport"] == "Yes"))
    if not (np.isnan(no_ts) or np.isnan(ts)):
        out.append(f"**Support is protective.** Internet customers without tech support churn at "
                   f"**{no_ts:.0%}**, versus **{ts:.0%}** with it.")
    return out


def page_dashboard() -> None:
    hero("Customer Churn Dashboard",
         "Understand who is leaving, why they leave, and how much revenue is at stake.")

    with st.expander("🔧 Filter customers", expanded=False):
        f1, f2, f3 = st.columns(3)
        contracts = f1.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
        internet = f2.multiselect("Internet service", sorted(df["InternetService"].unique()),
                                  default=sorted(df["InternetService"].unique()))
        t_min, t_max = f3.slider("Tenure (months)", 0, 72, (0, 72))

    fdf = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet) & df["tenure"].between(t_min, t_max)]
    if fdf.empty:
        st.warning("No customers match the current filters.")
        return

    churned = fdf[fdf[cfg.TARGET] == 1]
    k = st.columns(5)
    k[0].markdown(kpi("Customers", f"{len(fdf):,}", "in current view", "indigo"), unsafe_allow_html=True)
    k[1].markdown(kpi("Churn rate", f"{fdf[cfg.TARGET].mean():.1%}", f"{len(churned):,} customers lost", "red"),
                  unsafe_allow_html=True)
    k[2].markdown(kpi("Avg monthly bill", f"${fdf['MonthlyCharges'].mean():,.2f}",
                      f"churned: ${churned['MonthlyCharges'].mean() if len(churned) else 0:,.2f}", "violet"),
                  unsafe_allow_html=True)
    k[3].markdown(kpi("Avg tenure", f"{fdf['tenure'].mean():.1f} mo",
                      f"churned: {churned['tenure'].mean() if len(churned) else 0:.1f} mo", "green"),
                  unsafe_allow_html=True)
    k[4].markdown(kpi("Monthly revenue lost", f"${churned['MonthlyCharges'].sum():,.0f}",
                      f"${churned['MonthlyCharges'].sum() * 12:,.0f} per year", "amber"), unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        show_chart(vz.churn_donut(fdf))
    with c2:
        show_chart(vz.churn_rate_by(fdf, "Contract", "Churn rate by contract type"))

    c3, c4 = st.columns(2)
    with c3:
        show_chart(vz.churn_rate_by_tenure(fdf))
    with c4:
        show_chart(vz.numeric_distribution(fdf, "MonthlyCharges"))

    c5, c6 = st.columns(2)
    with c5:
        show_chart(vz.churn_rate_by(fdf, "InternetService", "Churn rate by internet service"))
    with c6:
        show_chart(vz.churn_rate_by(fdf, "PaymentMethod", "Churn rate by payment method"))

    st.subheader("💡 Key insights")
    insights = churn_insights(fdf)
    if insights:
        for text in insights:
            st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)
    else:
        st.info("Widen the filters to see comparative insights.")


# =========================================================================== #
# PAGE 2 - DATA EXPLORER
# =========================================================================== #
def page_explorer() -> None:
    hero("Data Explorer", "Inspect the data cleaning steps and explore how each feature relates to churn.")
    t1, t2, t3, t4 = st.tabs(["🧹 Data & Cleaning", "🏷️ Categorical features", "📈 Numeric features", "🔗 Correlations"])

    with t1:
        q = dp.data_quality_report(raw_df)
        c = st.columns(5)
        c[0].metric("Rows", f"{q['rows']:,}")
        c[1].metric("Columns", q["columns"])
        c[2].metric("Missing cells", q["missing_cells"])
        c[3].metric("Blank TotalCharges", q["blank_total_charges"])
        c[4].metric("Duplicate IDs", q["duplicate_ids"])

        st.markdown(
            f"""
**Cleaning pipeline**
1. `TotalCharges` is stored as text (`{q['total_charges_dtype']}`) because {q['blank_total_charges']} rows contain a blank space → converted to numeric.
2. Those blank rows are customers with `tenure = 0` (not billed yet), so `TotalCharges` is set to **0**.
3. `SeniorCitizen` (0/1) is converted to **No/Yes** to match the other flags.
4. Whitespace is trimmed, duplicate customer IDs are dropped, and the target `Churn` is encoded **Yes → 1 / No → 0**.
5. Feature engineering: **`NumServices`** counts the add-on services each customer subscribes to.
6. In the model pipeline, numeric features are standardised and categorical features are one-hot encoded.
"""
        )
        blanks = raw_df[pd.to_numeric(raw_df["TotalCharges"], errors="coerce").isna()]
        with st.expander(f"Show the {len(blanks)} raw rows with a blank TotalCharges"):
            show_df(blanks[[cfg.ID_COL, "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])
        st.markdown("**Cleaned dataset preview**")
        show_df(df.head(100))
        with st.expander("Summary statistics"):
            st.dataframe(df[cfg.NUMERIC_FEATURES + [cfg.TARGET]].describe().T.round(2))

    with t2:
        col = st.selectbox("Choose a feature", cfg.CATEGORICAL_FEATURES, index=cfg.CATEGORICAL_FEATURES.index("Contract"))
        left, right = st.columns([1.5, 1])
        with left:
            show_chart(vz.churn_rate_by(df, col))
        with right:
            tbl = (df.groupby(col)[cfg.TARGET].agg(Customers="count", Churned="sum", ChurnRate="mean")
                   .reset_index().sort_values("ChurnRate", ascending=False))
            tbl["ChurnRate"] = (tbl["ChurnRate"] * 100).round(1)
            tbl = tbl.rename(columns={"ChurnRate": "Churn rate (%)"})
            show_df(tbl)
        show_chart(vz.contract_internet_heatmap(df))

    with t3:
        col = st.selectbox("Choose a numeric feature", cfg.NUMERIC_FEATURES, index=0)
        a, b = st.columns(2)
        with a:
            show_chart(vz.numeric_distribution(df, col))
        with b:
            show_chart(vz.numeric_box(df, col))
        grp = df.groupby(cfg.TARGET)[col].describe().round(2)
        grp.index = ["Retained", "Churned"]
        st.dataframe(grp)

    with t4:
        show_chart(vz.correlation_heatmap(df))
        corr = df[cfg.NUMERIC_FEATURES + [cfg.TARGET]].corr()[cfg.TARGET].drop(cfg.TARGET).sort_values()
        st.caption("Correlation with churn - " + ", ".join(f"**{k}** {v:+.2f}" for k, v in corr.items()))


# =========================================================================== #
# PAGE 3 - MODEL PERFORMANCE
# =========================================================================== #
def page_models() -> None:
    hero("Model Performance", "Logistic Regression vs Random Forest, evaluated on a held-out 20% test set.")
    d = metrics["dataset"]
    best = metrics["best_model"]
    st.success(f"🏆 **Best model: {best}** - ROC-AUC {metrics['models'][best]['roc_auc']:.3f}  ·  "
               f"trained {metrics['trained_at'].replace('T', ' ')}  ·  "
               f"{d['train_rows']:,} train / {d['test_rows']:,} test rows  ·  "
               f"{'tuned' if metrics.get('tuned') else 'default hyper-parameters'}")

    rows = []
    for name, m in metrics["models"].items():
        rows.append({
            "Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
            "F1": m["f1"], "ROC-AUC": m["roc_auc"], "CV ROC-AUC (5-fold)": f"{m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f}",
        })
    comp = pd.DataFrame(rows)
    show_df(comp.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]})
            .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], color="rgba(16,185,129,.25)"))

    a, b = st.columns(2)
    with a:
        show_chart(vz.model_comparison(metrics))
    with b:
        show_chart(vz.roc_curves(metrics))

    st.subheader("Confusion matrices")
    cols = st.columns(len(metrics["models"]))
    for col, (name, m) in zip(cols, metrics["models"].items()):
        with col:
            show_chart(vz.confusion_matrix_fig(m["confusion_matrix"], name), key=f"cm_{name}")
            (tn, fp), (fn, tp) = m["confusion_matrix"]
            st.caption(f"Caught **{tp}** of {tp + fn} churners ({tp / (tp + fn):.0%} recall) · "
                       f"{fp} false alarms · {fn} missed")

    st.subheader("What drives churn?")
    i1, i2 = st.columns(2)
    with i1:
        show_chart(vz.feature_importance(metrics["feature_importance"]["Random Forest"]))
    with i2:
        show_chart(vz.logistic_coefficients_fig(metrics["feature_importance"]["Logistic Regression"]))

    with st.expander("📘 How to read these metrics"):
        st.markdown(
            """
- **Accuracy** - share of all predictions that are correct. Misleading here because only ~27% of customers churn.
- **Precision** - of the customers flagged as *will churn*, how many really did. High precision = fewer wasted retention offers.
- **Recall** - of the customers who *really* churned, how many we caught. For retention, recall is usually the metric that matters most.
- **F1** - harmonic mean of precision and recall.
- **ROC-AUC** - how well the model *ranks* customers by risk (0.5 = coin flip, 1.0 = perfect). Threshold-independent.
- Both models use **class weighting** to counter the 73/27 class imbalance, which trades some precision for higher recall.
"""
        )

    with st.expander("🔁 Retrain models"):
        tune = st.checkbox("Tune hyper-parameters with GridSearchCV (slower, ~1-3 min)")
        if st.button("Retrain now", type="primary"):
            with st.spinner("Training ..."):
                mt.train_models(tune=tune)
            reset_caches()
            st.rerun()


# =========================================================================== #
# PAGE 4 - PREDICT A CUSTOMER
# =========================================================================== #
PRESETS = {
    "Average customer": dict(
        gender="Female", SeniorCitizen="No", Partner="Yes", Dependents="No", PhoneService="Yes",
        MultipleLines="No", InternetService="DSL", OnlineSecurity="No", OnlineBackup="No",
        DeviceProtection="No", TechSupport="No", StreamingTV="No", StreamingMovies="No",
        Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Mailed check",
        tenure=30, MonthlyCharges=65.0),
    "High-risk example": dict(
        gender="Female", SeniorCitizen="No", Partner="No", Dependents="No", PhoneService="Yes",
        MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
        DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes",
        Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check",
        tenure=2, MonthlyCharges=95.0),
    "Low-risk example": dict(
        gender="Male", SeniorCitizen="No", Partner="Yes", Dependents="Yes", PhoneService="Yes",
        MultipleLines="Yes", InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
        DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No", StreamingMovies="No",
        Contract="Two year", PaperlessBilling="No", PaymentMethod="Bank transfer (automatic)",
        tenure=58, MonthlyCharges=64.0),
}
FORM_KEYS = list(PRESETS["Average customer"])
YES_NO = ["No", "Yes"]


def apply_preset() -> None:
    choice = st.session_state.get("preset", "Custom")
    if choice in PRESETS:
        for key, value in PRESETS[choice].items():
            st.session_state[f"f_{key}"] = value


def page_predict() -> None:
    hero("Predict a Customer", "Describe a customer to get their churn probability, key risk drivers "
                               "and a tailored retention plan.")
    for key, value in PRESETS["Average customer"].items():
        st.session_state.setdefault(f"f_{key}", value)

    st.selectbox("Start from a sample profile", ["Custom", *PRESETS], key="preset", on_change=apply_preset)

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 👤 Profile")
            st.selectbox("Gender", cfg.CATEGORY_OPTIONS["gender"], key="f_gender")
            st.selectbox("Senior citizen", YES_NO, key="f_SeniorCitizen")
            st.selectbox("Has partner", YES_NO, key="f_Partner")
            st.selectbox("Has dependents", YES_NO, key="f_Dependents")
            st.slider("Tenure (months)", 0, 72, key="f_tenure")
        with c2:
            st.markdown("##### 📡 Services")
            st.selectbox("Phone service", YES_NO, key="f_PhoneService")
            st.selectbox("Multiple lines", YES_NO, key="f_MultipleLines")
            st.selectbox("Internet service", cfg.CATEGORY_OPTIONS["InternetService"], key="f_InternetService")
            a1, a2 = st.columns(2)
            a1.selectbox("Online security", YES_NO, key="f_OnlineSecurity")
            a2.selectbox("Online backup", YES_NO, key="f_OnlineBackup")
            a1.selectbox("Device protection", YES_NO, key="f_DeviceProtection")
            a2.selectbox("Tech support", YES_NO, key="f_TechSupport")
            a1.selectbox("Streaming TV", YES_NO, key="f_StreamingTV")
            a2.selectbox("Streaming movies", YES_NO, key="f_StreamingMovies")
        with c3:
            st.markdown("##### 💳 Account")
            st.selectbox("Contract", cfg.CATEGORY_OPTIONS["Contract"], key="f_Contract")
            st.selectbox("Paperless billing", YES_NO, key="f_PaperlessBilling")
            st.selectbox("Payment method", cfg.CATEGORY_OPTIONS["PaymentMethod"], key="f_PaymentMethod")
            st.slider("Monthly charges ($)", 18.0, 120.0, step=0.5, key="f_MonthlyCharges")
            st.caption("Add-on services and 'multiple lines' are ignored automatically when the customer "
                       "has no internet / phone service.")
        submitted = st.form_submit_button("🔮 Predict churn risk", type="primary")

    if submitted:
        cust = {k: st.session_state[f"f_{k}"] for k in FORM_KEYS}
        cust["tenure"] = int(cust["tenure"])
        cust["MonthlyCharges"] = float(cust["MonthlyCharges"])
        if cust["InternetService"] == "No":
            for svc in cfg.ADDON_SERVICES:
                cust[svc] = "No internet service"
        if cust["PhoneService"] == "No":
            cust["MultipleLines"] = "No phone service"
        cust["TotalCharges"] = round(cust["tenure"] * cust["MonthlyCharges"], 2)
        st.session_state["customer"] = cust

    cust = st.session_state.get("customer")
    if not cust:
        st.info("Fill in the form above and click **Predict churn risk**.")
        return

    frame = dp.prepare_for_model(pd.DataFrame([cust]))
    probs = {name: float(pr.predict_proba(m, frame)[0]) for name, m in models.items()}
    p = probs[selected_model]
    level = pr.risk_level(p)
    colour = cfg.RISK_COLORS[level]

    st.divider()
    left, right = st.columns([1, 1.15])
    with left:
        show_chart(vz.risk_gauge(p), key="gauge")
        st.markdown(
            f'<div style="text-align:center"><span class="badge" style="background:{colour}">'
            f'{level.upper()} RISK</span></div>', unsafe_allow_html=True)
    with right:
        st.markdown(f"#### Prediction summary  ·  *{selected_model}*")
        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{p:.1%}")
        m2.metric("Expected monthly revenue at risk", f"${p * cust['MonthlyCharges']:,.2f}")
        m3, m4 = st.columns(2)
        m3.metric("Lifetime revenue so far", f"${cust['TotalCharges']:,.0f}")
        m4.metric("Verdict", "Likely to churn" if p >= cfg.DECISION_THRESHOLD else "Likely to stay")
        st.markdown("**Model agreement**")
        agree = pd.DataFrame({"Model": list(probs), "Churn probability": [f"{v:.1%}" for v in probs.values()],
                              "Risk": [pr.risk_level(v) for v in probs.values()]})
        show_df(agree)

    st.subheader("🔎 Why this prediction?")
    drivers = ex.explain_customer(models["Logistic Regression"], frame, df)
    d1, d2 = st.columns([1.3, 1])
    with d1:
        show_chart(vz.driver_bars(drivers), key="drivers")
    with d2:
        up = drivers[drivers["contribution"] > 0.05].head(3)
        down = drivers[drivers["contribution"] < -0.05].head(3)
        st.markdown("**🔴 Raising the risk**")
        for _, r in up.iterrows():
            st.markdown(f"- {r['feature']}")
        if up.empty:
            st.markdown("- nothing notable")
        st.markdown("**🟢 Lowering the risk**")
        for _, r in down.iterrows():
            st.markdown(f"- {r['feature']}")
        if down.empty:
            st.markdown("- nothing notable")
        st.caption("Drivers come from the Logistic Regression model, measured against the average customer.")

    st.subheader("🛟 Retention recommendations")
    for r in rec.generate_recommendations(cust, p):
        pc = {"High": cfg.CHURN_COLOR, "Medium": cfg.WARN_COLOR, "Low": cfg.RETAIN_COLOR}[r["priority"]]
        st.markdown(
            f'<div class="rec" style="border-left:5px solid {pc}">'
            f'<div class="t"><span class="pill" style="background:{pc}">{r["priority"].upper()}</span>{r["title"]}</div>'
            f'<div class="a">{r["action"]}</div><div class="r">Why: {r["reason"]}</div></div>',
            unsafe_allow_html=True)


# =========================================================================== #
# PAGE 5 - BATCH SCORING
# =========================================================================== #
def page_batch() -> None:
    hero("Batch Scoring", "Score your whole customer base, find who to call first, and export the list.")
    source = st.radio("Customer data", ["Telco dataset (demo)", "Upload a CSV"], horizontal=True)

    if source == "Upload a CSV":
        st.caption("The file needs the Telco columns (e.g. `tenure`, `MonthlyCharges`, `Contract`, `InternetService`, ...). "
                   "`customerID` and `TotalCharges` are optional.")
        up = st.file_uploader("CSV file", type="csv")
        if up is None:
            with st.expander("Required columns"):
                st.code(", ".join(cfg.REQUIRED_INPUT_COLUMNS))
            return
        try:
            uploaded = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")
            return
        missing = dp.missing_input_columns(dp.normalize_categoricals(uploaded))
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            return
        frame = dp.prepare_for_model(uploaded)
        if cfg.ID_COL not in frame.columns:
            frame.insert(0, cfg.ID_COL, [f"ROW-{i + 1}" for i in range(len(frame))])
        bad = frame[cfg.FEATURES].isna().any(axis=1)
        if bad.any():
            st.warning(f"{int(bad.sum())} rows with missing values were skipped.")
            frame = frame[~bad]
        with st.spinner("Scoring ..."):
            scored = build_scored(frame, models[selected_model])
    else:
        st.info("ℹ️ About 80% of these customers were used to train the model, so scores are optimistic. "
                "Upload a CSV of new customers for realistic results.")
        scored = score_demo(selected_model, metrics["trained_at"])

    counts = scored["Risk Level"].value_counts()
    high = scored[scored["Risk Level"] == "High"]
    k = st.columns(4)
    k[0].markdown(kpi("Customers scored", f"{len(scored):,}", selected_model, "indigo"), unsafe_allow_html=True)
    k[1].markdown(kpi("High risk", f"{counts.get('High', 0):,}", f"{counts.get('High', 0) / len(scored):.0%} of base", "red"),
                  unsafe_allow_html=True)
    k[2].markdown(kpi("Medium risk", f"{counts.get('Medium', 0):,}", f"{counts.get('Medium', 0) / len(scored):.0%} of base", "amber"),
                  unsafe_allow_html=True)
    k[3].markdown(kpi("Monthly revenue at high risk", f"${high['MonthlyCharges'].sum():,.0f}",
                      f"${high['MonthlyCharges'].sum() * 12:,.0f} per year", "violet"), unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2)
    with a:
        show_chart(vz.probability_histogram(scored["Churn Probability"]), key="phist")
    with b:
        show_chart(vz.revenue_by_risk(scored), key="rrisk")

    st.subheader("🎯 Retention call list")
    f1, f2 = st.columns([2, 1])
    levels = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    top_n = f2.number_input("Max rows", min_value=10, max_value=5000, value=200, step=10)
    view = scored[scored["Risk Level"].isin(levels)].sort_values("Churn Probability", ascending=False).head(int(top_n))

    table = view[[cfg.ID_COL, "Contract", "InternetService", "tenure", "MonthlyCharges",
                  "Churn Probability", "Risk Level", "Suggested Action"]].copy()
    table["Churn Probability"] = (table["Churn Probability"] * 100).round(1)
    table = table.rename(columns={"Churn Probability": "Churn probability (%)", "tenure": "Tenure (mo)",
                                  "MonthlyCharges": "Monthly ($)"})
    show_df(table, column_config={
        "Churn probability (%)": st.column_config.ProgressColumn(
            "Churn probability (%)", min_value=0, max_value=100, format="%.1f"),
        "Monthly ($)": st.column_config.NumberColumn(format="%.2f"),
    })
    st.download_button("⬇️ Download call list (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="churn_call_list.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
{
    PAGES[0]: page_dashboard,
    PAGES[1]: page_explorer,
    PAGES[2]: page_models,
    PAGES[3]: page_predict,
    PAGES[4]: page_batch,
}[page]()
