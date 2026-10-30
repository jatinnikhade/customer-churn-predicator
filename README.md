# 📉 Customer Churn Prediction System

An end-to-end machine-learning project that analyses customer behaviour in the **Telco Customer Churn** dataset, predicts which customers are likely to leave, and recommends how to keep them - all wrapped in an interactive **Streamlit** dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Page | What it does |
|---|---|
| **📊 Dashboard** | KPI cards (churn rate, revenue lost), churn by contract / tenure / internet service / payment method, interactive filters and auto-generated business insights |
| **🔍 Data Explorer** | Data-cleaning report, churn analysis for every feature, distributions, box plots, correlation matrix and a contract × internet heat-map |
| **🤖 Model Performance** | Logistic Regression vs Random Forest: accuracy, precision, recall, F1, ROC-AUC, 5-fold CV, ROC curves, confusion matrices, feature importance. One-click retraining (optional GridSearchCV) |
| **🎯 Predict a Customer** | Fill in a customer profile → churn probability gauge, risk band, revenue at risk, per-customer **risk drivers**, and a prioritised **retention plan** |
| **📋 Batch Scoring** | Score the whole base or upload your own CSV, filter by risk level and download a ranked retention call-list |

## 🗂️ Project structure

```
churn-prediction-system/
├── app.py                      # Streamlit dashboard (5 pages)
├── train.py                    # CLI: train + evaluate + save models
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   └── Telco-Customer-Churn.csv      # 7,043 customers x 21 columns
├── models/                     # generated: *.joblib + metrics.json
├── src/
│   ├── config.py               # paths, schema, constants, palette
│   ├── data_processing.py      # loading, cleaning, feature engineering
│   ├── model_training.py       # pipelines, tuning, evaluation, persistence
│   ├── predict.py              # model loading + scoring + risk bands
│   ├── explain.py              # feature importance + per-customer drivers
│   ├── recommendations.py      # rule-based retention playbook
│   └── visuals.py              # Plotly figure factories
├── tests/
│   └── test_pipeline.py        # pytest suite
└── .streamlit/config.toml      # theme
```

## 🚀 Quick start

```bash
# 1. Clone
git clone https://github.com/<your-username>/churn-prediction-system.git
cd churn-prediction-system

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

On first launch the app trains both models automatically (~20 seconds) and caches them in `models/`.
To train from the command line instead:

```bash
python train.py            # fast, sensible defaults
python train.py --tune     # GridSearchCV hyper-parameter search (slower)
```

Run the tests:

```bash
pytest
```

## 🧪 Methodology

### 1. Data cleaning
- `TotalCharges` is stored as text because **11 rows hold a blank space**. These are brand-new customers (`tenure = 0`) who have not been billed, so they are set to `0`.
- `SeniorCitizen` (0/1) is converted to No/Yes to match the other flags; whitespace is trimmed; duplicate IDs are dropped.
- Target `Churn` is encoded Yes → 1 / No → 0 (**26.5 %** of customers churned - an imbalanced problem).

### 2. Feature engineering
- `NumServices` - number of add-on services (security, backup, device protection, tech support, streaming TV/movies).

### 3. Modelling
Preprocessing and model live in a single scikit-learn `Pipeline`, so raw customer rows can be scored directly and there is **no train/test leakage** (the scaler and encoder are fit on training data only).

| Step | Detail |
|---|---|
| Split | 80 / 20 stratified train-test |
| Numeric features | `StandardScaler` |
| Categorical features | `OneHotEncoder(handle_unknown="ignore")` |
| Imbalance | `class_weight="balanced"` (LR) / `"balanced_subsample"` (RF) |
| Models | Logistic Regression, Random Forest (300 trees, depth 10) |
| Validation | 5-fold stratified CV ROC-AUC + held-out test set |
| Tuning (optional) | `GridSearchCV`, 3-fold, ROC-AUC |

### 4. Results (held-out test set, default settings)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | **0.783** | 0.614 | 0.842 |
| Random Forest | **0.767** | **0.543** | 0.757 | **0.632** | 0.842 |

Both models rank customers equally well (AUC 0.84). Class weighting deliberately favours **recall** - catching churners matters more than a few extra retention offers. Exact values vary slightly with library versions; the *Model Performance* page always shows the numbers from your latest training run.

### 5. Explainability
- **Global:** Random Forest importances (summed back to the original features) and Logistic Regression coefficients.
- **Per customer:** each feature's contribution = (customer value − average customer) × coefficient, shown as a bar chart of what pushes the risk up or down.

### 6. Risk bands & recommendations
| Churn probability | Risk |
|---|---|
| < 30 % | 🟢 Low |
| 30 – 60 % | 🟠 Medium |
| ≥ 60 % | 🔴 High |

A rule-based playbook turns a customer's profile and risk into prioritised actions (longer-term contract offers, loyalty discounts, tech-support bundles, auto-pay incentives, onboarding programmes, senior-friendly plans ...). Edit `src/recommendations.py` to match your own business rules.

## 🔑 Key findings from the data

- **Contract type is the strongest signal:** month-to-month customers churn at **42.7 %** vs **2.8 %** on two-year contracts.
- **The first year is critical:** customers with ≤ 12 months of tenure churn at **47.4 %** vs 17.1 % afterwards.
- **Fibre optic customers churn more:** 41.9 % vs 19.0 % for DSL - a likely price/quality issue.
- **Electronic-check payers** churn at 45.3 % vs 16.0 % for automatic payers.
- **Support protects:** internet customers without tech support churn at 41.6 % vs 15.2 % with it.
- Customers who left were paying about **$139K per month** in recurring revenue.

## ☁️ Deploy for free

1. Push this repo to GitHub (the dataset is only ~1 MB, so it's included).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo and choose `app.py` as the entry point.
3. Done - the app trains its models on first start.

## 📸 Screenshots

Add your own screenshots to a `docs/` folder and reference them here, e.g. `![Dashboard](docs/dashboard.png)`.

## 🛣️ Ideas for extension

- Add XGBoost / LightGBM and compare against the current models
- SHAP values for richer explanations
- Probability calibration and cost-based threshold tuning (cost of a retention offer vs value of a saved customer)
- Customer lifetime value (CLV) weighting in the call-list ranking
- Dockerfile and CI workflow running `pytest`

## 📚 Data source

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - an IBM sample dataset (7,043 customers, 21 columns) describing a fictional telecom company.

## 📄 License

Released under the [MIT License](LICENSE).

## 👤 Author

**Jatin Pramod Nikhade** - Data Science student
