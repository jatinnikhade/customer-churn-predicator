"""Command-line training script.

Usage
-----
    python train.py            # fast: sensible defaults
    python train.py --tune     # GridSearchCV hyper-parameter tuning (slower)
"""
import argparse

from src import config as cfg
from src.model_training import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn prediction models.")
    parser.add_argument("--tune", action="store_true", help="run GridSearchCV before the final fit")
    args = parser.parse_args()

    summary = train_models(tune=args.tune, verbose=True)
    print(f"\nBest model: {summary['best_model']}")
    print(f"Artifacts saved to: {cfg.MODELS_DIR}")


if __name__ == "__main__":
    main()
