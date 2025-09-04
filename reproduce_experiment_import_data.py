import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset


try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - xgboost is optional
    XGBClassifier = None


def load_dataset_from_huggingface(sample_frac: float = 0.1) -> pd.DataFrame:
    dataset = load_dataset("SushantGautam/nvss-birth-records-usa-2016-2020")
    df = dataset["train"].to_pandas()

    # sample only a fraction of the data (default = 10%)
    df = df.sample(frac=sample_frac, random_state=8).reset_index(drop=True)

    return df


def split_and_scale(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data and apply standard scaling with random undersampling.

    Returns ``X_train, y_train, X_val, y_val, X_test, y_test``.
    """
    drop_cols = [
        "OEGest_R10",
        "preterm",
        "ILLB_R11",
        "PWgt_R",
        "BMI",
        "ME_ROUT",
        "ME_TRIAL",
        "LD_INDL",
        "PRECARE",
    ]
    X = df.drop(columns=drop_cols)
    y = df["preterm"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=8
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=8
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    rus = RandomUnderSampler(random_state=8)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_models(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Train all models used in the thesis."""
    models: Dict[str, object] = {
        "Log. Regression": LogisticRegression(max_iter=1000, random_state=8),
        "Random Forest": RandomForestClassifier(random_state=8, n_jobs=-1),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=8),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=8,
        )

    for model in models.values():
        model.fit(X, y)

    return models


def evaluate(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return AUC and TPR at 10% FPR."""
    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    tpr_at_10 = tpr[np.argmin(np.abs(fpr - 0.1))]
    return auc, tpr_at_10 * 100


def run_partitioned_experiment(df: pd.DataFrame, partition_col: str = "PARTITION") -> None:
    """Train and evaluate models for each data partition.

    At the end, aggregate evaluation results (AUC and TPR@10%FPR) across all partitions.
    """

    partitions = df[partition_col].unique()
    combined_preds = {}  # model_name -> list of predicted probabilities
    combined_labels = []  # true labels (same for all models)

    for partition in partitions:
        subset = df[df[partition_col] == partition].drop(columns=[partition_col])
        X_train, y_train, _, _, X_test, y_test = split_and_scale(subset)
        models = train_models(X_train, y_train)

        print(f"Partition {partition}")
        print("Model\t\tAUC\tTPR at 10% FPR")
        for name, model in models.items():
            prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, prob)
            fpr, tpr, _ = roc_curve(y_test, prob)
            tpr_at_10 = tpr[np.argmin(np.abs(fpr - 0.1))]

            print(f"{name}\t{auc:.4f}\t{tpr_at_10 * 100:.2f}%")

            if name not in combined_preds:
                combined_preds[name] = []
            combined_preds[name].extend(prob)

        combined_labels.extend(y_test)

        print()

    # Print combined results
    print("Combined Evaluation (All Partitions)")
    print("Model\tAUC\tTPR at 10% FPR")
    y_all = np.array(combined_labels)
    for name, prob_list in combined_preds.items():
        prob = np.array(prob_list)
        auc = roc_auc_score(y_all, prob)
        fpr, tpr, _ = roc_curve(y_all, prob)
        tpr_at_10 = tpr[np.argmin(np.abs(fpr - 0.1))]
        print(f"{name}\t{auc:.4f}\t{tpr_at_10 * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce thesis experiment")
    # parser.add_argument(
    #     "--data",
    #     type=Path,
    #     required=True,
    #     help="Path to preprocessed pickle file containing the data.",
    # )
    parser.add_argument(
        "--with_cluster",
        action="store_true",
        help="Run experiment separately for each data partition.",
    )
    args = parser.parse_args()

    print("📥 Loading dataset from Hugging Face...")
    df = load_dataset_from_huggingface()
    print("✅ Dataset loaded with shape:", df.shape)

  
    if args.with_cluster:
        run_partitioned_experiment(df)
    else:
        X_train, y_train, _, _, X_test, y_test = split_and_scale(df)
        models = train_models(X_train, y_train)

        print("Model\t\tAUC\tTPR at 10% FPR")
        for name, model in models.items():
            auc, tpr_at_10 = evaluate(model, X_test, y_test)
            print(f"{name}\t{auc:.4f}\t{tpr_at_10:.2f}%")


if __name__ == "__main__":
    main()
