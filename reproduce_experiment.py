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

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - xgboost is optional
    XGBClassifier = None


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the prepared dataset.

    The original project uses a pickle file containing all processed
    birth certificate records with a ``preterm`` target column.
    """
    return pd.read_pickle(path)


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
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=8),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce thesis experiment")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to preprocessed pickle file containing the data.",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    X_train, y_train, _, _, X_test, y_test = split_and_scale(df)

    models = train_models(X_train, y_train)

    print("Model\tAUC\tTPR at 10% FPR")
    for name, model in models.items():
        auc, tpr_at_10 = evaluate(model, X_test, y_test)
        print(f"{name}\t{auc:.4f}\t{tpr_at_10:.2f}%")


if __name__ == "__main__":
    main()
