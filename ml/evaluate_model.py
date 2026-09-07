"""
evaluate_model.py
-------------------
Standalone utility to re-evaluate the currently saved model against the
dataset (useful after retraining, or to double check saved artifacts
without re-running the full training pipeline).

Run:
    python ml/evaluate_model.py
"""

import os
import sys
import json

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.append(os.path.dirname(__file__))
from feature_extraction import extract_features, FEATURE_NAMES  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "urls_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")


def main():
    model_path = os.path.join(MODEL_DIR, "phishing_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("No saved model found. Run 'python ml/train_model.py' first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(DATA_PATH).dropna(subset=["url", "label"])
    rows = [extract_features(u) for u in df["url"]]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES).fillna(0)
    y = df["label"].astype(int).values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
