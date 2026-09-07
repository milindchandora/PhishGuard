"""
train_model.py
----------------
Full training pipeline for the phishing-detection model.

Steps:
 1. Load dataset (data/urls_dataset.csv -- columns: url,label)
 2. Clean data / handle missing values
 3. Extract URL features (via feature_extraction.py -- SAME module used
    at prediction time in app.py, so train/serve features always match)
 4. Train/test split
 5. Train several candidate classifiers
 6. Evaluate each on the held-out test set
 7. Select the best model by F1 score
 8. Save the trained model + scaler + evaluation report to model/

Run:
    python ml/train_model.py
"""

import os
import json
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import joblib
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from feature_extraction import extract_features, FEATURE_NAMES  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "urls_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run 'python ml/build_sample_dataset.py' to generate a demo "
            "dataset, or place a real dataset (columns: url,label) at "
            "this path."
        )
    df = pd.read_csv(path)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'url' and 'label' columns.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 0]
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["url"])
    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for url in df["url"]:
        try:
            rows.append(extract_features(url))
        except Exception:
            # If a single malformed URL fails feature extraction, fill
            # with zeros rather than crashing the whole pipeline.
            rows.append({name: 0 for name in FEATURE_NAMES})
    feat_df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    feat_df = feat_df.fillna(0)
    return feat_df


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def plot_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate", "Phishing"])
    ax.set_yticklabels(["Legitimate", "Phishing"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_model_comparison(results: dict, out_path: str):
    names = list(results.keys())
    f1s = [results[n]["f1_score"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, f1s, color="#2563eb")
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison (F1 Score)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                 ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    print("Step 1/8: Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"  Loaded {len(df)} rows.")

    print("Step 2/8: Cleaning dataset...")
    df = clean_dataset(df)
    print(f"  {len(df)} rows after cleaning/deduplication.")

    print("Step 3/8: Extracting URL features...")
    X = build_feature_matrix(df)
    y = df["label"].values

    print("Step 4/8: Train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Step 5/8: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Step 6/8: Training and evaluating candidate models...")
    models = get_candidate_models()
    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        results[name] = metrics
        trained_models[name] = model
        print(f"  {name:24s} | Acc={metrics['accuracy']:.3f} "
              f"Prec={metrics['precision']:.3f} Rec={metrics['recall']:.3f} "
              f"F1={metrics['f1_score']:.3f}")

    print("Step 7/8: Selecting best model by F1 score...")
    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = trained_models[best_name]
    print(f"  Best model: {best_name} (F1={results[best_name]['f1_score']})")

    print("Step 8/8: Saving model, scaler, and evaluation report...")
    joblib.dump(best_model, os.path.join(MODEL_DIR, "phishing_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_names.pkl"))

    report = {
        "best_model": best_name,
        "feature_names": FEATURE_NAMES,
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "results": {
            name: {k: v for k, v in m.items() if k != "classification_report"}
            for name, m in results.items()
        },
    }
    with open(os.path.join(MODEL_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save a human-readable text report too
    with open(os.path.join(MODEL_DIR, "evaluation_report.txt"), "w") as f:
        for name, m in results.items():
            f.write(f"=== {name} ===\n")
            f.write(m["classification_report"])
            f.write(f"\nConfusion Matrix: {m['confusion_matrix']}\n\n")

    plot_confusion_matrix(
        results[best_name]["confusion_matrix"], best_name,
        os.path.join(MODEL_DIR, "confusion_matrix.png"),
    )
    plot_model_comparison(results, os.path.join(MODEL_DIR, "model_comparison.png"))

    print("\nDone. Artifacts saved in model/:")
    print("  phishing_model.pkl, scaler.pkl, feature_names.pkl")
    print("  evaluation_report.json, evaluation_report.txt")
    print("  confusion_matrix.png, model_comparison.png")


if __name__ == "__main__":
    main()
