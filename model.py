"""
model.py
Trains, evaluates, and persists three fraud-detection classifiers.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost                import XGBClassifier

from data_generator import generate_transactions, encode_categoricals

FEATURE_COLS = [
    "amount", "hour", "day_of_week", "merchant_category",
    "country", "prev_txn_mins", "card_age_days",
    "failed_attempts", "distance_km", "is_online",
]
TARGET_COL = "is_fraud"
MODEL_DIR  = Path("models")


# ── Public API ───────────────────────────────────────────────────────────────

def load_data(n_samples: int = 10_000):
    """Generate, encode, and split data. Returns X_train, X_test, y_train, y_test."""
    df = generate_transactions(n_samples)
    df = encode_categoricals(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE: balance training set
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)

    return (X_train_res, X_test_sc, y_train_res, y_test,
            X_train, X_test, scaler, df)


def build_models() -> dict:
    """Return a dict of untrained model instances."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=20, use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
    }


def train_all(n_samples: int = 10_000) -> dict:
    """Train all models; return a results dict for the dashboard."""
    (X_train, X_test, y_train, y_test,
     X_train_raw, X_test_raw, scaler, df) = load_data(n_samples)

    models   = build_models()
    results  = {}

    MODEL_DIR.mkdir(exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}…")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _  = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        cm           = confusion_matrix(y_test, y_pred)
        report       = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "model":        model,
            "y_pred":       y_pred,
            "y_prob":       y_prob,
            "roc_auc":      roc_auc_score(y_test, y_prob),
            "avg_precision":average_precision_score(y_test, y_prob),
            "fpr":          fpr,
            "tpr":          tpr,
            "precision":    prec,
            "recall":       rec,
            "cm":           cm,
            "report":       report,
            "feature_names":FEATURE_COLS,
        }

        # Feature importance (where available)
        if hasattr(model, "feature_importances_"):
            results[name]["feature_importance"] = dict(
                zip(FEATURE_COLS, model.feature_importances_)
            )

        # Persist model + scaler
        joblib.dump(model,  MODEL_DIR / f"{name.replace(' ','_')}.pkl")

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    results["_meta"] = {
        "X_test":       X_test,
        "y_test":       y_test,
        "X_test_raw":   X_test_raw,
        "scaler":       scaler,
        "df":           df,
        "feature_cols": FEATURE_COLS,
    }

    return results


def predict_single(transaction: dict, model_name: str = "XGBoost") -> dict:
    """
    Predict fraud for a single transaction dict.
    Loads the persisted model and scaler from disk.
    Returns {'fraud': bool, 'probability': float, 'risk_level': str}
    """
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model  = joblib.load(MODEL_DIR / f"{model_name.replace(' ','_')}.pkl")

    row = pd.DataFrame([transaction])[FEATURE_COLS]
    row_sc = scaler.transform(row)

    prob  = model.predict_proba(row_sc)[0][1]
    fraud = prob >= 0.5

    if   prob < 0.3:  risk = "Low"
    elif prob < 0.6:  risk = "Medium"
    elif prob < 0.8:  risk = "High"
    else:             risk = "Critical"

    return {"fraud": fraud, "probability": round(float(prob), 4), "risk_level": risk}


if __name__ == "__main__":
    results = train_all()
    for name, res in results.items():
        if name.startswith("_"):
            continue
        print(f"\n{'='*40}")
        print(f" {name}")
        print(f"  ROC-AUC : {res['roc_auc']:.4f}")
        print(f"  Avg-Prec: {res['avg_precision']:.4f}")
        print(classification_report(
            results["_meta"]["y_test"], res["y_pred"]
        ))