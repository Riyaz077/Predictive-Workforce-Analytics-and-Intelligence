from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .validation import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, validate_frame

def build_pipeline():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES)
    ])
    classifier = RandomForestClassifier(
        n_estimators=250, max_depth=9, min_samples_leaf=4,
        class_weight="balanced", random_state=42
    )
    return Pipeline([("preprocess", prep), ("classifier", classifier)])

def train(df):
    clean = validate_frame(df)
    X, y = clean[FEATURES], clean[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob))
    }
    return model, metrics

def save_model(model, path="models/talentiq_model.joblib"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path

def score_employee(model, employee: dict):
    frame = pd.DataFrame([employee], columns=FEATURES)
    probability = float(model.predict_proba(frame)[0][1])
    if probability >= .65:
        band = "High"
    elif probability >= .35:
        band = "Medium"
    else:
        band = "Low"
    return {
        "attrition_probability": probability,
        "illustrative_risk_band": band,
        "notice": "Synthetic-data demonstration only; do not use for employment decisions."
    }
