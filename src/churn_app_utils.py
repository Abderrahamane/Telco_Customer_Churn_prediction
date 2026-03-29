from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

TENURE_BINS = [0, 10, 50, 72]
TENURE_LABELS = ["new", "medium", "old"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_model_path(project_root: Path | None = None) -> Path:
    root = project_root or get_project_root()
    candidates = [
        root / "logreg_model.pkl",
        root / "models" / "logreg_model.pkl",
        root / "models" / "logistic_model.pkl",
        root / "notebooks" / "logreg_model.pkl",
        root / "notebooks" / "logistic_model.pkl",
        root / "notebooks" / "models" / "logistic_model.pkl",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find a trained model file. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def find_dataset_path(project_root: Path | None = None) -> Path:
    root = project_root or get_project_root()
    candidates = [
        root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        root / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find Telco dataset CSV. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_model(model_path: Path | None = None):
    path = model_path or find_model_path()
    return joblib.load(path), path


def load_dataset(dataset_path: Path | None = None) -> Tuple[pd.DataFrame, Path]:
    path = dataset_path or find_dataset_path()
    return pd.read_csv(path), path


def _prepare_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    if "customerID" in clean_df.columns:
        clean_df = clean_df.drop(columns=["customerID"])

    # Convert numeric columns to numeric dtype exactly like notebook steps.
    for col in NUMERIC_COLUMNS:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Keep only valid rows for model features.
    clean_df = clean_df.dropna(subset=NUMERIC_COLUMNS)

    # Feature engineering from notebook.
    clean_df["tenure_group"] = pd.cut(
        clean_df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
    )

    return clean_df


def build_training_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, MinMaxScaler]:
    clean_df = _prepare_raw_dataframe(df)

    if "Churn" not in clean_df.columns:
        raise ValueError("Dataset must contain 'Churn' column for evaluation.")

    y = clean_df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    x_raw = clean_df.drop(columns=["Churn"])

    categorical_columns = [col for col in x_raw.columns if col not in NUMERIC_COLUMNS]
    x_encoded = pd.get_dummies(x_raw, columns=categorical_columns, drop_first=True)

    scaler = MinMaxScaler()
    x_encoded[NUMERIC_COLUMNS] = scaler.fit_transform(x_encoded[NUMERIC_COLUMNS])

    return x_encoded, y, scaler


def preprocess_single_input(raw_input_df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    clean_df = _prepare_raw_dataframe(raw_input_df)
    clean_df = clean_df.drop(columns=["Churn"], errors="ignore")

    categorical_columns = [col for col in clean_df.columns if col not in NUMERIC_COLUMNS]
    x_encoded = pd.get_dummies(clean_df, columns=categorical_columns, drop_first=True)

    x_encoded[NUMERIC_COLUMNS] = scaler.transform(x_encoded[NUMERIC_COLUMNS])
    return x_encoded


def build_raw_input_dataframe(payload: Mapping[str, Any]) -> pd.DataFrame:
    # Start from defaults so API payloads can omit optional fields safely.
    merged_payload = default_customer_input()
    merged_payload.update(dict(payload))

    # Keep schema stable and apply minimal cleaning before preprocessing.
    row = {feature: merged_payload.get(feature) for feature in RAW_FEATURE_COLUMNS}

    for col in NUMERIC_COLUMNS:
        row[col] = pd.to_numeric(row[col], errors="coerce")

    # Fallback to defaults for invalid or missing numeric values.
    defaults = default_customer_input()
    for col in NUMERIC_COLUMNS:
        if pd.isna(row[col]):
            row[col] = float(defaults[col])

    row["SeniorCitizen"] = int(row["SeniorCitizen"])
    row["tenure"] = float(row["tenure"])
    row["MonthlyCharges"] = float(row["MonthlyCharges"])
    row["TotalCharges"] = float(row["TotalCharges"])

    return pd.DataFrame([row])


def get_expected_columns(model, fallback_columns: Iterable[str]) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_columns)


def align_features(x: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    return x.reindex(columns=list(expected_columns), fill_value=0.0)


def predict_local(
    model,
    scaler: MinMaxScaler,
    expected_columns: Iterable[str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_input_df = build_raw_input_dataframe(payload)
    processed_input = preprocess_single_input(raw_input_df, scaler)
    aligned_input = align_features(processed_input, expected_columns)

    prediction = int(model.predict(aligned_input)[0])
    probability = float(model.predict_proba(aligned_input)[0, 1])

    return {
        "prediction": prediction,
        "prediction_label": "Yes" if prediction == 1 else "No",
        "churn_probability": probability,
        "processed_input": aligned_input,
        "raw_input": raw_input_df,
    }


def logistic_feature_importance(model, feature_names: Iterable[str], top_n: int = 15) -> pd.DataFrame:
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    names = list(feature_names)
    coefficients = model.coef_[0]

    # Guard against occasional mismatch between model metadata and provided names.
    size = min(len(names), len(coefficients))
    importance_df = pd.DataFrame(
        {
            "feature": names[:size],
            "coefficient": coefficients[:size],
        }
    )
    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
    importance_df = importance_df.sort_values("abs_coefficient", ascending=False).head(top_n)
    return importance_df.reset_index(drop=True)


def default_customer_input() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 1020.0,
    }

