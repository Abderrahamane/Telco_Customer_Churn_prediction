from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Make project modules importable when Streamlit runs from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.churn_app_utils import (  # noqa: E402
    align_features,
    build_training_matrix,
    build_raw_input_dataframe,
    default_customer_input,
    get_expected_columns,
    logistic_feature_importance,
    load_dataset,
    load_model,
    predict_local,
)


@st.cache_resource
def get_model_and_path():
    return load_model()


@st.cache_data
def get_dataset_and_path():
    return load_dataset()


@st.cache_resource
def get_training_artifacts():
    dataset_df, _ = get_dataset_and_path()
    x_matrix, y, scaler = build_training_matrix(dataset_df)
    return x_matrix, y, scaler


def render_input_form() -> dict[str, Any]:
    defaults = default_customer_input()

    st.subheader("Customer Input")
    st.caption("Enter customer details to run churn prediction.")

    with st.form("prediction_form"):
        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], index=defaults["SeniorCitizen"])
            partner = st.selectbox("Partner", ["Yes", "No"], index=0)
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=defaults["tenure"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"],
                index=1,
            )

        with col_2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
            online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"], index=1)
            online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"], index=2)
            device_protection = st.selectbox(
                "Device Protection",
                ["No internet service", "No", "Yes"],
                index=1,
            )
            tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"], index=1)
            streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"], index=2)
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No internet service", "No", "Yes"],
                index=2,
            )

        with col_3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=0,
            )
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=float(defaults["MonthlyCharges"]),
                step=0.1,
            )
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=float(defaults["TotalCharges"]),
                step=0.1,
            )

        submitted = st.form_submit_button("Predict Churn")

    if not submitted:
        return {}

    return {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": float(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }


def call_prediction_api(api_url: str, payload: dict[str, Any], timeout_seconds: int = 10) -> dict[str, Any]:
    response = requests.post(
        api_url.rstrip("/") + "/predict",
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def render_prediction_section(model, scaler, expected_columns: list[str], x_matrix: pd.DataFrame):
    st.header("Prediction")
    prediction_mode = st.radio(
        "Inference mode",
        ["Local model", "REST API"],
        horizontal=True,
        help="Use local inference or call the FastAPI endpoint.",
    )

    api_base_url = "http://127.0.0.1:8000"
    if prediction_mode == "REST API":
        api_base_url = st.text_input("API base URL", value="http://127.0.0.1:8000")

    payload = render_input_form()
    if not payload:
        st.info("Submit the form to get prediction output.")
        return

    raw_input_df = build_raw_input_dataframe(payload)
    processed_input = predict_local(model, scaler, expected_columns, payload)["processed_input"]

    prediction = None
    probability = None
    source = ""
    try:
        if prediction_mode == "REST API":
            api_result = call_prediction_api(api_base_url, payload)
            prediction = int(api_result["prediction"])
            probability = float(api_result["churn_probability"])
            source = "API"
        else:
            local_result = predict_local(model, scaler, expected_columns, payload)
            prediction = local_result["prediction"]
            probability = local_result["churn_probability"]
            source = "Local"
    except Exception as error:
        st.warning(f"API request failed ({error}). Falling back to local inference.")
        local_result = predict_local(model, scaler, expected_columns, payload)
        prediction = local_result["prediction"]
        probability = local_result["churn_probability"]
        source = "Local fallback"

    st.subheader("Prediction Result")
    if prediction == 1:
        st.markdown("**Customer is likely to churn.**")
    else:
        st.markdown("**Customer is not likely to churn.**")

    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Churn probability", f"{probability:.2%}")
    metric_col_2.metric("Inference source", source)

    with st.expander("Show processed model input"):
        st.dataframe(processed_input, use_container_width=True)

    history_row = raw_input_df.copy()
    history_row["prediction"] = "Yes" if prediction == 1 else "No"
    history_row["churn_probability"] = probability
    history_row["inference_source"] = source
    history = st.session_state.get("prediction_history", pd.DataFrame())
    st.session_state["prediction_history"] = pd.concat([history, history_row], ignore_index=True)

    if not st.session_state["prediction_history"].empty:
        st.download_button(
            label="Export predictions to CSV",
            data=st.session_state["prediction_history"].to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions_history.csv",
            mime="text/csv",
        )


def render_dashboard_section(dataset_df: pd.DataFrame, model, expected_columns: list[str]):
    st.header("Dashboard")

    vis_col_1, vis_col_2 = st.columns(2)

    with vis_col_1:
        st.subheader("Churn Distribution")
        churn_counts = dataset_df["Churn"].value_counts().rename_axis("Churn").reset_index(name="count")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=churn_counts, x="Churn", y="count", palette="viridis", ax=ax)
        ax.set_title("Customer Churn Count")
        st.pyplot(fig)

    with vis_col_2:
        st.subheader("Monthly Charges Distribution")
        charges = pd.to_numeric(dataset_df["MonthlyCharges"], errors="coerce")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(charges.dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Monthly Charges Histogram")
        st.pyplot(fig)

    st.subheader("Logistic Regression Feature Importance")
    importance_df = logistic_feature_importance(model, expected_columns, top_n=15)
    if importance_df.empty:
        st.info("Feature importance is unavailable for this model artifact.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importance_df, x="coefficient", y="feature", palette="coolwarm", ax=ax)
    ax.set_title("Top Features by Absolute Coefficient")
    st.pyplot(fig)
    st.dataframe(importance_df, use_container_width=True)


def render_model_comparison_section(model, x_matrix: pd.DataFrame, y: pd.Series, expected_columns: list[str]):
    st.header("Model Comparison")
    st.caption("Compare Logistic Regression against Random Forest on the same split.")

    x_aligned = align_features(x_matrix, expected_columns)
    x_train, x_test, y_train, y_test = train_test_split(
        x_aligned,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logistic_pred = model.predict(x_test)
    logistic_proba = model.predict_proba(x_test)[:, 1]

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)
    rf_proba = rf_model.predict_proba(x_test)[:, 1]

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Accuracy": accuracy_score(y_test, logistic_pred),
                "F1-score": f1_score(y_test, logistic_pred),
                "ROC-AUC": roc_auc_score(y_test, logistic_proba),
            },
            {
                "Model": "Random Forest",
                "Accuracy": accuracy_score(y_test, rf_pred),
                "F1-score": f1_score(y_test, rf_pred),
                "ROC-AUC": roc_auc_score(y_test, rf_proba),
            },
        ]
    )
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Confusion Matrix (Logistic Regression)")
    cm = confusion_matrix(y_test, logistic_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def render_api_info_section(default_payload: dict[str, Any]):
    st.header("API Info")
    st.write("Use the FastAPI service to score customer payloads from any client.")

    st.code(
        "uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload",
        language="bash",
    )

    st.write("Endpoint: `POST /predict`")
    st.json(default_payload)

    st.code(
        """curl -X POST "http://127.0.0.1:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":85.0,"TotalCharges":1020.0}'""",
        language="bash",
    )


def render_business_insight_section():
    st.header("Business Insight")
    st.markdown(
        """
        Telecom operators such as Djezzy, Ooredoo, and Mobilis can use this solution to prioritize retention offers.

        - Detect high-risk customers before contract renewal windows.
        - Trigger proactive campaigns (discounts, loyalty bundles, service follow-up).
        - Focus call-center effort on high-probability churn segments.
        - Track monthly churn-risk trends to improve revenue forecasting.
        """
    )


def main():
    st.set_page_config(page_title="Telco Churn Predictor", page_icon=":chart_with_downwards_trend:", layout="wide")

    st.title("Telecom Customer Churn Prediction")
    st.caption("Professional ML app with prediction, analytics, API integration, and business insights")

    try:
        model, model_path = get_model_and_path()
        dataset_df, dataset_path = get_dataset_and_path()
        x_matrix, y, scaler = get_training_artifacts()
    except Exception as error:
        st.error(f"Startup error: {error}")
        st.stop()

    expected_columns = get_expected_columns(model, x_matrix.columns)

    with st.expander("Loaded artifacts", expanded=False):
        st.write(f"Model: `{model_path}`")
        st.write(f"Dataset: `{dataset_path}`")
        st.write(f"Features expected by model: **{len(expected_columns)}**")
        st.write(f"Rows available for optional evaluation: **{len(dataset_df)}**")

    section = st.sidebar.radio(
        "Navigation",
        ["Prediction", "Dashboard", "Model Comparison", "API Info", "Business Insight"],
    )

    if section == "Prediction":
        render_prediction_section(model, scaler, expected_columns, x_matrix)
    elif section == "Dashboard":
        render_dashboard_section(dataset_df, model, expected_columns)
    elif section == "Model Comparison":
        render_model_comparison_section(model, x_matrix, y, expected_columns)
    elif section == "API Info":
        render_api_info_section(default_customer_input())
    else:
        render_business_insight_section()


if __name__ == "__main__":
    main()

