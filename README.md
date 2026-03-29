# Telco Customer Churn Prediction App

Professional end-to-end churn project with:

- Streamlit frontend (prediction + analytics + business insight)
- FastAPI prediction service (REST endpoint)
- Shared preprocessing and inference utilities

## Core Features

### 1) Prediction Section

- Input all required customer features from UI controls
- Preprocessing inside app/service:
  - numeric conversion (`tenure`, `MonthlyCharges`, `TotalCharges`)
  - missing value fallback for invalid numeric payloads
  - feature engineering (`tenure_group`)
  - one-hot encoding for categorical variables
  - MinMax scaling for numeric features
  - final feature alignment to model schema
- Output:
  - churn label (`Yes` / `No`)
  - churn probability
- Supports both **Local model** inference and **REST API** inference
- Optional export of prediction history as CSV

### 2) Dashboard Section

- Churn distribution plot
- Monthly charges histogram
- Logistic Regression feature-importance chart (coefficients)

### 3) Model Comparison Section

- Compares Logistic Regression vs Random Forest
- Metrics:
  - Accuracy
  - F1-score
  - ROC-AUC
- Confusion matrix visualization

### 4) API Section

- FastAPI app with:
  - `GET /`
  - `GET /health`
  - `POST /predict`
- Accepts JSON payload and returns prediction + churn probability

### 5) Business Insight Section

- Explains how telecom operators (Djezzy, Ooredoo, Mobilis) can use churn scoring to reduce churn and improve retention ROI.

## Project Structure

- `streamlit_app.py` - Streamlit Cloud entrypoint
- `app/streamlit_app.py` - full Streamlit UI
- `api/main.py` - FastAPI app
- `src/churn_app_utils.py` - shared preprocessing and inference helpers
- `src/smoke_test.py` - local smoke test (local + API parity)

## Local Setup

```bash
pip install -r requirements.txt
python -m src.smoke_test
```

## Run Streamlit App

```bash
streamlit run streamlit_app.py
```

## Run FastAPI Service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs:

- `http://127.0.0.1:8000/docs`

## API Request Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":85.0,"TotalCharges":1020.0}'
```

## Streamlit Cloud Deployment

1. Push repo to GitHub.
2. Create app in Streamlit Cloud.
3. Set entrypoint to `streamlit_app.py`.
4. Ensure model + dataset are committed.
5. Deploy.

## Model Artifact Discovery

The app checks these model locations in order:

1. `logreg_model.pkl`
2. `models/logreg_model.pkl`
3. `models/logistic_model.pkl`
4. `notebooks/logreg_model.pkl`
5. `notebooks/logistic_model.pkl`
6. `notebooks/models/logistic_model.pkl`

If your trained model has a different name, place it in one of the paths above.

