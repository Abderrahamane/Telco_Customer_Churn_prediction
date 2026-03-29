from fastapi.testclient import TestClient

from api.main import app
from src.churn_app_utils import (
    build_training_matrix,
    default_customer_input,
    get_expected_columns,
    load_dataset,
    load_model,
    predict_local,
)


def main():
    model, model_path = load_model()
    dataset_df, dataset_path = load_dataset()
    x_matrix, _, scaler = build_training_matrix(dataset_df)
    expected_columns = get_expected_columns(model, x_matrix.columns)

    payload = default_customer_input()
    local_result = predict_local(model, scaler, expected_columns, payload)

    client = TestClient(app)
    api_response = client.post("/predict", json=payload)
    api_response.raise_for_status()
    api_result = api_response.json()

    print(f"Model path: {model_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Local prediction: {local_result['prediction']}")
    print(f"Local churn probability: {local_result['churn_probability']:.4f}")
    print(f"API prediction: {api_result['prediction']}")
    print(f"API churn probability: {api_result['churn_probability']:.4f}")

    if local_result["prediction"] != api_result["prediction"]:
        raise AssertionError("Local and API predictions do not match.")


if __name__ == "__main__":
    main()

