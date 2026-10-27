"""
Comprehensive Integration Tests for FastAPI Endpoints.
Uses fastapi.testclient.TestClient to test all REST API routes.
"""

import pytest
from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_list_datasets():
    res = client.get("/api/datasets")
    assert res.status_code == 200
    datasets = res.json()
    assert len(datasets) == 3
    ids = [d["id"] for d in datasets]
    assert "50_startups" in ids
    assert "student_performance" in ids
    assert "flood_probability" in ids


def test_dataset_overview_50_startups():
    res = client.get("/api/datasets/50_startups/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["n_rows"] == 50
    assert data["target_column"] == "Profit"
    assert len(data["available_features"]) == 4
    assert not data["is_compliant_14_39"]  # 4 features is < 14


def test_dataset_overview_student_performance():
    res = client.get("/api/datasets/student_performance/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["n_rows"] == 6607
    assert data["target_column"] == "Exam_Score"
    assert len(data["available_features"]) == 19
    assert data["is_compliant_14_39"]  # 19 features satisfies 14~39


def test_analysis_train():
    req_body = {
        "dataset_id": "50_startups",
        "test_size": 0.20,
        "random_state": 42,
    }
    res = client.post("/api/analysis/train", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert data["model_type"] == "LinearRegression"
    assert "intercept" in data
    assert len(data["coefficients"]) > 0


def test_analysis_feature_selection():
    req_body = {
        "dataset_id": "50_startups",
        "k_features": 3,
        "test_size": 0.20,
        "cv_folds": 3,
        "fast_mode": True,
    }
    res = client.post("/api/analysis/feature-selection", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert "comparison_table" in data
    assert len(data["comparison_table"]) == 6
    methods = [m["method"] for m in data["comparison_table"]]
    assert "Baseline (All Features)" in methods


def test_analysis_evaluation():
    req_body = {
        "dataset_id": "50_startups",
        "test_size": 0.20,
        "random_state": 42,
    }
    res = client.post("/api/analysis/evaluation", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert "train_metrics" in data
    assert "test_metrics" in data
    assert "actual_vs_predicted_sample" in data
    assert "prediction_intervals_sample" in data
    assert "assumptions" in data

    # Verify 5 assumptions present
    assumptions = data["assumptions"]
    assert "linearity" in assumptions
    assert "independence" in assumptions
    assert "homoscedasticity" in assumptions
    assert "normality" in assumptions
    assert "multicollinearity" in assumptions


def test_predict_single():
    # Predict for 50 startups
    req_body = {
        "dataset_id": "50_startups",
        "input_features": {
            "R&D Spend": 100000.0,
            "Administration": 120000.0,
            "Marketing Spend": 250000.0,
            "State": "New York",
        },
    }
    res = client.post("/api/predict", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_value" in data
    assert "ci_lower_95" in data
    assert "ci_upper_95" in data
    assert "pi_lower_95" in data
    assert "pi_upper_95" in data

    # PI must be wider than CI
    assert data["pi_lower_95"] <= data["ci_lower_95"]
    assert data["ci_upper_95"] <= data["pi_upper_95"]


def test_reports_markdown():
    res = client.get("/api/reports/markdown")
    assert res.status_code == 200
    text = res.text
    assert "多元線性回歸" in text
    assert "CRISP-DM" in text


def test_custom_url_ssrf_blocked():
    req_body = {
        "url": "http://127.0.0.1:8080/data.csv",
    }
    res = client.post("/api/datasets/custom-url", json=req_body)
    assert res.status_code == 400


def test_flood_probability_id_exclusion():
    # Verify flood_probability defaults to excluding 'id' even if empty list passed
    req_body = {
        "dataset_id": "flood_probability",
        "excluded_columns": [],  # explicitly empty
        "test_size": 0.20,
        "random_state": 42,
    }
    res = client.post("/api/analysis/train", json=req_body)
    assert res.status_code == 200
    data = res.json()
    features = [c["feature"] for c in data["coefficients"]]
    assert "id" not in features
    assert "id" not in data["encoded_feature_names"]


def test_custom_upload_workflow_end_to_end():
    # 1. Upload custom CSV
    csv_content = (
        "feature_a,feature_b,target_y\n"
        "1.0,2.0,10.5\n"
        "2.0,1.5,12.0\n"
        "3.0,3.0,15.2\n"
        "4.0,2.5,17.1\n"
        "5.0,4.0,21.0\n"
        "6.0,3.5,23.5\n"
        "7.0,5.0,27.0\n"
        "8.0,4.5,29.8\n"
        "9.0,6.0,33.1\n"
        "10.0,5.5,35.9\n"
    )
    files = {"file": ("test_custom.csv", csv_content, "text/csv")}
    upload_res = client.post("/api/datasets/upload", files=files)
    assert upload_res.status_code == 200
    overview = upload_res.json()
    assert overview["id"].startswith("custom_")
    assert overview["n_rows"] == 10
    assert overview["target_column"] == "target_y"
    custom_id = overview["id"]
    custom_records = overview["custom_data"]

    # 2. Train with cached custom_id and with custom_data payload
    train_res = client.post(
        "/api/analysis/train",
        json={
            "dataset_id": custom_id,
            "target_column": "target_y",
            "custom_data": custom_records,
            "test_size": 0.2,
        },
    )
    assert train_res.status_code == 200
    train_data = train_res.json()
    assert "coefficients" in train_data
    assert len(train_data["coefficients"]) == 2

    # 3. Feature selection with custom data
    fs_res = client.post(
        "/api/analysis/feature-selection",
        json={
            "dataset_id": custom_id,
            "target_column": "target_y",
            "custom_data": custom_records,
            "k_features": 1,
            "fast_mode": True,
        },
    )
    assert fs_res.status_code == 200
    fs_data = fs_res.json()
    assert len(fs_data["comparison_table"]) > 0

    # 4. Evaluation with custom data
    eval_res = client.post(
        "/api/analysis/evaluation",
        json={
            "dataset_id": custom_id,
            "target_column": "target_y",
            "custom_data": custom_records,
            "test_size": 0.2,
        },
    )
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert "train_metrics" in eval_data
    assert "test_metrics" in eval_data

    # 5. Prediction with custom data
    pred_res = client.post(
        "/api/predict",
        json={
            "dataset_id": custom_id,
            "target_column": "target_y",
            "custom_data": custom_records,
            "input_features": {"feature_a": 4.5, "feature_b": 3.2},
        },
    )
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "predicted_value" in pred_data
    assert pred_data["predicted_value"] > 0


def test_download_results_metrics_and_features():
    # Metrics download
    res_metrics = client.get("/api/reports/download-results?type=metrics")
    assert res_metrics.status_code == 200
    assert "text/csv" in res_metrics.headers["content-type"]
    assert "exam_score" in res_metrics.text.lower() or "ci_lower" in res_metrics.text.lower()

    # Features download
    res_features = client.get("/api/reports/download-results?type=features")
    assert res_features.status_code == 200
    assert "text/csv" in res_features.headers["content-type"]
    assert "method" in res_features.text.lower() or "features" in res_features.text.lower()

