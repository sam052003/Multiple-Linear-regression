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


def test_custom_dataset_over_2000_rows_data_consistency():
    """
    Ensure custom datasets with > 2,000 rows (e.g. 2,500 rows) are NEVER silently truncated to 2,000.
    Verify that Dataset Overview, Training, and Evaluation analyze exactly the same 2,500 rows.
    """
    import numpy as np
    np.random.seed(42)
    n = 2500
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 2.5 * x1 - 1.8 * x2 + np.random.randn(n) * 0.1

    lines = ["feat1,feat2,target_y"]
    for i in range(n):
        lines.append(f"{x1[i]:.4f},{x2[i]:.4f},{y[i]:.4f}")
    csv_text = "\n".join(lines)

    # 1. Upload 2500 rows
    files = {"file": ("dataset_2500.csv", csv_text, "text/csv")}
    res_upload = client.post("/api/datasets/upload", files=files)
    assert res_upload.status_code == 200
    overview = res_upload.json()

    # Verify overview counts
    assert overview["n_rows"] == 2500
    assert len(overview["custom_data"]) == 2500, "custom_data must retain all 2500 rows without silent truncation"

    custom_id = overview["id"]
    custom_records = overview["custom_data"]

    # 2. Train model with 2500 rows (80/20 train/test split = 2000 train + 500 test)
    res_train = client.post(
        "/api/analysis/train",
        json={
            "dataset_id": custom_id,
            "target_column": "target_y",
            "custom_data": custom_records,
            "test_size": 0.20,
        },
    )
    assert res_train.status_code == 200
    train_data = res_train.json()
    assert train_data["n_train_samples"] == 2000
    assert train_data["n_test_samples"] == 500
    assert train_data["n_train_samples"] + train_data["n_test_samples"] == 2500

    # 3. Evaluation consistency
    res_eval = client.post(
        "/api/analysis/evaluation",
        json={
            "dataset_id": custom_id,
            "target_column": "target_y",
            "custom_data": custom_records,
            "test_size": 0.20,
        },
    )
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert eval_data["train_metrics"]["n_samples"] + eval_data["test_metrics"]["n_samples"] == 2500


def test_stateless_environment_simulation_without_shared_memory():
    """
    Simulate Vercel Serverless environment where instances do not share memory or persistent disk.
    Ensure requests carrying custom_data succeed independently.
    """
    from api.services.ml_service import MLService
    import tempfile
    import os

    # Prepare small records
    records = [
        {"a": 1.0, "b": 2.0, "c": 3.0},
        {"a": 2.0, "b": 3.0, "c": 5.0},
        {"a": 3.0, "b": 4.0, "c": 7.0},
        {"a": 4.0, "b": 5.0, "c": 9.0},
        {"a": 5.0, "b": 6.0, "c": 11.0},
        {"a": 6.0, "b": 7.0, "c": 13.0},
        {"a": 7.0, "b": 8.0, "c": 15.0},
        {"a": 8.0, "b": 9.0, "c": 17.0},
        {"a": 9.0, "b": 10.0, "c": 19.0},
        {"a": 10.0, "b": 11.0, "c": 21.0},
    ]

    custom_id = "custom_stateless_sim_999"

    # WIPE memory cache and disk tempfile to simulate a brand-new Serverless instance
    MLService._CUSTOM_DATASETS_CACHE.clear()
    tmp_path = os.path.join(tempfile.gettempdir(), f"{custom_id}.csv")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # 1. Calling train WITH custom_data MUST succeed statelessly
    res_with_data = client.post(
        "/api/analysis/train",
        json={
            "dataset_id": custom_id,
            "target_column": "c",
            "custom_data": records,
            "test_size": 0.2,
        },
    )
    assert res_with_data.status_code == 200
    assert "coefficients" in res_with_data.json()

    # WIPE AGAIN
    MLService._CUSTOM_DATASETS_CACHE.clear()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # 2. Calling train on custom dataset WITHOUT custom_data on cold instance fails with clean 500/400
    res_without_data = client.post(
        "/api/analysis/train",
        json={
            "dataset_id": custom_id,
            "target_column": "c",
            "custom_data": None,
            "test_size": 0.2,
        },
    )
    assert res_without_data.status_code in (400, 404, 500)
    assert "custom_data" in res_without_data.text or "找不到自訂資料集" in res_without_data.text


def test_custom_dataset_exceeding_max_rows_rejected():
    """
    Ensure datasets exceeding the 5,000 rows ceiling are rejected with HTTP 400
    rather than silently truncated.
    """
    # Create 5,005 rows
    lines = ["x,y"]
    for i in range(5005):
        lines.append(f"{i},{i*2}")
    csv_text = "\n".join(lines)

    files = {"file": ("too_large.csv", csv_text, "text/csv")}
    res = client.post("/api/datasets/upload", files=files)
    assert res.status_code == 400
    assert "5000" in res.text or "安全上限" in res.text


def test_csv_url_redirect_to_private_ip_blocked():
    """
    Ensure HTTP redirects pointing to private / loopback IPs are blocked hop-by-hop.
    """
    from unittest.mock import patch
    from src.data_validation import fetch_csv_safely

    class MockRedirectResponse:
        status_code = 302
        headers = {"Location": "http://127.0.0.1:8080/internal.csv"}

    fake_public_dns = [(2, 1, 6, "", ("93.184.216.34", 443))]

    with patch("socket.getaddrinfo", return_value=fake_public_dns):
        with patch("requests.Session.get", return_value=MockRedirectResponse()):
            df, err = fetch_csv_safely("https://example.com/redirect")
            assert df is None
            assert "轉址安全性阻擋" in err or "安全性阻擋" in err

    # Also test redirect to https://192.168.1.1/internal.csv
    class MockRedirectPrivateHttps:
        status_code = 302
        headers = {"Location": "https://192.168.1.1/internal.csv"}

    with patch("socket.getaddrinfo", return_value=fake_public_dns):
        with patch("requests.Session.get", return_value=MockRedirectPrivateHttps()):
            df, err = fetch_csv_safely("https://example.com/redirect")
            assert df is None
            assert "轉址安全性阻擋" in err or "安全性阻擋" in err


def test_html_response_rejected():
    """
    Ensure HTML pages returned instead of CSV are cleanly rejected.
    """
    from unittest.mock import patch
    from src.data_validation import fetch_csv_safely

    class MockHtmlResponse:
        status_code = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}
        def iter_content(self, chunk_size=65536):
            yield b"<!DOCTYPE html><html><head><title>404</title></head><body>Not Found</body></html>"

    with patch("requests.Session.get", return_value=MockHtmlResponse()):
        df, err = fetch_csv_safely("https://example.com/dataset.csv")
        assert df is None
        assert "HTML" in err


def test_upload_exceeding_byte_limit_rejected():
    """
    Ensure file upload exceeding 4.5 MB is rejected with HTTP 400.
    """
    oversized = b"a,b\n" + b"1,2\n" * (5 * 1024 * 1024 // 4)
    files = {"file": ("huge.csv", oversized, "text/csv")}
    res = client.post("/api/datasets/upload", files=files)
    assert res.status_code == 400
    assert "超過" in res.text and "4.5" in res.text

