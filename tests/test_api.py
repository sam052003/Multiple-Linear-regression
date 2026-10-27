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
