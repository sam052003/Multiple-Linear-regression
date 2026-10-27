"""
Regression tests for Custom Dataset Selector, Target Validation, Sample Accounting,
and Zero-Sample Training Error Prevention.
"""

import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.index import app
from api.services.ml_service import MLService
from src.preprocessing import prepare_dataset

client = TestClient(app)


def test_custom_dataset_300_rows_text_target_returns_chinese_error_not_n_samples_0():
    """
    Test that uploading or training a 300-row custom dataset where target is categorical text
    returns a clear Traditional Chinese HTTP 400 error and NEVER raises raw sklearn n_samples=0 exception.
    """
    np.random.seed(42)
    # 300 rows dataset with text target
    data = [
        {
            "feature_x1": float(np.random.rand() * 100),
            "feature_x2": float(np.random.rand() * 50),
            "target_category": ["Class_A", "Class_B", "Class_C"][i % 3],
        }
        for i in range(300)
    ]

    req_payload = {
        "dataset_id": "custom_test_text_target",
        "target_column": "target_category",
        "test_size": 0.20,
        "custom_data": data,
    }

    resp = client.post("/api/analysis/train", json=req_payload)
    assert resp.status_code == 400
    detail = resp.json()["detail"]

    # Verify meaningful Traditional Chinese error message
    assert "模型訓練失敗" in detail
    assert "目前選取的目標欄位無法轉換為有效數值" in detail or "屬於類別型目標變數" in detail
    assert "300 筆資料中有 0 筆可用" in detail or "不適合目前的多元線性回歸模型" in detail
    assert "請選擇適合回歸分析的數值欄位" in detail or "請選擇連續型數值目標欄位" in detail

    # Crucially ensure raw sklearn internal exception string NEVER leaks
    assert "n_samples=0" not in detail
    assert "Adjust any of the aforementioned parameters" not in detail


def test_custom_dataset_300_rows_valid_numeric_80_20_split():
    """
    Test that a 300-row valid numeric dataset with 80/20 split produces exactly:
    240 train samples and 60 test samples, with accurate sample accounting.
    """
    np.random.seed(42)
    data = [
        {
            "feature_1": float(i * 0.5 + np.random.normal(0, 1)),
            "feature_2": float(i * 1.2 + np.random.normal(0, 2)),
            "feature_3": float(np.random.rand() * 10),
            "target_value": float(100.0 + i * 2.5 + np.random.normal(0, 0.5)),
        }
        for i in range(300)
    ]

    req_payload = {
        "dataset_id": "custom_test_300_valid",
        "target_column": "target_value",
        "test_size": 0.20,
        "random_state": 42,
        "custom_data": data,
    }

    resp = client.post("/api/analysis/train", json=req_payload)
    assert resp.status_code == 200
    res_data = resp.json()

    assert res_data["n_train_samples"] == 240
    assert res_data["n_test_samples"] == 60
    assert res_data["n_total_samples"] == 300
    assert res_data["n_valid_target_samples"] == 300
    assert res_data["n_dropped_target_samples"] == 0
    assert res_data["train_metrics"]["R2"] > 0.90
    assert res_data["test_metrics"]["R2"] > 0.90


def test_custom_dataset_last_column_text_with_valid_numeric_target_specified():
    """
    Test that when the last column of a CSV is text, but another column is valid numeric target:
    1. Automatic candidate inspection identifies the numeric column.
    2. User can specify or system defaults to the valid numeric target and trains successfully.
    """
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_a": np.random.randn(300),
        "feature_b": np.random.randn(300),
        "target_score": np.linspace(10, 100, 300) + np.random.randn(300),
        "notes_or_status": ["Active", "Pending", "Closed"] * 100,  # last column is text!
    })

    # Test automatic target recommendation
    target, rec_target, candidates = MLService.pick_default_target(df)
    assert rec_target == "target_score"
    assert target == "target_score"

    cand_map = {c["column"]: c for c in candidates}
    assert cand_map["target_score"]["is_suitable_for_regression"] is True
    assert cand_map["notes_or_status"]["is_suitable_for_regression"] is False

    # Upload via CSV endpoint
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    resp_upload = client.post(
        "/api/datasets/upload",
        files={"file": ("experiment_data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp_upload.status_code == 200
    overview = resp_upload.json()
    assert overview["target_column"] == "target_score"
    assert overview["is_target_valid"] is True

    # Train model using the uploaded dataset
    train_resp = client.post(
        "/api/analysis/train",
        json={
            "dataset_id": overview["id"],
            "target_column": "target_score",
            "test_size": 0.20,
            "custom_data": overview["custom_data"],
        },
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["n_train_samples"] == 240
    assert train_data["n_test_samples"] == 60


def test_custom_dataset_partial_missing_target_accounting():
    """
    Test that when Target has partial missing values (e.g. 20 missing out of 300):
    1. Overview correctly reports n_rows=300, n_valid_target_rows=280, n_dropped_target_rows=20.
    2. Training executes 80/20 split on 280 valid samples -> 224 train / 56 test.
    """
    np.random.seed(42)
    rows = []
    for i in range(300):
        # 20 rows with missing target
        target_val = None if i < 20 else float(50.0 + i * 0.5 + np.random.normal(0, 0.1))
        rows.append({
            "var_1": float(np.random.rand() * 10),
            "var_2": float(np.random.rand() * 20),
            "target_metric": target_val,
        })

    # Overview endpoint via POST
    overview_resp = client.post(
        "/api/datasets/custom_partial_missing/overview",
        json={"target_column": "target_metric", "custom_data": rows},
    )
    assert overview_resp.status_code == 200
    ov_data = overview_resp.json()
    assert ov_data["n_rows"] == 300
    assert ov_data["n_valid_target_rows"] == 280
    assert ov_data["n_dropped_target_rows"] == 20

    # Train endpoint
    train_resp = client.post(
        "/api/analysis/train",
        json={
            "dataset_id": "custom_partial_missing",
            "target_column": "target_metric",
            "test_size": 0.20,
            "custom_data": rows,
        },
    )
    assert train_resp.status_code == 200
    tr_data = train_resp.json()
    assert tr_data["n_total_samples"] == 300
    assert tr_data["n_valid_target_samples"] == 280
    assert tr_data["n_dropped_target_samples"] == 20
    assert tr_data["n_train_samples"] == 224  # 280 * 0.8
    assert tr_data["n_test_samples"] == 56    # 280 * 0.2


def test_custom_dataset_invalid_target_scenarios_return_clear_errors():
    """
    Test all target missing, constant target, and binary classification target scenarios.
    """
    # 1. All target values are missing
    data_all_nan = [{"f": i, "target": None} for i in range(50)]
    resp1 = client.post(
        "/api/analysis/train",
        json={"dataset_id": "custom_all_nan", "target_column": "target", "custom_data": data_all_nan},
    )
    assert resp1.status_code == 400
    assert "0 筆可用" in resp1.json()["detail"]

    # 2. Target is constant (variance == 0)
    data_const = [{"f": i, "target": 42.0} for i in range(50)]
    resp2 = client.post(
        "/api/analysis/train",
        json={"dataset_id": "custom_const", "target_column": "target", "custom_data": data_const},
    )
    assert resp2.status_code == 400
    assert "無變異性" in resp2.json()["detail"] or "常數" in resp2.json()["detail"]

    # 3. Target is binary classification (nunique == 2)
    data_binary = [{"f": i, "target": i % 2} for i in range(50)]
    resp3 = client.post(
        "/api/analysis/train",
        json={"dataset_id": "custom_binary", "target_column": "target", "custom_data": data_binary},
    )
    assert resp3.status_code == 400
    assert "二元分類標籤" in resp3.json()["detail"] or "類別型目標變數" in resp3.json()["detail"]


def test_custom_dataset_feature_selection_evaluation_prediction():
    """
    Test that Feature Selection, Model Evaluation, and Prediction Interval work normally
    on a custom 300-row dataset without errors.
    """
    np.random.seed(42)
    data = [
        {
            "f1": float(i + np.random.randn()),
            "f2": float(i * 2 + np.random.randn()),
            "f3": float(np.random.randn()),
            "f4": float(np.random.randn()),
            "target": float(10 + i * 3 + np.random.randn()),
        }
        for i in range(300)
    ]

    dataset_id = "custom_300_full_flow"

    # 1. Feature Selection
    fs_resp = client.post(
        "/api/analysis/feature-selection",
        json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "k_features": 3,
            "fast_mode": True,
            "custom_data": data,
        },
    )
    assert fs_resp.status_code == 200
    fs_data = fs_resp.json()
    assert len(fs_data["comparison_table"]) == 6  # 5 methods + baseline
    assert fs_data["best_method"] is not None

    # 2. Evaluation & Diagnostics
    eval_resp = client.post(
        "/api/analysis/evaluation",
        json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "selected_features": ["f1", "f2"],
            "test_size": 0.20,
            "custom_data": data,
        },
    )
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert len(eval_data["prediction_intervals_sample"]) > 0
    assert "linearity" in eval_data["assumptions"]
    assert "normality" in eval_data["assumptions"]

    # 3. Single Prediction
    pred_resp = client.post(
        "/api/predict",
        json={
            "dataset_id": dataset_id,
            "input_features": {"f1": 150.0, "f2": 300.0, "f3": 0.5, "f4": -0.2},
            "target_column": "target",
            "custom_data": data,
        },
    )
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["predicted_value"] > 0
    assert pred_data["ci_lower_95"] < pred_data["ci_upper_95"]
    assert pred_data["pi_lower_95"] < pred_data["pi_upper_95"]
