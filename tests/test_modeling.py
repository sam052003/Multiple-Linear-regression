"""
Tests for Data Preprocessing, Baseline Linear Regression, and Evaluation Metrics.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import prepare_dataset
from src.modeling import train_baseline_model
from src.evaluation import calculate_adjusted_r2, compute_metrics_dict, evaluate_regression_model


def test_prepare_dataset_and_baseline_fit():
    # Synthetic dataset with numeric and categorical features
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "cat1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [2.5, 4.8, 6.9, 9.1, 11.2, 13.5, 15.6, 17.8, 19.9, 22.1],
    })

    artifacts = prepare_dataset(
        df=df,
        target_col="target",
        test_size=0.20,
        random_state=42,
    )

    # 10 rows: 8 train, 2 test
    assert len(artifacts.X_train_proc) == 8
    assert len(artifacts.X_test_proc) == 2
    assert "num1" in artifacts.feature_names_out
    assert "num2" in artifacts.feature_names_out
    assert any("cat1" in col for col in artifacts.feature_names_out)

    # Train Baseline Model
    model = train_baseline_model(artifacts.X_train_proc, artifacts.y_train)
    assert model.fitted
    assert len(model.coef_) == artifacts.X_train_proc.shape[1]
    assert isinstance(model.intercept_, float)

    # Predictions
    preds = model.predict(artifacts.X_test_proc)
    assert len(preds) == 2


def test_adjusted_r2_calculation():
    # Valid degrees of freedom
    r2 = 0.85
    n_samples = 100
    n_features = 5
    adj_r2 = calculate_adjusted_r2(r2, n_samples, n_features)
    assert adj_r2 is not None
    assert 0.80 < adj_r2 < 0.85

    # Degrees of freedom <= 0 (e.g. n=5, p=5)
    adj_r2_invalid = calculate_adjusted_r2(r2, n_samples=5, n_features=5)
    assert adj_r2_invalid is None


def test_compute_metrics_dict():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([10.5, 19.5, 30.2, 39.8, 50.1])
    metrics = compute_metrics_dict(y_true, y_pred, n_features=2)

    assert "MAE" in metrics
    assert "MSE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "Adj_R2" in metrics

    assert metrics["MAE"] < 1.0
    assert metrics["R2"] > 0.99
