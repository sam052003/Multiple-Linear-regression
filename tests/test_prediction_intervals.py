"""
Tests for Statsmodels OLS 95% Prediction Intervals, Confidence Intervals, and Regression Assumptions.
"""

import pytest
import numpy as np
import pandas as pd
from src.prediction_intervals import OLSPredictionModel
from src.statistical_tests import diagnose_regression_assumptions


@pytest.fixture
def regression_sample():
    np.random.seed(42)
    n = 60
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 2.0 * x1 + 1.5 * x2 + np.random.randn(n) * 0.5 + 10.0
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.Series(y, name="target")
    return X, y


def test_ols_prediction_intervals(regression_sample):
    X, y = regression_sample
    X_train, X_test = X.iloc[:50], X.iloc[50:]
    y_train, y_test = y.iloc[:50], y.iloc[50:]

    model = OLSPredictionModel()
    model.fit(X_train, y_train)

    assert model.ols_results is not None
    summary_frame = model.predict_intervals(X_test, alpha=0.05)

    assert "mean" in summary_frame.columns
    assert "mean_ci_lower" in summary_frame.columns
    assert "mean_ci_upper" in summary_frame.columns
    assert "obs_ci_lower" in summary_frame.columns
    assert "obs_ci_upper" in summary_frame.columns

    # Verify key statistical property: Prediction Interval (PI) is strictly wider than Confidence Interval (CI)
    for i in range(len(summary_frame)):
        obs_l = summary_frame["obs_ci_lower"].iloc[i]
        mean_l = summary_frame["mean_ci_lower"].iloc[i]
        m = summary_frame["mean"].iloc[i]
        mean_u = summary_frame["mean_ci_upper"].iloc[i]
        obs_u = summary_frame["obs_ci_upper"].iloc[i]

        assert obs_l <= mean_l, f"PI lower ({obs_l}) should be <= CI lower ({mean_l})"
        assert mean_l <= m <= mean_u, f"Mean ({m}) should be between CI bounds [{mean_l}, {mean_u}]"
        assert mean_u <= obs_u, f"CI upper ({mean_u}) should be <= PI upper ({obs_u})"


def test_regression_assumptions_diagnostics(regression_sample):
    X, y = regression_sample
    model = OLSPredictionModel()
    model.fit(X, y)

    diag = diagnose_regression_assumptions(model.ols_results, X, y)

    assert "linearity" in diag
    assert "independence" in diag
    assert "homoscedasticity" in diag
    assert "normality" in diag
    assert "multicollinearity" in diag

    # DW value
    assert 1.0 <= diag["independence"]["value"] <= 3.0
    # Linearity correlation
    assert abs(diag["linearity"]["value"]) < 0.1
