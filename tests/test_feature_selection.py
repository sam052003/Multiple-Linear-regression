"""
Tests for Feature Selection Methods (SelectKBest, RFE, SFS, Lasso, Permutation Importance).
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_selection import (
    select_k_best_features,
    select_rfe_features,
    select_sfs_features,
    select_lasso_features,
    select_permutation_features,
    run_feature_selection_comparison,
)


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n = 100
    # True relationship: y = 3*x1 - 2*x2 + 0.5*x3 + noise (x4, x5 are noise)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)
    y = 3.0 * x1 - 2.0 * x2 + 0.5 * x3 + np.random.randn(n) * 0.1

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
    y = pd.Series(y, name="target")
    return X, y


def test_select_k_best(regression_data):
    X, y = regression_data
    res = select_k_best_features(X, y, k=3)
    assert len(res.selected_features) == 3
    assert "x1" in res.selected_features
    assert "x2" in res.selected_features
    assert res.feature_scores is not None
    assert len(res.feature_rankings) == 5


def test_select_rfe(regression_data):
    X, y = regression_data
    res = select_rfe_features(X, y, n_features_to_select=2)
    assert len(res.selected_features) == 2
    assert "x1" in res.selected_features
    assert "x2" in res.selected_features
    assert res.feature_rankings["x1"] == 1
    assert res.feature_rankings["x2"] == 1


def test_select_sfs(regression_data):
    X, y = regression_data
    res = select_sfs_features(X, y, n_features_to_select=2, direction="forward", fast_mode=False)
    assert len(res.selected_features) == 2
    assert "x1" in res.selected_features
    assert "x2" in res.selected_features


def test_select_lasso(regression_data):
    X, y = regression_data
    res = select_lasso_features(X, y, target_k=3, cv=3, random_state=42)
    assert len(res.selected_features) >= 1
    assert "x1" in res.selected_features
    assert "alpha" in res.details
    assert res.details["alpha"] > 0


def test_select_permutation(regression_data):
    X, y = regression_data
    res = select_permutation_features(X, y, k=2, random_state=42)
    assert len(res.selected_features) == 2
    assert "x1" in res.selected_features


def test_run_feature_selection_comparison(regression_data):
    X, y = regression_data
    X_tr, X_te = X.iloc[:80], X.iloc[80:]
    y_tr, y_te = y.iloc[:80], y.iloc[80:]

    summary_df, results_map, models_map = run_feature_selection_comparison(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        k=3,
        cv_folds=3,
        random_state=42,
    )

    assert len(summary_df) == 6
    assert "Baseline (All Features)" in summary_df["Method"].values
    assert "SelectKBest (Filter)" in summary_df["Method"].values
    assert "RFE (Wrapper)" in summary_df["Method"].values
    assert "SFS Forward (Wrapper)" in summary_df["Method"].values
    assert "Lasso Selection (Embedded)" in summary_df["Method"].values
    assert "Permutation Importance" in summary_df["Method"].values

    # Check metrics existence
    assert "R2" in summary_df.columns
    assert "RMSE" in summary_df.columns
    assert "MAE" in summary_df.columns
    assert (summary_df["R2"] > 0.9).any()
