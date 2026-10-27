"""
Evaluation Module.
Computes regression metrics: MAE, MSE, RMSE, R², Adjusted R² across Train, CV, and Test splits.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


def calculate_adjusted_r2(r2: float, n_samples: int, n_features: int) -> Optional[float]:
    """
    Compute Adjusted R²: 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    If degrees of freedom n - p - 1 <= 0, returns None.
    """
    df_resid = n_samples - n_features - 1
    if df_resid <= 0 or n_samples <= 1:
        return None
    adj_r2 = 1.0 - (1.0 - r2) * (n_samples - 1) / df_resid
    return float(adj_r2)


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, Any]:
    """
    Compute MAE, MSE, RMSE, R2, Adjusted R2 for a set of true and predicted targets.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n_samples = len(y_true)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    adj_r2 = calculate_adjusted_r2(r2, n_samples, n_features)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adj_R2": adj_r2,
        "n_samples": n_samples,
        "n_features": n_features,
    }


def evaluate_regression_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation returning Training, Cross-Validation (K-Fold), and Test metrics.
    """
    n_features = X_train.shape[1]

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = compute_metrics_dict(y_train, y_train_pred, n_features)
    test_metrics = compute_metrics_dict(y_test, y_test_pred, n_features)

    # Cross Validation on Training Set only
    cv_scores_r2 = []
    cv_scores_rmse = []
    if len(X_train) >= cv_folds * 2:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        # neg_mean_squared_error
        cv_neg_mse = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")
        cv_scores_rmse = np.sqrt(-cv_neg_mse)
        # r2
        cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

        cv_metrics = {
            "CV_R2_Mean": float(np.mean(cv_scores_r2)),
            "CV_R2_Std": float(np.std(cv_scores_r2)),
            "CV_RMSE_Mean": float(np.mean(cv_scores_rmse)),
            "CV_RMSE_Std": float(np.std(cv_scores_rmse)),
            "cv_folds": cv_folds,
            "cv_r2_scores": cv_scores_r2.tolist(),
            "cv_rmse_scores": cv_scores_rmse.tolist(),
        }
    else:
        cv_metrics = {
            "CV_R2_Mean": None,
            "CV_R2_Std": None,
            "CV_RMSE_Mean": None,
            "CV_RMSE_Std": None,
            "cv_folds": cv_folds,
            "note": "樣本數不足以執行多折交叉驗證",
        }

    return {
        "train": train_metrics,
        "test": test_metrics,
        "cv": cv_metrics,
        "y_test_true": y_test,
        "y_test_pred": y_test_pred,
        "y_train_true": y_train,
        "y_train_pred": y_train_pred,
    }
