"""
Modeling Module.
Implements Linear Regression training, coefficient extraction, and single-sample prediction.
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso


class RegressionModelWrapper:
    def __init__(
        self,
        model_type: str = "LinearRegression",
        alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.alpha = alpha
        self.random_state = random_state

        if model_type == "LinearRegression":
            self.estimator = LinearRegression()
        elif model_type == "Ridge":
            self.estimator = Ridge(alpha=alpha, random_state=random_state)
        elif model_type == "Lasso":
            self.estimator = Lasso(alpha=alpha, random_state=random_state, max_iter=5000)
        else:
            raise ValueError(f"不支援的模型形態: {model_type}")

        self.fitted = False
        self.intercept_ = None
        self.coef_ = None
        self.coef_dict = {}
        self.feature_names_ = []
        self.training_time_sec = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        start_time = time.perf_counter()
        self.feature_names_ = list(X.columns)

        self.estimator.fit(X, y)
        self.training_time_sec = time.perf_counter() - start_time
        self.fitted = True

        self.intercept_ = float(self.estimator.intercept_)
        self.coef_ = np.asarray(self.estimator.coef_).ravel()
        self.coef_dict = {
            feat: float(c) for feat, c in zip(self.feature_names_, self.coef_)
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("模型尚未訓練，請先調用 fit()")
        return self.estimator.predict(X)

    def get_coefficient_table(self) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "Feature": self.feature_names_,
                "Coefficient": self.coef_,
                "Abs_Coefficient": np.abs(self.coef_),
            }
        ).sort_values(by="Abs_Coefficient", ascending=False)
        return df


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> RegressionModelWrapper:
    """
    Train a Baseline Multiple Linear Regression model with all available features.
    """
    wrapper = RegressionModelWrapper(model_type="LinearRegression")
    wrapper.fit(X_train, y_train)
    return wrapper
