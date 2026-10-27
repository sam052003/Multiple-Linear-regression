"""
Prediction Intervals Module.
Uses Statsmodels OLS to compute 95% Confidence Intervals for Mean Response
and 95% Prediction Intervals for Individual Observations.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


class OLSPredictionModel:
    def __init__(self):
        self.ols_model = None
        self.ols_results: Optional[RegressionResultsWrapper] = None
        self.feature_names = []
        self.has_constant = True

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit Statsmodels OLS model on the selected/preprocessed features.
        Adds constant term to design matrix.
        """
        self.feature_names = list(X_train.columns)
        X_design = sm.add_constant(X_train, has_constant="add")
        y_vec = np.asarray(y_train).ravel()

        self.ols_model = sm.OLS(y_vec, X_design)
        self.ols_results = self.ols_model.fit()
        return self

    def predict_intervals(
        self,
        X_new: pd.DataFrame,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compute point predictions, 95% Confidence Intervals for Mean,
        and 95% Prediction Intervals for Observations.
        """
        if self.ols_results is None:
            raise RuntimeError("OLS 模型尚未擬合，請先調用 fit()")

        # Ensure column ordering matches training
        X_aligned = X_new[self.feature_names].copy()
        X_design = sm.add_constant(X_aligned, has_constant="add")

        # In case add_constant omitted 'const' (e.g. if column was already constant)
        if "const" not in X_design.columns:
            X_design.insert(0, "const", 1.0)

        # Get prediction summary frame
        pred_obj = self.ols_results.get_prediction(X_design)
        summary_frame = pred_obj.summary_frame(alpha=alpha)

        # Output columns:
        # - mean: 預測期望值
        # - mean_se: 估計標準誤
        # - mean_ci_lower, mean_ci_upper: 95% Confidence Interval for Mean (母體平均信賴區間)
        # - obs_ci_lower, obs_ci_upper: 95% Prediction Interval for Observation (單一觀測值預測區間)
        summary_frame.index = X_new.index
        return summary_frame

    def get_summary_text(self) -> str:
        """Return formatted Statsmodels OLS summary."""
        if self.ols_results is None:
            return "OLS 模型尚未訓練。"
        return str(self.ols_results.summary())

    def get_degrees_of_freedom(self) -> Dict[str, Any]:
        if self.ols_results is None:
            return {}
        return {
            "df_model": float(self.ols_results.df_model),
            "df_resid": float(self.ols_results.df_resid),
            "nobs": int(self.ols_results.nobs),
            "rsquared": float(self.ols_results.rsquared),
            "rsquared_adj": float(self.ols_results.rsquared_adj),
            "fvalue": float(self.ols_results.fvalue) if not np.isnan(self.ols_results.fvalue) else None,
            "f_pvalue": float(self.ols_results.f_pvalue) if not np.isnan(self.ols_results.f_pvalue) else None,
        }
