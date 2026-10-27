"""
Evaluation UI Component.
Displays regression evaluation metrics (Train, CV, Test), Actual vs Predicted, and Residual plots.
"""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

from src.visualization import (
    plot_actual_vs_predicted,
    plot_residuals_vs_fitted,
    plot_residuals_qq,
    plot_prediction_interval,
)
from src.statistical_tests import diagnose_regression_assumptions
from src.prediction_intervals import OLSPredictionModel


def render_evaluation(
    eval_results: Optional[Dict[str, Any]],
    ols_model: Optional[OLSPredictionModel],
    config: Dict[str, Any],
):
    """
    Render Evaluation Tab with metrics, assumption diagnostics, and 95% Prediction Interval.
    """
    st.markdown("## 📈 階段 5：模型評估與統計假設診斷 (Evaluation & Diagnostics)")

    if eval_results is None:
        st.info("💡 尚未執行模型評估。請點擊側邊欄的「🚀 執行分析與模型訓練」按鈕。")
        return

    test_metrics = eval_results["test"]
    train_metrics = eval_results["train"]
    cv_metrics = eval_results["cv"]

    # 1. KPI Cards for Test Set
    st.markdown("### 🎯 測試集最終表現 (Test Set Performance)")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("R² (決定係數)", f"{test_metrics['R2']:.4f}")
    with k2:
        adj_r2_val = f"{test_metrics['Adj_R2']:.4f}" if test_metrics['Adj_R2'] is not None else "N/A"
        st.metric("Adjusted R²", adj_r2_val)
    with k3:
        st.metric("RMSE", f"{test_metrics['RMSE']:.4f}")
    with k4:
        st.metric("MAE", f"{test_metrics['MAE']:.4f}")
    with k5:
        st.metric("MSE", f"{test_metrics['MSE']:.4f}")

    if config["problem_name"] == "50 Startups — Profit Prediction":
        st.caption("⚠️ **小樣本提示**：50 Startups 資料集僅 50 筆，測試集樣本數較小，指標容易隨資料切分出現較大波動。")

    st.markdown("---")

    # 2. Side-by-Side Comparison Table (Train vs CV vs Test)
    st.markdown("### 📊 訓練集、交叉驗證與測試集指標對照 (Train vs CV vs Test)")

    cv_r2_str = f"{cv_metrics.get('CV_R2_Mean', 0.0):.4f} (±{cv_metrics.get('CV_R2_Std', 0.0):.4f})" if cv_metrics.get("CV_R2_Mean") is not None else "N/A"
    cv_rmse_str = f"{cv_metrics.get('CV_RMSE_Mean', 0.0):.4f} (±{cv_metrics.get('CV_RMSE_Std', 0.0):.4f})" if cv_metrics.get("CV_RMSE_Mean") is not None else "N/A"

    comparison_data = {
        "評估分割 (Split)": ["訓練集 (Train)", f"{config['cv_folds']}-Fold 交叉驗證 (CV)", "測試集 (Test)"],
        "樣本筆數 (N)": [train_metrics["n_samples"], train_metrics["n_samples"], test_metrics["n_samples"]],
        "R² (決定係數)": [f"{train_metrics['R2']:.4f}", cv_r2_str, f"{test_metrics['R2']:.4f}"],
        "Adjusted R²": [
            f"{train_metrics['Adj_R2']:.4f}" if train_metrics["Adj_R2"] is not None else "N/A",
            "-",
            f"{test_metrics['Adj_R2']:.4f}" if test_metrics["Adj_R2"] is not None else "N/A",
        ],
        "RMSE": [f"{train_metrics['RMSE']:.4f}", cv_rmse_str, f"{test_metrics['RMSE']:.4f}"],
        "MAE": [f"{train_metrics['MAE']:.4f}", "-", f"{test_metrics['MAE']:.4f}"],
    }
    st.table(pd.DataFrame(comparison_data))

    st.markdown("---")

    # 3. 95% Prediction Interval Section
    st.markdown("### 📏 95% 預測區間 (95% Prediction Interval for Observations)")
    st.caption("基於 Statsmodels OLS 統計推論模型計算，繪製排序後之預測點、95% 預測上下限 (PI) 與實際觀測值。")

    y_test_true = np.asarray(eval_results["y_test_true"])
    y_test_pred = np.asarray(eval_results["y_test_pred"])

    if ols_model and ols_model.ols_results:
        try:
            # Predict intervals on test set
            X_test_proc = pd.DataFrame(eval_results.get("X_test_proc", {}))
            if not X_test_proc.empty:
                pi_df = ols_model.predict_intervals(X_test_proc, alpha=0.05)
                lower_bound = pi_df["obs_ci_lower"].values
                upper_bound = pi_df["obs_ci_upper"].values
            else:
                # Estimate PI bound from residual standard error (RMSE * 1.96) as fallback
                rmse_val = test_metrics["RMSE"]
                lower_bound = y_test_pred - 1.96 * rmse_val
                upper_bound = y_test_pred + 1.96 * rmse_val

            fig_pi = plot_prediction_interval(
                y_true=y_test_true,
                y_pred=y_test_pred,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                sample_limit=50,
                title=f"95% 預測區間與實際觀測值比較 (95% Prediction Interval - {config['target_col']})",
            )
            st.plotly_chart(fig_pi, use_container_width=True)
        except Exception as e:
            st.warning(f"計算測試集預測區間時發生提示: {e}")

    st.markdown("---")

    # 4. Diagnostic Evaluation Charts
    st.markdown("### 🎨 預測視覺化與殘差分佈 (Evaluation Charts)")

    residuals = y_test_true - y_test_pred

    c1, c2 = st.columns(2)
    with c1:
        fig_actual_pred = plot_actual_vs_predicted(
            y_test_true,
            y_test_pred,
            title="實際值 vs. 預測值 (Actual vs. Predicted - Test Set)",
            target_name=config["target_col"],
        )
        st.plotly_chart(fig_actual_pred, use_container_width=True)

    with c2:
        fig_resid = plot_residuals_vs_fitted(
            y_test_pred,
            residuals,
            title="殘差 vs. 擬合值 (Residuals vs. Fitted Values)",
        )
        st.plotly_chart(fig_resid, use_container_width=True)

    fig_qq = plot_residuals_qq(residuals, title="殘差常態分佈檢驗 (Normal Q-Q Plot)")
    st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("---")

    # 5. Five Regression Assumptions Diagnostic Suite
    st.markdown("### 📋 多元線性回歸五大統計假設診斷 (Regression Assumptions)")
    st.caption("全面檢視線性回歸之 Linearity, Independence, Homoscedasticity, Normality, Multicollinearity 假定。")

    if ols_model and ols_model.ols_results:
        diag = diagnose_regression_assumptions(
            ols_results=ols_model.ols_results,
            X_train=pd.DataFrame(eval_results.get("X_train_proc", {})),
            y_train=eval_results["y_train_true"],
        )

        # Render 5 tabs or cards
        t1, t2, t3, t4, t5 = st.tabs([
            "1. 線性性 (Linearity)",
            "2. 獨立性 (Independence)",
            "3. 同質變異 (Homoscedasticity)",
            "4. 殘差常態性 (Normality)",
            "5. 多重共線性 (VIF)",
        ])

        with t1:
            st.markdown(f"**診斷項目**：{diag['linearity']['diagnostic']}")
            st.markdown(f"**檢定方法**：{diag['linearity']['method']}")
            st.markdown(f"**數值指標**：`{diag['linearity']['metric_name']} = {diag['linearity']['value']}`")
            st.markdown(f"**統計解釋**：\n{diag['linearity']['interpretation']}")
            st.caption(f"⚠️ 限制說明：{diag['linearity']['limitations']}")

        with t2:
            st.markdown(f"**診斷項目**：{diag['independence']['diagnostic']}")
            st.markdown(f"**檢定方法**：{diag['independence']['method']}")
            st.markdown(f"**統計量**：`{diag['independence']['metric_name']} = {diag['independence']['value']}`")
            st.markdown(f"**統計解釋**：\n{diag['independence']['interpretation']}")
            st.caption(f"⚠️ 限制說明：{diag['independence']['limitations']}")

        with t3:
            st.markdown(f"**診斷項目**：{diag['homoscedasticity']['diagnostic']}")
            st.markdown(f"**檢定方法**：{diag['homoscedasticity']['method']}")
            st.markdown(f"**檢定數值**：`{diag['homoscedasticity']['value']}`")
            st.markdown(f"**統計解釋**：\n{diag['homoscedasticity']['interpretation']}")
            st.caption(f"⚠️ 限制說明：{diag['homoscedasticity']['limitations']}")

        with t4:
            st.markdown(f"**診斷項目**：{diag['normality']['diagnostic']}")
            st.markdown(f"**檢定方法**：{diag['normality']['method']}")
            st.markdown(f"**檢定數值**：`{diag['normality']['value']}` (偏度 Skewness = {diag['normality'].get('skewness', '-')}, 峰度 Kurtosis = {diag['normality'].get('kurtosis', '-')})")
            st.markdown(f"**統計解釋**：\n{diag['normality']['interpretation']}")
            st.caption(f"⚠️ 限制說明：{diag['normality']['limitations']}")

        with t5:
            st.markdown(f"**診斷項目**：{diag['multicollinearity']['diagnostic']}")
            st.markdown(f"**檢定方法**：{diag['multicollinearity']['method']}")
            st.markdown(f"**最大 VIF**：`{diag['multicollinearity']['value']}`")
            st.markdown(f"**統計解釋**：\n{diag['multicollinearity']['interpretation']}")
            if not diag['multicollinearity']['vif_table'].empty:
                st.dataframe(diag['multicollinearity']['vif_table'], height=250, use_container_width=True)
            st.caption(f"⚠️ 限制說明：{diag['multicollinearity']['limitations']}")

