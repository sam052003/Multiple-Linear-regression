"""
Modeling UI Component.
Displays Linear Regression model fitting, intercept, coefficients, and training metadata.
"""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd

from src.modeling import RegressionModelWrapper
from src.visualization import plot_feature_importance


def render_modeling(model_wrapper: Optional[RegressionModelWrapper], config: Dict[str, Any]):
    """
    Render Modeling Tab.
    """
    st.markdown("## 🧠 階段 4：多元線性回歸模型建構 (Modeling)")

    if model_wrapper is None or not model_wrapper.fitted:
        st.info("💡 模型尚未訓練。請點擊側邊欄的「🚀 執行分析與模型訓練」按鈕。")
        return

    # Metadata Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("核心演算法", model_wrapper.model_type)
    with m2:
        st.metric("納入特徵數", len(model_wrapper.feature_names_))
    with m3:
        st.metric("截距項 Intercept (β₀)", f"{model_wrapper.intercept_:.4f}")
    with m4:
        st.metric("訓練耗時", f"{model_wrapper.training_time_sec * 1000:.2f} ms")

    st.markdown("---")

    # Coefficients Section
    st.markdown("### 📐 回歸係數與權重解析 (Regression Coefficients)")
    st.caption("註：數值特徵已進行標準化 (Z-score Scaling)，係數大小可直接反映該特徵每變動一個標準差時，目標變數之預期變化量。")

    coef_df = model_wrapper.get_coefficient_table()

    col_table, col_fig = st.columns([1, 1.2])
    with col_table:
        st.markdown("#### 係數清單 (Coefficients Table)")
        st.dataframe(
            coef_df.style.format(
                {"Coefficient": "{:.4f}", "Abs_Coefficient": "{:.4f}"}
            ),
            height=380,
            use_container_width=True,
        )

    with col_fig:
        fig_coef = plot_feature_importance(
            coef_df["Feature"].tolist(),
            coef_df["Coefficient"].values,
            title="回歸特徵絕對權重排序 (Absolute Coefficients)",
        )
        st.plotly_chart(fig_coef, use_container_width=True)
