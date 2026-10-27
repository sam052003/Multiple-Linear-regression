"""
Data Understanding (EDA) UI Component.
Presents dataset exploration, distributions, correlation heatmaps, and data quality diagnostics.
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd

from src.data_validation import generate_data_quality_report
from src.visualization import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_missing_values,
)


def render_data_understanding(df: pd.DataFrame, config: Dict[str, Any]):
    """
    Render Data Understanding (EDA) Tab.
    """
    st.markdown("## 🔍 階段 2：資料探索與理解 (Data Understanding)")

    target_col = config["target_col"]

    # 1. Dataset Preview
    with st.expander("📄 資料集前 10 筆原始預覽 (Dataset Preview)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    # 2. Data Quality & Diagnostics
    quality = generate_data_quality_report(df, target_col=target_col)

    st.markdown("### 📊 資料品質檢測摘要 (Data Quality Summary)")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("缺失欄位數", f"{len(quality['missing_summary'])} / {df.shape[1]}")
    with q2:
        st.metric("總缺失格數", f"{quality['total_missing_cells']:,}")
    with q3:
        st.metric("完全重複列數", f"{quality['duplicate_rows']:,}")
    with q4:
        st.metric("常數欄位數", f"{len(quality['constant_cols'])}")

    if quality["potential_id_cols"]:
        st.warning(f"⚠️ 偵測到疑似識別碼/序號欄位 (建議排除)：`{', '.join(quality['potential_id_cols'])}`")

    if quality["high_corr_pairs"]:
        with st.expander("⚠️ 高度共線性特徵警告 (Pearson |r| >= 0.85)", expanded=False):
            st.markdown("下列特徵對之間存在高度線性相關，可能在多元回歸中引發多重共線性 (Multicollinearity)：")
            st.table(pd.DataFrame(quality["high_corr_pairs"]))

    st.markdown("---")

    # 3. Descriptive Statistics
    st.markdown("### 📈 統計描述量 (Descriptive Statistics)")
    tab_num, tab_cat = st.tabs(["數值型特徵 (Numeric)", "類別型特徵 (Categorical)"])

    with tab_num:
        if quality["numeric_cols"]:
            st.dataframe(df[quality["numeric_cols"]].describe().T.style.format("{:.3f}"), use_container_width=True)
        else:
            st.info("資料集中無數值型特徵。")

    with tab_cat:
        if quality["categorical_cols"]:
            st.dataframe(df[quality["categorical_cols"]].describe().T, use_container_width=True)
        else:
            st.info("資料集中無類別型特徵。")

    st.markdown("---")

    # 4. Exploratory Visualizations
    st.markdown("### 🎨 互動式探索視覺化 (Visualizations)")

    v1, v2 = st.columns(2)
    with v1:
        if target_col in df.columns:
            target_series = pd.to_numeric(df[target_col], errors="coerce").dropna()
            fig_target = plot_target_distribution(target_series, target_col)
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            st.warning(f"找不到目標變數 {target_col}")

    with v2:
        fig_miss = plot_missing_values(df)
        st.plotly_chart(fig_miss, use_container_width=True)

    # 5. Correlation Heatmap
    if len(quality["numeric_cols"]) >= 2:
        st.markdown("#### 數值特徵相關矩陣 (Correlation Heatmap)")
        fig_corr = plot_correlation_heatmap(df, quality["numeric_cols"])
        st.plotly_chart(fig_corr, use_container_width=True)
