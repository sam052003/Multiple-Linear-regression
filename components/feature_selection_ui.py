"""
Feature Selection UI Component.
Displays comparative results across Baseline and 5 Feature Selection methods.
"""

from typing import Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np

from src.visualization import (
    plot_feature_selection_comparison,
    plot_feature_importance,
)
from src.feature_selection import FeatureSelectionResult


def render_feature_selection_ui(
    fs_summary_df: Optional[pd.DataFrame],
    fs_results_map: Optional[Dict[str, FeatureSelectionResult]],
    config: Dict[str, Any],
):
    """
    Render Feature Selection Tab.
    """
    st.markdown("## 🔍 特徵選擇深度評估與比較 (Feature Selection Comparison)")

    if fs_summary_df is None or fs_summary_df.empty:
        st.info("💡 尚未執行特徵選擇比較。請點擊側邊欄的「🚀 執行分析與模型訓練」按鈕以啟動全面特徵選擇評估。")
        return

    # 1. Best Model Highlight Card
    best_row = fs_summary_df.iloc[0]
    st.success(
        f"🏆 **最佳預測表現方法**：`{best_row['Method']}` (選取 {best_row['Features']} 個特徵, Test R² = **{best_row['R2']:.4f}**, RMSE = **{best_row['RMSE']:.4f}**)"
    )

    # 2. Comparison Table
    st.markdown("### 📊 6 大模型效能全面評比 (Performance Comparison Table)")
    st.caption("所有方法皆在完全一致的 Train/Test Split (80/20) 上評估，超參數與特徵篩選均僅使用訓練資料。")

    display_cols = ["Method", "Features", "R2", "Adj_R2", "RMSE", "MAE", "Train_R2", "CV_R2_Mean", "Time_Sec"]
    rename_dict = {
        "Method": "特徵選擇方法",
        "Features": "特徵數 (K)",
        "R2": "測試集 R²",
        "Adj_R2": "Adjusted R²",
        "RMSE": "測試集 RMSE",
        "MAE": "測試集 MAE",
        "Train_R2": "訓練集 R²",
        "CV_R2_Mean": "CV 平均 R²",
        "Time_Sec": "總耗時 (秒)",
    }

    styled_df = (
        fs_summary_df[display_cols]
        .rename(columns=rename_dict)
        .style.format(
            {
                "測試集 R²": "{:.4f}",
                "Adjusted R²": "{:.4f}",
                "測試集 RMSE": "{:.4f}",
                "測試集 MAE": "{:.4f}",
                "訓練集 R²": "{:.4f}",
                "CV 平均 R²": "{:.4f}",
                "總耗時 (秒)": "{:.2f}",
            },
            na_rep="N/A",
        )
    )
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")

    # 3. Comparative Visualizations
    st.markdown("### 🎨 特徵選擇效能視覺化 (Performance Charts)")
    fig_comp = plot_feature_selection_comparison(fs_summary_df)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # 4. Method Details & Feature Rankings Inspector
    st.markdown("### 🔬 各方法特徵排名與重要度檢視 (Feature Importance Inspector)")

    method_choices = [m for m in fs_summary_df["Method"] if m in fs_results_map and fs_results_map[m].feature_scores]
    if method_choices:
        inspect_method = st.selectbox("選擇欲深入檢視的特徵選擇方法：", options=method_choices, index=0)
        selected_res = fs_results_map[inspect_method]

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown(f"#### `{inspect_method}` 選出特徵")
            st.markdown(f"**選取特徵數**：`{len(selected_res.selected_features)}` 個")
            st.markdown(f"**演算法耗時**：`{selected_res.execution_time_sec * 1000:.2f} ms`")

            # Table of rankings
            if selected_res.feature_rankings:
                feat_ranking_list = []
                for feat in selected_res.all_features:
                    feat_ranking_list.append({
                        "特徵名稱": feat,
                        "排名 (Rank)": selected_res.feature_rankings.get(feat, "-"),
                        "重要度分數": selected_res.feature_scores.get(feat, 0.0) if selected_res.feature_scores else 0.0,
                        "是否被選取": "✅ 是" if feat in selected_res.selected_features else "❌ 否",
                    })
                rf_df = pd.DataFrame(feat_ranking_list).sort_values(by="排名 (Rank)").reset_index(drop=True)
                st.dataframe(rf_df.style.format({"重要度分數": "{:.4f}"}), height=360, use_container_width=True)

        with col_right:
            if selected_res.feature_scores:
                feats = list(selected_res.feature_scores.keys())
                scores = list(selected_res.feature_scores.values())
                fig_scores = plot_feature_importance(feats, np.array(scores), title=f"{inspect_method} 特徵重要度排序")
                st.plotly_chart(fig_scores, use_container_width=True)
