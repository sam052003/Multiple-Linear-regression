"""
Sidebar UI Component.
Controls problem selection, dataset sources, hyperparameters, and session state isolation.
"""

from typing import Dict, Any, Tuple, Optional
import streamlit as st
import pandas as pd

from src.config import (
    PROBLEM_50_STARTUPS,
    PROBLEM_STUDENT,
    PROBLEM_FLOOD,
    PROBLEM_CHOICES,
    PROBLEM_METADATA,
    FEATURE_SELECTION_METHODS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_CV_FOLDS,
)
from src.data_loader import (
    load_dataset_by_name,
    load_custom_url,
    load_custom_file,
)
from src.data_validation import validate_target_column, validate_feature_count


def render_sidebar() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Render sidebar controls, manage session state, and load active dataset.
    Returns: (df, config_dict)
    """
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 12px 0 18px 0;">
            <h2 style="margin: 0; color: #4F46E5; font-size: 1.4rem;">📊 MLR Dashboard</h2>
            <p style="margin: 4px 0 0 0; color: #64748B; font-size: 0.85rem;">CRISP-DM 回歸分析工作台</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Problem Selector
    st.sidebar.markdown("### 🎯 選擇回歸問題 (Problem)")
    selected_problem = st.sidebar.selectbox(
        "選擇內建研究主題：",
        options=PROBLEM_CHOICES,
        index=0,  # 50 Startups is default
        key="selected_problem_choice",
        help="切換不同的多元線性回歸問題。首頁預設為 50 Startups 教學案例。",
    )

    # 2. Dataset Source Selector
    st.sidebar.markdown("### 📁 資料來源 (Dataset Source)")
    source_choice = st.sidebar.radio(
        "資料來源模式：",
        options=["內建資料集 (Built-in)", "自訂 CSV 網址 (CSV URL)", "上傳本機 CSV (Upload CSV)"],
        index=0,
        key="dataset_source_choice",
    )

    # State tracking for dataset switching
    current_dataset_key = f"{selected_problem}_{source_choice}"
    if st.session_state.get("active_dataset_key") != current_dataset_key:
        st.session_state["active_dataset_key"] = current_dataset_key
        st.session_state["model_trained"] = False
        st.session_state["ml_artifacts"] = None
        st.session_state["ols_results"] = None
        st.session_state["fs_comparison"] = None

    df = None
    is_custom = False
    meta = PROBLEM_METADATA.get(selected_problem, {})
    target_col = meta.get("default_target", "")
    excluded_cols = meta.get("excluded_cols", [])

    # Load based on source
    if source_choice == "內建資料集 (Built-in)":
        with st.spinner(f"正在載入 {selected_problem} ..."):
            df = load_dataset_by_name(selected_problem)
        st.sidebar.success(f"已載入: {len(df):,} 筆資料, {df.shape[1]} 欄位")

    elif source_choice == "自訂 CSV 網址 (CSV URL)":
        is_custom = True
        st.sidebar.info("💡 提示範例網址：\n`https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`")
        url_input = st.sidebar.text_input(
            "輸入公開 HTTPS CSV 網址：",
            value="",
            placeholder="https://.../dataset.csv",
            key="custom_csv_url_input",
        )
        if url_input:
            with st.spinner("安全驗證與下載中..."):
                loaded_df, err = load_custom_url(url_input)
            if err:
                st.sidebar.error(err)
            else:
                df = loaded_df
                st.sidebar.success(f"成功載入: {len(df):,} 筆, {df.shape[1]} 欄位")

    elif source_choice == "上傳本機 CSV (Upload CSV)":
        is_custom = True
        uploaded_file = st.sidebar.file_uploader(
            "選擇本機 CSV 檔案 (上限 20MB)：",
            type=["csv"],
            key="custom_csv_file_uploader",
        )
        if uploaded_file is not None:
            loaded_df, err = load_custom_file(uploaded_file)
            if err:
                st.sidebar.error(err)
            else:
                df = loaded_df
                st.sidebar.success(f"成功上傳: {len(df):,} 筆, {df.shape[1]} 欄位")

    # Fallback to 50 startups if custom data not yet provided
    if df is None:
        st.sidebar.warning("尚未載入有效自訂資料，目前暫以預設 50 Startups 顯示。")
        df = load_dataset_by_name(PROBLEM_50_STARTUPS)
        is_custom = False

    # Target & Feature Configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 目標與特徵設定")

    all_cols = list(df.columns)
    if is_custom:
        # User chooses target
        default_idx = len(all_cols) - 1
        target_col = st.sidebar.selectbox("選擇預測目標 (Target Y)：", options=all_cols, index=default_idx)
        # User chooses excluded ID columns
        default_excluded = [c for c in all_cols if c.lower() in ["id", "index", "unnamed: 0"] and c != target_col]
        excluded_cols = st.sidebar.multiselect("排除識別欄位 (ID / Excluded)：", options=[c for c in all_cols if c != target_col], default=default_excluded)
    else:
        st.sidebar.markdown(f"**預測目標 (Target)**: `{target_col}`")
        if excluded_cols:
            st.sidebar.markdown(f"**排除欄位**: `{', '.join(excluded_cols)}`")

    # Available feature count
    available_features = [c for c in all_cols if c != target_col and c not in excluded_cols]
    n_avail = len(available_features)
    is_compliant, feat_msg = validate_feature_count(n_avail)
    if not is_compliant:
        st.sidebar.caption(f"⚠️ {feat_msg}")
    else:
        st.sidebar.caption(f"✅ {feat_msg}")

    # 3. Model & Feature Selection Hyperparameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 模型與特徵選擇參數")

    test_size = st.sidebar.slider("測試集比例 (Test Size)：", min_value=0.10, max_value=0.40, value=DEFAULT_TEST_SIZE, step=0.05)
    random_state = st.sidebar.number_input("隨機種子 (Random State)：", value=DEFAULT_RANDOM_STATE, step=1)

    fs_method = st.sidebar.selectbox(
        "特徵選擇方法 (Feature Selection)：",
        options=FEATURE_SELECTION_METHODS,
        index=0,
    )

    # Dynamic K limit
    max_k = max(1, n_avail)
    default_k = min(5, max_k)
    k_features = st.sidebar.slider(
        "選取特徵數 (K Features)：",
        min_value=1,
        max_value=max_k,
        value=default_k,
        help="設定特徵選擇器欲保留的最佳特徵數目 K",
    )

    cv_folds = st.sidebar.slider("交叉驗證折數 (CV Folds)：", min_value=2, max_value=10, value=DEFAULT_CV_FOLDS)

    # Run Analysis Button
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 執行分析與模型訓練", type="primary", use_container_width=True)

    config = {
        "problem_name": selected_problem,
        "is_custom": is_custom,
        "source_choice": source_choice,
        "target_col": target_col,
        "excluded_cols": excluded_cols,
        "available_features": available_features,
        "test_size": test_size,
        "random_state": random_state,
        "fs_method": fs_method,
        "k_features": k_features,
        "cv_folds": cv_folds,
        "run_button": run_button,
    }

    return df, config
