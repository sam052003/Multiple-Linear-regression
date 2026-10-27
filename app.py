"""
Multiple Linear Regression Dashboard
Interactive Regression Analysis with CRISP-DM and Feature Selection
Main Application Entry Point
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.config import PROBLEM_50_STARTUPS
from src.preprocessing import prepare_dataset
from src.modeling import train_baseline_model
from src.evaluation import evaluate_regression_model
from src.feature_selection import run_feature_selection_comparison
from src.prediction_intervals import OLSPredictionModel

from components.sidebar import render_sidebar
from components.overview import render_overview
from components.data_understanding import render_data_understanding
from components.data_preparation import render_data_preparation
from components.feature_selection_ui import render_feature_selection_ui
from components.evaluation_ui import render_evaluation
from components.prediction_ui import render_prediction_ui
from components.report_ui import render_report_ui


# Page configuration
st.set_page_config(
    page_title="Multiple Linear Regression Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        padding: 22px 28px;
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #E0E7FF;
        margin-bottom: 0px;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        color: #1E293B !important;
        font-weight: 700;
    }

    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # 1. Render Sidebar and retrieve dataset & config
    df, config = render_sidebar()

    # Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">📈 Multiple Linear Regression Dashboard</div>
            <div class="hero-subtitle">Interactive Regression Analysis with CRISP-DM and Feature Selection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Check if Run Analysis was clicked or if we should auto-train baseline on initial start
    should_train = config["run_button"]
    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False

    # Auto-train on first load for 50 startups to provide immediate wow factor
    if not st.session_state["model_trained"] and config["problem_name"] == PROBLEM_50_STARTUPS and not config["is_custom"]:
        should_train = True

    if should_train:
        with st.spinner("🚀 正在執行前處理、多元線性回歸、5 大特徵選擇與統計推論..."):
            try:
                # Preprocessing
                artifacts = prepare_dataset(
                    df=df,
                    target_col=config["target_col"],
                    excluded_cols=config["excluded_cols"],
                    selected_feature_cols=config["available_features"],
                    test_size=config["test_size"],
                    random_state=config["random_state"],
                    scale_numeric=True,
                )

                # Baseline Model Training
                model_wrapper = train_baseline_model(artifacts.X_train_proc, artifacts.y_train)

                # Evaluation
                eval_results = evaluate_regression_model(
                    model=model_wrapper.estimator,
                    X_train=artifacts.X_train_proc,
                    y_train=artifacts.y_train,
                    X_test=artifacts.X_test_proc,
                    y_test=artifacts.y_test,
                    cv_folds=config["cv_folds"],
                    random_state=config["random_state"],
                )
                eval_results["X_train_proc"] = artifacts.X_train_proc
                eval_results["X_test_proc"] = artifacts.X_test_proc

                # Feature Selection Comparison & Modeling
                fs_summary_df, fs_results_map, fs_models_map = run_feature_selection_comparison(
                    X_train=artifacts.X_train_proc,
                    y_train=artifacts.y_train,
                    X_test=artifacts.X_test_proc,
                    y_test=artifacts.y_test,
                    k=config["k_features"],
                    cv_folds=config["cv_folds"],
                    random_state=config["random_state"],
                    fast_mode=True,
                )

                # Statsmodels OLS for 95% Confidence & Prediction Intervals
                ols_model = OLSPredictionModel()
                ols_model.fit(artifacts.X_train_proc, artifacts.y_train)
                pi_df = ols_model.predict_intervals(artifacts.X_test_proc, alpha=0.05)

                st.session_state["artifacts"] = artifacts
                st.session_state["model_wrapper"] = model_wrapper
                st.session_state["eval_results"] = eval_results
                st.session_state["fs_summary_df"] = fs_summary_df
                st.session_state["fs_results_map"] = fs_results_map
                st.session_state["fs_models_map"] = fs_models_map
                st.session_state["ols_model"] = ols_model
                st.session_state["pi_df"] = pi_df
                st.session_state["model_trained"] = True
                st.sidebar.success("✅ 模型訓練、特徵選擇與統計推論已成功完成！")
            except Exception as e:
                st.sidebar.error(f"❌ 訓練失敗: {e}")

    # 3. Navigation Tabs corresponding to CRISP-DM
    tabs = st.tabs(
        [
            "1. 商業與問題理解 (Overview)",
            "2. 資料理解 (Data Understanding)",
            "3. 資料準備 (Preparation)",
            "4. 特徵選擇與模型 (Feature Selection & Modeling)",
            "5. 模型評估與統計診斷 (Evaluation & Diagnostics)",
            "6. 互動單筆預測 (Interactive Prediction)",
            "7. 部署與報告匯出 (Deployment & Reports)",
        ]
    )

    artifacts = st.session_state.get("artifacts")
    model_wrapper = st.session_state.get("model_wrapper")
    eval_results = st.session_state.get("eval_results")
    fs_summary_df = st.session_state.get("fs_summary_df")
    fs_results_map = st.session_state.get("fs_results_map")
    ols_model = st.session_state.get("ols_model")
    pi_df = st.session_state.get("pi_df")

    with tabs[0]:
        render_overview(df, config)

    with tabs[1]:
        render_data_understanding(df, config)

    with tabs[2]:
        render_data_preparation(artifacts, config)

    with tabs[3]:
        render_feature_selection_ui(fs_summary_df, fs_results_map, config)

    with tabs[4]:
        render_evaluation(eval_results, ols_model, config)

    with tabs[5]:
        render_prediction_ui(artifacts, model_wrapper, ols_model, config)

    with tabs[6]:
        render_report_ui(fs_summary_df, eval_results, pi_df, config)


if __name__ == "__main__":
    main()
