"""
Interactive Prediction UI Component.
Allows users to input custom feature values and compute point predictions with 95% Prediction Intervals.
"""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

from src.preprocessing import PreprocessingArtifacts
from src.modeling import RegressionModelWrapper
from src.prediction_intervals import OLSPredictionModel


def render_prediction_ui(
    artifacts: Optional[PreprocessingArtifacts],
    model_wrapper: Optional[RegressionModelWrapper],
    ols_model: Optional[OLSPredictionModel],
    config: Dict[str, Any],
):
    """
    Render Interactive Prediction Tab.
    """
    st.markdown("## 🎯 互動式單筆樣本預測與 95% 預測區間 (Interactive Prediction)")

    if artifacts is None or model_wrapper is None or not model_wrapper.fitted:
        st.warning("⚠️ 請先在側邊欄點擊「🚀 執行分析與模型訓練」完成模型訓練後，再進行互動預測。")
        return

    st.markdown(
        """
        請在下方調整各特徵數值，系統將自動套用相同的資料前處理流程（標準化與 One-Hot Encoding），
        並計算**點預測值**、**95% 母體平均信賴區間 (CI)** 與**95% 個別觀測值預測區間 (PI)**。
        """
    )

    X_train_raw = artifacts.X_train_raw
    input_values = {}

    # Display form in 2 or 3 responsive columns
    st.markdown("### 📝 特徵輸入表單 (Feature Inputs)")
    cols = st.columns(3)

    for i, col_name in enumerate(X_train_raw.columns):
        col_widget = cols[i % 3]
        series = X_train_raw[col_name]

        if col_name in artifacts.numeric_features:
            min_v = float(series.min()) if not series.empty else 0.0
            max_v = float(series.max()) if not series.empty else 100.0
            med_v = float(series.median()) if not series.empty else 50.0
            step_v = (max_v - min_v) / 100.0 if max_v > min_v else 1.0

            val = col_widget.number_input(
                f"{col_name} (數值)",
                min_value=min_v,
                max_value=max_v,
                value=med_v,
                step=float(round(step_v, 2)),
                key=f"input_{col_name}",
            )
            input_values[col_name] = val
        else:
            categories = sorted([str(c) for c in series.dropna().unique()])
            default_cat = str(series.mode()[0]) if not series.empty else categories[0]
            val = col_widget.selectbox(
                f"{col_name} (類別)",
                options=categories,
                index=categories.index(default_cat) if default_cat in categories else 0,
                key=f"input_{col_name}",
            )
            input_values[col_name] = val

    predict_btn = st.button("🔮 執行即時預測 (Generate Prediction)", type="primary", use_container_width=True)

    if predict_btn or "last_prediction" in st.session_state:
        # Build 1-row dataframe
        user_df = pd.DataFrame([input_values])

        # Cast categorical to str
        for c in artifacts.categorical_features:
            user_df[c] = user_df[c].astype(str)

        try:
            # Transform via fitted ColumnTransformer
            user_proc_arr = artifacts.preprocessor.transform(user_df)
            user_proc_df = pd.DataFrame(
                user_proc_arr,
                columns=artifacts.feature_names_out,
                index=[0],
            )

            # Point prediction from Scikit-Learn
            pred_val = float(model_wrapper.predict(user_proc_df)[0])

            # Prediction Intervals from Statsmodels OLS
            ci_pi_frame = None
            if ols_model and ols_model.ols_results:
                ci_pi_frame = ols_model.predict_intervals(user_proc_df, alpha=0.05)

            st.markdown("---")
            st.markdown("### 🏆 預測結果展示 (Prediction Output)")

            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric(
                    f"預測點估計值 ({config['target_col']})",
                    f"{pred_val:.4f}",
                )

            with p2:
                if ci_pi_frame is not None:
                    mean_low = float(ci_pi_frame["mean_ci_lower"].iloc[0])
                    mean_upp = float(ci_pi_frame["mean_ci_upper"].iloc[0])
                    st.metric(
                        "95% 母體均值信賴區間 (CI for Mean)",
                        f"[{mean_low:.2f}, {mean_upp:.2f}]",
                        help="代表在相同特徵條件下，整體母體的『平均目標期望值』有 95% 機率落在此區間內。",
                    )
                else:
                    st.metric("95% 信賴區間 (CI)", "N/A")

            with p3:
                if ci_pi_frame is not None:
                    obs_low = float(ci_pi_frame["obs_ci_lower"].iloc[0])
                    obs_upp = float(ci_pi_frame["obs_ci_upper"].iloc[0])
                    st.metric(
                        "95% 個別觀測值預測區間 (PI for Observation)",
                        f"[{obs_low:.2f}, {obs_upp:.2f}]",
                        help="代表『單一具體的新觀測值』有 95% 機率落在此區間內。包含個體隨機誤差 (σ²)，必定寬於 CI。",
                    )
                else:
                    st.metric("95% 預測區間 (PI)", "N/A")

            # Explanation of CI vs PI
            st.info(
                """
                💡 **統計觀念解析：信賴區間 (CI) vs 預測區間 (PI)**
                - **95% Confidence Interval for Mean (信賴區間)**：估計的是「母體平均反應 $\mathbb{E}[Y|X]$」的不確定性。若有成千上萬個具備相同特徵的樣本，他們的平均值有 95% 機率落在 CI 內。
                - **95% Prediction Interval for Observation (預測區間)**：估計的是「單一全新新樣本 $Y_{new}$」的落點。因為除了估計係數的不確定性外，還必須納入個別觀測值自身的隨機誤差 $\epsilon \sim \mathcal{N}(0, \sigma^2)$，因此**預測區間 (PI) 範圍必定明顯寬於信賴區間 (CI)**。
                - **限制揭露**：此區間基於 OLS 殘差同質且常態之統計假設；在經過資料驅動特徵選擇後，區間可能存在 Post-selection inference 膨脹效應。
                """
            )

            # Download single prediction
            out_dict = dict(input_values)
            out_dict["Predicted_" + config["target_col"]] = pred_val
            if ci_pi_frame is not None:
                out_dict["CI_Lower_95"] = mean_low
                out_dict["CI_Upper_95"] = mean_upp
                out_dict["PI_Lower_95"] = obs_low
                out_dict["PI_Upper_95"] = obs_upp

            pred_csv = pd.DataFrame([out_dict]).to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 下載本次預測結果 CSV",
                data=pred_csv,
                file_name=f"single_prediction_{config['target_col']}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"預測運算失敗: {e}")
