"""
Overview and Business Understanding UI Component.
Presents the CRISP-DM business goals, dataset profile cards, and research context.
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd

from src.config import PROBLEM_METADATA
from src.data_validation import validate_feature_count


def render_overview(df: pd.DataFrame, config: Dict[str, Any]):
    """
    Render Business Understanding & Project Overview Tab.
    """
    st.markdown("## 📌 階段 1：商業與研究問題理解 (Business Understanding)")

    problem_name = config["problem_name"]
    is_custom = config["is_custom"]
    meta = PROBLEM_METADATA.get(problem_name, {})

    if is_custom:
        st.info("💡 **目前分析模式：自訂資料集 (Custom Dataset)**。下方展示您所上傳或由 URL 讀入的資料結構。")

    # Metric KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("資料總筆數 (Samples)", f"{len(df):,}")
    with col2:
        st.metric("總欄位數 (Columns)", f"{df.shape[1]}")
    with col3:
        st.metric("原始特徵數 (Original Features)", f"{len(config['available_features'])}")
    with col4:
        st.metric("預測目標 (Target Y)", f"{config['target_col']}")

    # Feature count badge
    is_compliant, feat_msg = validate_feature_count(len(config["available_features"]))
    if is_compliant:
        st.success(f"🎓 **規格檢驗**：{feat_msg}")
    else:
        st.warning(f"⚠️ **規格提醒**：{feat_msg}")

    st.markdown("---")

    # Two-column research layout
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown("### 🎯 問題背景與研究動機")
        if is_custom:
            st.markdown(
                """
                使用者自訂之多元線性回歸資料集。系統將自動辨識數值與類別特徵，建立防止資料洩漏的前處理 Pipeline，
                並提供特徵選擇、統計假設診斷與 95% 預測區間。
                """
            )
        else:
            st.markdown(f"**研究主題**：{meta.get('title', problem_name)}")
            st.markdown(f"**分類標籤**：`{meta.get('tag', 'General Study')}`")
            st.markdown(f"**資料來源**：{meta.get('source', 'Public')}")
            st.markdown(f"**問題描述**：\n{meta.get('description', '')}")

        st.markdown("#### 🔬 核心研究問題")
        if problem_name == "Student Performance Factors — Exam Score Prediction":
            st.markdown(
                """
                > **「哪些學習行為與環境因素與學生考試成績 (Exam_Score) 具有最高的預測關聯性？透過不同的特徵選擇演算法（Filter, Wrapper, Embedded, Model-based），是否能在精簡特徵數量的同時維持甚至提升預測準確度？」**
                """
            )
        elif problem_name == "Flood Probability Prediction":
            st.markdown(
                """
                > **「在大規模氣象與水文合成特徵中，各項環境指標對洪水機率 (FloodProbability) 的線性預測貢獻為何？多元線性回歸在合成資料上的預測表現與局限性為何？」**
                """
            )
        else:
            st.markdown(
                """
                > **「在新創公司營運中，研發費用、行政支出與行銷花費如何共同影響公司淨利？類別變數 (State) 經虛擬變數編碼後的影響為何？」**
                """
            )

    with right_col:
        st.markdown("### ⚠️ 資料限制與因果推論警示")
        if not is_custom:
            st.markdown(f"**已知限制**：\n{meta.get('limitations', '')}")

        st.markdown(
            """
            > [!IMPORTANT]
            > **關聯性 $\neq$ 因果性 (Correlation is NOT Causation)**：
            > 多元線性回歸所估計之迴歸係數 ($\beta_j$) 僅代表在控制其他變數下的**統計預測關聯**。在無隨機對照實驗 (RCT) 或外生工具變數之條件下，絕不可將係數逕自解釋為政策干預之因果效果！
            """
        )

        st.markdown("### 🔄 CRISP-DM 流程狀態")
        st.markdown(
            """
            - ✅ **1. 商業與研究理解**：已定義目標與評估指標
            - 🔍 **2. 資料理解**：請至「Data Understanding」分頁探索分佈與品質
            - 🛠️ **3. 資料準備**：Pipeline 嚴格於 Training 集 fit，杜絕資料洩漏
            - 🧠 **4. 模型建立**：支援 Baseline 與 5 種特徵選擇方法
            - 📈 **5. 模型評估**：MAE, RMSE, R², 95% 預測區間與五大假設檢定
            - 🚀 **6. 部署應用**：互動式單筆預測、圖表與報告匯出
            """
        )
