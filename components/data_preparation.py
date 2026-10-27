"""
Data Preparation UI Component.
Displays preprocessing pipelines, leakage prevention verification, and train/test splits.
"""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd

from src.preprocessing import PreprocessingArtifacts


def render_data_preparation(artifacts: Optional[PreprocessingArtifacts], config: Dict[str, Any]):
    """
    Render Data Preparation Tab.
    """
    st.markdown("## 🛠️ 階段 3：資料準備與前處理 (Data Preparation)")

    if artifacts is None:
        st.info("💡 請在側邊欄確認參數後，點擊「🚀 執行分析與模型訓練」以擬合前處理 Pipeline。")
        return

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("訓練集筆數 (Train N)", f"{len(artifacts.X_train_proc):,} ({(1-config['test_size'])*100:.0f}%)")
    with c2:
        st.metric("測試集筆數 (Test N)", f"{len(artifacts.X_test_proc):,} ({config['test_size']*100:.0f}%)")
    with c3:
        st.metric("原始輸入特徵數", f"{len(config['available_features'])}")
    with c4:
        st.metric("編碼後特徵數 (Encoded)", f"{len(artifacts.feature_names_out)}")

    st.markdown("---")

    # Anti-leakage checklist
    st.markdown("### 🛡️ 防資料洩漏 (Data Leakage Prevention) 驗證機制")
    st.markdown(
        """
        本系統嚴格遵循機器學習最佳實踐，確保測試集 (Test Set) 僅作為最終無偏評估依據：
        - ✅ **切分優先**：最先執行 `train_test_split`，絕不對全體資料集預先進行標準化或特徵篩選。
        - ✅ **純訓練集擬合**：`SimpleImputer(strategy='median')` 與 `StandardScaler` 之統計量（中位數、均值、標準差）僅由 Training Data 計算。
        - ✅ **類別編碼隔離**：`OneHotEncoder(handle_unknown='ignore', drop='first')` 僅學習訓練集出現的類別，並採用 `drop='first'` 避免虛擬變數陷阱 (Dummy Variable Trap)。
        - ✅ **特徵選擇無洩漏**：所有 Filter、Wrapper、Embedded 挑選程序均在訓練集內擬合。
        """
    )

    st.markdown("---")

    # Preview of preprocessed training data
    with st.expander("🔍 前處理後之訓練特徵矩陣預覽 (X_train Processed - 前 5 筆)", expanded=True):
        st.dataframe(artifacts.X_train_proc.head(5), use_container_width=True)

    with st.expander("📋 編碼後特徵欄位清單 (Encoded Feature Names)", expanded=False):
        st.write(artifacts.feature_names_out)
