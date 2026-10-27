"""
Deployment & Reports UI Component.
Provides downloads for CSV metrics, predictions with intervals, final Markdown and PDF reports.
"""

import os
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd

from src.reporting import export_csv_data, build_markdown_report, generate_pdf_report
from src.config import REPORTS_DIR


def render_report_ui(
    fs_summary_df: Optional[pd.DataFrame],
    eval_results: Optional[Dict[str, Any]],
    pi_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
):
    """
    Render Deployment & Final Reports Tab.
    """
    st.markdown("## 🚀 階段 6：部署應用、分析報告與成果匯出 (Deployment & Reports)")

    st.markdown(
        """
        本分頁整合 CRISP-DM 最終產出：支援模型指標 CSV、預測結果 CSV、靜態圖表下載，
        並提供完整 Markdown 與 PDF 專案期末研究報告。
        """
    )

    # 1. Export Data Section
    st.markdown("### 📥 成果數據下載 (Data & Metrics Exports)")

    exports = export_csv_data(
        fs_summary_df=fs_summary_df,
        eval_results=eval_results,
        pi_df=pi_df,
        target_name=config.get("target_col", "Target"),
    )

    c1, c2 = st.columns(2)
    with c1:
        if "model_metrics.csv" in exports:
            st.download_button(
                label="📊 下載模型評估與特徵選擇指標 CSV (model_metrics.csv)",
                data=exports["model_metrics.csv"].encode("utf-8-sig"),
                file_name="model_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("模型訓練完成後即可下載 model_metrics.csv")

    with c2:
        if "test_predictions_intervals.csv" in exports:
            st.download_button(
                label="📏 下載測試集預測值與 95% 預測區間 CSV (test_predictions_intervals.csv)",
                data=exports["test_predictions_intervals.csv"].encode("utf-8-sig"),
                file_name="test_predictions_intervals.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("模型訓練完成後即可下載 test_predictions_intervals.csv")

    st.markdown("---")

    # 2. Comprehensive Analysis Report Generation (Markdown & PDF)
    st.markdown("### 📑 正式 CRISP-DM 期末分析報告 (Formal Report)")

    # Build report content
    report_md = build_markdown_report(student_summary_df=fs_summary_df)

    # Generate PDF
    pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")
    try:
        generate_pdf_report(report_md, pdf_path)
        pdf_ready = True
    except Exception as e:
        pdf_ready = False
        st.warning(f"PDF 產出提示: {e}")

    r1, r2 = st.columns(2)
    with r1:
        st.download_button(
            label="📄 下載正式分析報告 Markdown (final_report.md)",
            data=report_md.encode("utf-8"),
            file_name="final_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with r2:
        if pdf_ready and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📕 下載正式分析報告 PDF (final_report.pdf)",
                data=pdf_bytes,
                file_name="final_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # In-browser Report Preview
    with st.expander("👀 在瀏覽器中即時預覽完整報告 (final_report.md Preview)", expanded=True):
        st.markdown(report_md)

    st.markdown("---")

    # 3. Required Deliverables & Guides
    st.markdown("### 📌 作業繳交項目導引 (Deliverables Guide)")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            """
            #### 🧠 Google NotebookLM 研究摘要
            - **要求**：使用 Google NotebookLM 研究同主題網路解法，並將至少 100 字摘要納入報告。
            - **操作指引**：請參考 [`docs/notebooklm_research_guide.md`](file:///c:/Users/sam98/Desktop/HW2-Multiple%20Linear%20regression/docs/notebooklm_research_guide.md)。
            - **摘要填寫處**：請將真實摘要填入 [`reports/notebooklm_summary.md`](file:///c:/Users/sam98/Desktop/HW2-Multiple%20Linear%20regression/reports/notebooklm_summary.md)。
            """
        )

    with d2:
        st.markdown(
            """
            #### 💬 ChatGPT / AI 對話紀錄 PDF
            - **要求**：將實際開發過程之對話匯出為 PDF 檔案。
            - **匯出指引**：請參考 [`docs/chatgpt_conversation_export_guide.md`](file:///c:/Users/sam98/Desktop/HW2-Multiple%20Linear%20regression/docs/chatgpt_conversation_export_guide.md)。
            - **存放位置**：請將 PDF 放置於 `reports/ai_conversations/` 資料夾中。
            """
        )

    st.markdown("---")

    # 4. Streamlit Cloud Deployment Readiness
    st.markdown("### 🌐 雲端部署就緒檢查 (Streamlit Cloud Readiness)")
    st.markdown(
        """
        - ✅ **進入點檔案 (Entry Point)**：`app.py`
        - ✅ **相依套件庫**：`requirements.txt` (已鎖定相容版本)
        - ✅ **內建資料來源**：已內嵌預處理樣本，離線與雲端無需外部授權金鑰即可執行
        - ✅ **部署指南**：詳細 Streamlit Community Cloud 連接步驟見 [`docs/deployment.md`](file:///c:/Users/sam98/Desktop/HW2-Multiple%20Linear%20regression/docs/deployment.md)
        """
    )
