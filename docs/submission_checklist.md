# HW2 作業成果與可選繳交資料檢查清單 (Submission Checklist)

本文件彙整 **HW2 — 多元線性回歸與特徵選擇（Multiple Linear Regression & Feature Selection）** 專案內所有核心成果與可選繳交項目之檔案位置、驗證方式與目前完成狀態。

> [!IMPORTANT]
> **架構分離原則**：公開互動式網站（Next.js 14 + FastAPI）僅提供即時回歸建模與視覺化診斷；正式學術研究報告、NotebookLM 研究心得、實驗 Notebook 與 AI 開發對話紀錄均作為 GitHub Repository 獨立檔案維護，不暴露於公開網站介面或公開 API。

---

## 1. 核心與可選繳交項目總覽

| 類別 | 項目名稱 | 檔案路徑 | 目前狀態 | 備註說明 |
| :--- | :--- | :--- | :---: | :--- |
| **互動系統** | Next.js 14 + FastAPI 互動分析平台 | [`app/`](../app/)、[`api/`](../api/)、[`src/`](../src/) | ✅ 已完成 | 繁體中文（`zh-Hant-TW`）CRISP-DM 六階段介面、3 組內建資料集與自訂 CSV（含逐跳 SSRF 防護） |
| **系統截圖** | 6 張實際執行系統畫面截圖 | [`assets/screenshots/`](../assets/screenshots/) | ✅ 已完成 | `01_overview.png` ～ `06_custom_dataset.png` 均由真實 Next.js 系統畫面擷取 |
| **研究報告** | 正式 CRISP-DM 研究報告（PDF） | [`reports/final_report.pdf`](../reports/final_report.pdf) | ✅ 已完成 | 共 13 頁出版級繁體中文排版，含完整六方法比較表、統計檢定、NotebookLM 心得與參考文獻 |
| **研究報告** | 正式 CRISP-DM 研究報告（Markdown） | [`reports/final_report.md`](../reports/final_report.md) | ✅ 已完成 | 與 PDF 同步由 `src/reporting.py` 動態綁定實驗 CSV 產出 |
| **文獻研究** | Google NotebookLM 實際研究心得與書目 | [`reports/notebooklm_summary.md`](../reports/notebooklm_summary.md) | ✅ 已完成 | 已整合使用者真實 NotebookLM 研讀心得與 6 項經核實之官方文獻來源（永久寫入報告第 9.1 節與第 10 節） |
| **實驗程式** | Student Performance 主要研究 Notebook | [`notebooks/student_performance_analysis.ipynb`](../notebooks/student_performance_analysis.ipynb) | ✅ 已完成 | 11 個儲存格完整真實執行完畢，含內嵌圖表、五大假設檢定與 95% CI/PI 分析 |
| **實驗程式** | Flood Probability 對照研究 Notebook | [`notebooks/flood_prediction_analysis.ipynb`](../notebooks/flood_prediction_analysis.ipynb) | ✅ 已完成 | 5 個儲存格完整真實執行完畢，呈現分散型訊號特徵縮減對比（$R^2: 0.844545 \to 0.180733$） |
| **實驗數據** | 模型效能比較與預測區間 CSV | [`reports/results/`](../reports/results/) | ✅ 已完成 | 含 `student_feature_selection_comparison.csv`、`flood_feature_selection_comparison.csv`、`student_predictions_intervals.csv` |
| **研究圖表** | 7 張高解析度研究圖表 | [`reports/figures/`](../reports/figures/) | ✅ 已完成 | 完整清單見下方第 2 節 |
| **AI 對話證據** | ChatGPT 完整開發對話紀錄 PDF | [`reports/ai_conversations/chatgpt_conversation_record.pdf`](../reports/ai_conversations/) | ⏳ **待使用者匯入** | 本專案僅提交使用者真實 ChatGPT 對話紀錄 PDF；目前已保留存放目錄與匯出指南，**絕不虛構或生成假對話** |

---

## 2. 七張研究圖表檢查明細 (`reports/figures/`)

| 編號 | 圖表檔案 | 狀態 | 用途與對應分析 |
| :---: | :--- | :---: | :--- |
| 1 | [`reports/figures/crisp_dm_process.png`](../reports/figures/crisp_dm_process.png) | ✅ 已完成 | CRISP-DM 六階段研究方法論流程圖 |
| 2 | [`reports/figures/feature_selection_comparison.png`](../reports/figures/feature_selection_comparison.png) | ✅ 已完成 | Student Performance 全特徵 Baseline 與 5 種特徵選擇方法之測試集 $R^2$／CV $R^2$ 比較 |
| 3 | [`reports/figures/actual_vs_predicted.png`](../reports/figures/actual_vs_predicted.png) | ✅ 已完成 | 測試集實際考試成績 vs. 模型預測值散佈圖（$N_{\text{test}}=1,322$） |
| 4 | [`reports/figures/residuals_vs_fitted.png`](../reports/figures/residuals_vs_fitted.png) | ✅ 已完成 | 殘差對擬合值散佈圖（檢驗線性性與同質變異性） |
| 5 | [`reports/figures/residuals_distribution_qq.png`](../reports/figures/residuals_distribution_qq.png) | ✅ 已完成 | 殘差直方圖與常態 Q-Q 圖（檢驗殘差常態性與尾端行為） |
| 6 | [`reports/figures/student_pi_intervals.png`](../reports/figures/student_pi_intervals.png) | ✅ 已完成 | 95% 母體均值信賴區間（CI）與 95% 個別觀測值預測區間（PI）寬度與涵蓋率視覺化 |
| 7 | [`reports/figures/student_vs_flood_comparison.png`](../reports/figures/student_vs_flood_comparison.png) | ✅ 已完成 | Student（集中型訊號）vs. Flood（分散型訊號）特徵縮減保留率跨資料集對照圖 |

---

## 3. 待使用者手動完成項目（ChatGPT 對話 PDF 匯入步驟）

目前 [`reports/ai_conversations/`](../reports/ai_conversations/) 目錄下僅保留說明檔 [`README.md`](../reports/ai_conversations/README.md)，尚未放入對話 PDF。繳交前若欲附上 AI 對話紀錄作為可選佐證資料，請依下列步驟匯入：

1. 開啟您於本作業期間實際使用的 **ChatGPT** 對話頁面。
2. 參考 [`docs/chatgpt_conversation_export_guide.md`](chatgpt_conversation_export_guide.md)，使用瀏覽器 `Ctrl + P`（列印）➔ **另存為 PDF**。
3. 將匯出之真實對話檔案命名並存放於：
   ```text
   reports/ai_conversations/chatgpt_conversation_record.pdf
   ```
4. （選用）若希望重新產出之 `reports/final_report.pdf` 第 9.2 節一併記錄，可於匯入後視需求更新文字或直接將該對話 PDF 隨同作業附件一併繳交。

---

## 4. 自動化驗證指令

```powershell
# 1. 執行完整 Python 單元測試、API 整合測試與報告無副作用驗證（57 passed）
pytest tests/ -v

# 2. 驗證 Next.js 14 前端型別檢查與生產環境建置
npm run build
```

