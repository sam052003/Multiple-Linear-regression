# 系統操作截圖與研究圖表索引 (Screenshots & Research Figures)

本目錄存放 **MLR Studio（Next.js 14 + FastAPI 多元線性回歸分析平台）** 實際執行擷取之系統畫面截圖。正式研究報告使用之靜態統計圖表則獨立存放於 [`../../reports/figures/`](../../reports/figures/)。

---

## 1. 互動式網站系統截圖清單 (`assets/screenshots/`)

所有畫面均由實際運行之 Next.js 14 App Router 前端與 FastAPI 後端分析服務擷取（主要展示正式研究資料集 `Student Performance` 與自訂資料集模組，不含任何私人帳號或金鑰資訊）：

| 檔案名稱 | 對應功能／CRISP-DM 階段 | 畫面重點說明 |
| :--- | :--- | :--- |
| [`01_overview.png`](01_overview.png) | Stage 1：問題理解（Overview） | `Student Performance`（6,607 筆、19 個原始特徵、符合 14~39 特徵規格）、目標變數指定面板與前 10 筆資料預覽 |
| [`02_data_understanding.png`](02_data_understanding.png) | Stage 2：資料理解（Data Understanding） | `Exam_Score` 統計輪廓、缺失值品質診斷表、皮爾森相關係數互動長條圖與數值／類別特徵清單 |
| [`03_feature_selection.png`](03_feature_selection.png) | Stage 4：特徵選擇與建模（Feature Selection） | 全特徵 Baseline 與 5 種特徵選擇演算法（Lasso、Permutation Importance、SelectKBest、SFS Forward、RFE）之測試集 R²、5 折交叉驗證 R² 圖表與完整比較表 |
| [`04_evaluation.png`](04_evaluation.png) | Stage 5：模型評估與診斷（Evaluation & Diagnostics） | 訓練／測試集誤差指標、95% CI vs. 95% PI 帶狀比較圖、實際值對預測值散布圖、殘差散布圖與高斯-馬可夫五大假設檢定（含 VIF 共線性表） |
| [`05_prediction.png`](05_prediction.png) | Stage 6：互動預測與應用（Interactive Prediction） | 19 個特徵互動輸入表單、Statsmodels 即時點預測值、95% 信賴區間（CI）、95% 預測區間（PI）與預測紀錄 CSV 匯出按鈕 |
| [`06_custom_dataset.png`](06_custom_dataset.png) | 自訂資料集載入模組（Custom Dataset） | 支援本機 CSV 檔案上傳（上限 4.5 MB／5,000 筆、自動預讀欄位與推薦數值 Target）及公開 HTTPS 網址載入（內建逐跳 SSRF 安全防護） |

---

## 2. 重新擷取系統畫面步驟（Next.js + FastAPI / Vercel）

若需在本機或 Vercel 部署環境重新擷取最新畫面，請依下列步驟操作：

1. **啟動後端 FastAPI 服務（Port 8000）**：
   ```powershell
   python -m uvicorn api.index:app --port 8000
   ```
2. **啟動前端 Next.js 伺服器（Port 3000）**：
   ```powershell
   npm run dev
   ```
3. **開啟瀏覽器擷取畫面**：
   - 本機開發網址：`http://localhost:3000`（或開啟已通過驗證之 Vercel 部署網址）。
   - 於頂端資料集選單切換至 `Student Performance — 學生成績預測（主要研究）`，依序操作各 CRISP-DM 分頁與自訂資料集視窗，擷取高清 PNG 並儲存於 `assets/screenshots/`。

---

## 3. 獨立研究報告統計圖表清單 (`reports/figures/`)

以下 7 張高解析度學術圖表由離線實驗腳本 [`../../scripts/run_experiments.py`](../../scripts/run_experiments.py) 實際運算產出，供正式研究報告（[`../../reports/final_report.pdf`](../../reports/final_report.pdf)）引用：

1. [`crisp_dm_process.png`](../../reports/figures/crisp_dm_process.png)：CRISP-DM 六階段研究架構流程圖
2. [`feature_selection_comparison.png`](../../reports/figures/feature_selection_comparison.png)：Student Performance 六種特徵選擇方法效能比較圖
3. [`actual_vs_predicted.png`](../../reports/figures/actual_vs_predicted.png)：測試集實際值對預測值散佈圖（Actual vs. Predicted）
4. [`residuals_vs_fitted.png`](../../reports/figures/residuals_vs_fitted.png)：殘差對擬合值散佈圖（Residuals vs. Fitted Values）
5. [`residuals_distribution_qq.png`](../../reports/figures/residuals_distribution_qq.png)：殘差直方圖與常態 Q-Q 檢定圖（Normality Diagnostics）
6. [`student_pi_intervals.png`](../../reports/figures/student_pi_intervals.png)：95% 母體均值信賴區間（CI）與 95% 個別觀測值預測區間（PI）實證對照圖
7. [`student_vs_flood_comparison.png`](../../reports/figures/student_vs_flood_comparison.png)：Student 集中型訊號與 Flood 分散型訊號跨資料集特徵縮減對照圖
