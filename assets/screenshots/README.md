# 系統操作截圖與成果圖表 (Screenshots & Infographics)

本目錄存放本專案實際運作之畫面截圖與核心資訊圖表。

---

## 1. 系統運行截圖指引 (Manual Screenshot Guide)
若欲在您本機擷取最新畫面繳交，請依照下列步驟：
1. 啟動 Streamlit 服務：
   ```bash
   streamlit run app.py
   ```
2. 在瀏覽器中開啟 `http://localhost:8501`。
3. 依序切換下列各分頁並使用截圖工具 (如 Windows `Win + Shift + S`) 擷取畫面並存於此目錄：
   - `01_dashboard_overview.png`：首頁商業理解與資料集 KPI 卡片
   - `02_data_understanding.png`：目標變數直方圖與數值特徵相關熱圖
   - `03_data_preparation.png`：防資料洩漏驗證與特徵矩陣
   - `04_feature_selection.png`：6 大方法效能評比表與特徵排名圖
   - `05_evaluation_diagnostics.png`：實際值 vs 預測值與 95% 預測區間圖
   - `06_interactive_prediction.png`：單筆樣本預測表單與 CI/PI 結果展示

---

## 2. 核心成果資訊圖表 (Core Infographics)
以下圖表由實際實驗腳本 (`scripts/run_experiments.py`) 自動產出於 `reports/figures/`：
1. **CRISP-DM 流程圖**：`reports/figures/crisp_dm_process.png`
2. **特徵選擇效能比較柱狀圖**：`reports/figures/feature_selection_comparison.png`
3. **實際值 vs 預測值散佈圖**：`reports/figures/actual_vs_predicted.png`
4. **殘差對擬合值散佈圖**：`reports/figures/residuals_vs_fitted.png`
