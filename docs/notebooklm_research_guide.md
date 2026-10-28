# Google NotebookLM 研究指南 (NotebookLM Research Guide)

> **作業規格要求**：
> 老師要求使用 Google NotebookLM 研究同主題（Multiple Linear Regression, Feature Selection, CRISP-DM, Model Evaluation, Prediction Interval）之網路解法，並將至少 100 字摘要放入報告。
> 
> ⚠️ **注意事項**：
> 本系統不得虛構或冒充您在 NotebookLM 的實際研究紀錄。請依照本指南完成操作後，將產生的真實摘要填入 [`reports/notebooklm_summary.md`](../reports/notebooklm_summary.md)。

---

## 1. 研究主題
- **主標題**：*Multiple Linear Regression, Feature Selection, CRISP-DM, Model Evaluation and Prediction Interval*
- **研究目標**：探索現代多元線性回歸實踐中，如何結合 CRISP-DM 流程、特徵選擇（Filter vs Wrapper vs Embedded）、統計假設檢驗與預測區間（Prediction Interval vs Confidence Interval）。

---

## 2. 推薦匯入 NotebookLM 的資料來源 (Sources to Add)

開啟 [Google NotebookLM](https://notebooklm.google.com/)，點選「新增記事本 (New Notebook)」，建議加入下列 5~8 個權威來源連結或 PDF：

1. **Scikit-learn 官方線性模型與特徵選擇文件**：
   - [Scikit-learn Linear Regression User Guide](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
   - [Scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html)
   - [Scikit-learn Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html) (特別注意防止 Data Leakage 章節)
2. **Statsmodels 官方 OLS 與預測區間文件**：
   - [Statsmodels OLS Regression Documentation](https://www.statsmodels.org/stable/regression.html)
   - [Statsmodels get_prediction and summary_frame](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_prediction.html)
3. **Kaggle 資料集與討論**：
   - [Kaggle Student Performance Factors Dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
   - [Kaggle Playground Series s4e5 Flood Prediction](https://www.kaggle.com/competitions/playground-series-s4e5)
4. **CRISP-DM 標準流程參考**：
   - [CRISP-DM 1.0 Step-by-Step Data Mining Guide](https://www.the-modeling-agency.com/crisp-dm.pdf)

---

## 3. 建議向 NotebookLM 提出的關鍵問題 (Prompt Queries)

在 NotebookLM 的對話框中輸入下列提問，以生成高品質研究筆記：

### 提問 1 (核心概念比較)
> 「在多元線性回歸中，『95% 信賴區間 (Confidence Interval for Mean)』與『95% 預測區間 (Prediction Interval for Observation)』在統計意義、計算公式與區間寬度上有何本質差異？為何特徵選擇後套用傳統 OLS 區間需要注意 post-selection inference 限制？」

### 提問 2 (特徵選擇方法論)
> 「請比較 Filter Method (如 SelectKBest)、Wrapper Method (如 RFE, SFS) 與 Embedded Method (如 Lasso) 在計算複雜度、特徵交互作用捕捉與防範資料洩漏 (Data Leakage) 上的優缺點。」

### 提問 3 (CRISP-DM 與因果推論)
> 「在分析教育數據（如 Student Performance Factors）時，如何運用 CRISP-DM 流程建立可信的預測模型？為何回歸模型的係數不能直接解釋為因果干預效果？」

---

## 4. 摘要撰寫架構建議 (至少 100 字)
請將 NotebookLM 產生的重點彙整為包含下列三點的結構：
1. **概念理解**：說明多元線性回歸在 CRISP-DM 架構下的角色與 95% 預測區間的統計意義。
2. **方法論反思**：總結 Filter、Wrapper、Embedded 三類特徵選擇的適用情境。
3. **實務啟發**：討論大樣本下統計假設檢定（如常態性與同質變異）的實務解讀方式與防資料洩漏原則。

---

## 5. 填寫與報告整合步驟
完成 NotebookLM 研究後：
1. 開啟專案內的 [`reports/notebooklm_summary.md`](../reports/notebooklm_summary.md)。
2. 將您的真實研究摘要與參考來源填入該檔案。
3. 重新執行 `python scripts/run_experiments.py --report-only`（或完整執行 `python scripts/run_experiments.py`）即可自動讀取 `reports/notebooklm_summary.md`，並同步更新獨立研究報告（`reports/final_report.md` 與 `reports/final_report.pdf`）。
   - **架構邊界說明**：正式研究報告與公開網站完全分離；NotebookLM 研究摘要僅整合至離線獨立研究報告，不會顯示或同步至公開 Web Dashboard。
