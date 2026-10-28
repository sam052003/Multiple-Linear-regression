# Google NotebookLM 文獻研究心得與參考來源

> **研究狀態**：已完成 NotebookLM 實際研究

---

## 研究心得摘要

本研究透過 Google NotebookLM 研讀 CRISP-DM、多元線性回歸及特徵選擇相關文獻，理解資料探勘的六階段流程，以及 Filter、Wrapper 與 Embedded 等方法的差異 [1, 2]。Scikit-learn 文件指出，LassoCV 結合 L1 正則化與交叉驗證選擇懲罰參數 α，並不保證固定選取五個特徵 [2, 3]。

實驗結果顯示，Student Performance 資料集的 Baseline 測試集 R² 為 0.769650，選取五個特徵後最佳 R² 降至 0.666460；Flood Probability 則由 0.844545 降至 0.180733 [6]。兩組結果顯示，特徵縮減不一定改善預測效能，其影響可能與資料中的訊號分布有關 [2, 6]。

在方法論方面，官方文件建議將前處理與特徵選擇整合進 Pipeline，使交叉驗證各折獨立擬合，降低資料洩漏風險 [2, 4]。此外，95% 信賴區間（CI）用於估計條件平均值，95% 預測區間（PI）則針對個別新觀測值 [5]。

本研究的 Student 資料屬於合成資料，無法直接推論真實教育情境中的因果關係 [6]。未來可改進交叉驗證 Pipeline，並加入非線性項與特徵交互作用，以進一步檢驗模型的預測能力 [3, 4, 6]。

---

## 參考文獻

1. **IBM CRISP-DM**
   https://www.ibm.com/docs/zh-tw/spss-modeler/saas?topic=dm-crisp-help-overview

2. **Scikit-learn Feature Selection**
   https://scikit-learn.org/stable/modules/feature_selection.html

3. **Scikit-learn Linear Models**
   https://scikit-learn.org/stable/modules/linear_model.html

4. **Scikit-learn Common Pitfalls**
   https://scikit-learn.org/stable/common_pitfalls.html

5. **Statsmodels get_prediction**
   https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_prediction.html

6. **專案內部研究成果**
   `reports/final_report.pdf`
