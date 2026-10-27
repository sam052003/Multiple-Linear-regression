# Google NotebookLM 研究摘要 (真實填寫處)

> **作業規範提示**：
> 老師要求使用 Google NotebookLM 研究同主題網路解法，並將**至少 100 字摘要**放入期末報告。
> 本檔案保留給您填入親自操作 NotebookLM 產生的真實研究內容。請依據 [`docs/notebooklm_research_guide.md`](file:///c:/Users/sam98/Desktop/HW2-Multiple%20Linear%20regression/docs/notebooklm_research_guide.md) 的指引完成後貼入下方。

---

## 狀態：待填寫 (User Action Required)

```markdown
[=== 請將您在 NotebookLM 產生的真實研究摘要文字（至少 100 字）貼至此處 ===]

範例填寫結構參考：
透過 Google NotebookLM 針對多元線性回歸、CRISP-DM 與特徵選擇之網路文獻進行研讀，主要收穫包括三點：
第一，在區間估計方面，深入理解了「母體均值信賴區間 (CI)」與「新觀測值預測區間 (PI)」的本質分野。CI 僅衡量給定特徵下期望值的估計誤差，而 PI 則額外涵蓋單一觀測值本身的隨機擾動 (ε ~ N(0, σ²))，因此預測區間必定更為寬廣。
第二，特徵選擇方法中，Filter 法 (SelectKBest) 雖快但忽略變數間交互作用；Wrapper 法 (RFE, SFS) 能直接考量模型預測反饋但計算代價高昂；Embedded 法 (Lasso) 能在連續凸優化中兼顧懲罰與篩選。
第三，所有特徵縮放與選擇必須嚴格受限於訓練集內，並意識到回歸係數僅代表關聯強度而非因果效果。
```

---

## 使用之 NotebookLM 來源清單
1. Scikit-learn Linear Models User Guide (scikit-learn.org)
2. Scikit-learn Feature Selection Documentation
3. Statsmodels OLS get_prediction Documentation
4. Kaggle Student Performance Factors Dataset Discussion
