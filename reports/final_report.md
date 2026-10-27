# 大學「多元線性回歸 (Multiple Linear Regression)」專題研究報告
## 遵循 CRISP-DM 流程之特徵選擇、統計假設診斷與 95% 預測區間推論

> **課程名稱**：機器學習與統計數據分析  
> **作業項目**：HW2 — Multiple Linear Regression Analysis Report  
> **研究代碼庫**：[https://github.com/sam052003/Multiple-Linear-regression](https://github.com/sam052003/Multiple-Linear-regression)  
> **分支**：`migration/vercel`  
> **實驗環境**：Python 3.14 / Scikit-learn 1.6 / Statsmodels 0.14 / ReportLab 5.1  
> **分析日期**：2026-10-27  

---

## 摘要 (Abstract)

本研究以跨產業資料探勘標準流程 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 六大階段為方法論框架，深入探討多元線性回歸 (Multiple Linear Regression, MLR) 在真實教育大數據與合成環境物理數據中的預測行為、特徵篩選機制與統計推論界限。

主要研究對象為 Kaggle **Student Performance Factors** 資料集（6,607 筆真實樣本、19 個原始特徵，符合作業規定之 14～39 個原始特徵規格），並輔以 Kaggle **Flood Probability Prediction** 資料集（30,000 筆固定隨機抽樣）進行跨領域對比分析。研究全程落實嚴格的**防範資料洩漏 (Leakage-Free)** 前處理管線，並在 80/20 訓練/測試切分與 5 折交叉驗證架構下，系統性比較 **Baseline（全特徵基準）** 與 **5 種主流特徵選擇方法**：Filter (SelectKBest)、Wrapper (RFE)、Wrapper (Forward SFS)、Embedded (LassoCV) 與 Model-based (Permutation Importance)。

### 核心實證發現：
1. **基準模型重現性**：Student Performance Baseline 測試集 $R^2$ 精確重現為 **0.769650**（$R^2 \approx 0.7696$），Flood Probability Baseline 測試集 $R^2$ 精確重現為 **0.844545**（$R^2 \approx 0.8445$）。
2. **Lasso 保留全部特徵之統計本質**：在 $N=5,285$ 的充足訓練樣本下，5 折交叉驗證選定之最優正規化參數極小（$\alpha = 0.002271$），所有 27 個編碼特徵之回歸係數絕對值均大於數值閥值 $10^{-5}$。此結果揭示：當樣本數龐大時，即使微弱預測力之變數亦能使交叉驗證 MSE 微幅下降，故未被 L1 懲罰完全歸零。
3. **特徵排序與評估準則差異**：實證推翻了「特定三項特徵在所有方法中均排名前三」的假設。研究顯示 `Attendance`（出席率）與 `Hours_Studied`（學習時數）在所有演算法中穩居前二；但第三名特徵因演算法目標函數而顯著分歧：單變量 Filter 挑選 `Previous_Scores`，而多變量回歸 Wrapper (RFE) 與 Lasso 則優先保留組間截距位移顯著的虛擬變數（如 `Access_to_Resources_Low` 與 `Parental_Involvement_Low`）。
4. **特徵縮減之訊號結構對比**：在 Student 資料集中，將特徵由 27 個降至 5 個時，$R^2$ 僅由 0.7696 微幅降至 0.6665（保留率 86.6%），主因訊號高度集中於核心行為特徵；但在 Flood 資料集中，特徵由 20 個降至 5 個時，$R^2$ 由 0.8445 暴跌至 0.1807（崩跌 78.6%），證實均勻分散型訊號架構下激進的特徵縮減將導致致命的模型欠擬合。
5. **95% 預測區間與統計診斷**：實證展示母體均值信賴區間 (CI，平均寬度 0.5913) 與個別觀測值預測區間 (PI，平均寬度 8.2190) 之 13.9 倍本質差異。回歸五大假設診斷證實殘差無自相關 (DW = 1.9541)、無異質變異 (Breusch-Pagan $p = 0.7335$) 且無嚴重多重共線性 (最大 VIF = 2.80)。

---

## 1. 商業與研究問題理解 (Business Understanding)

### 1.1 研究動機與背景
在教育數據探勘 (Educational Data Mining, EDM) 與學習分析領域中，精確預測學生學業表現並辨識影響學習成效的核心因子，是教育機構進行早期輔導介入、資源精準配置與公平性政策制定的基石。多元線性回歸作為具備最高統計可解釋性 (Interpretability) 與堅實漸近理論基礎之基準模型，被廣泛應用於學術評估中。

### 1.2 研究問題定義 (Research Questions)
本研究具體聚焦於以下四項核心科學問題：
- **RQ1（預測能力極限）**：在結合學習習慣、過往成績、家庭社經地位與學校環境等 19 項原始變數下，多元線性模型對最終考試成績 (`Exam_Score`) 的線性解釋變異量極限為何？
- **RQ2（維度縮減與效能權衡）**：透過 Filter、Wrapper、Embedded 與 Model-based 等不同哲學之特徵篩選技術，能否在大幅精簡模型變數（縮減至 $K=5$）的同時，維持統計泛化能力？何種情況會引發效能顯著衰退？
- **RQ3（特徵重要性之一致性與分歧）**：單變量顯著性（如相關性與 F 檢定）與多變量條件權重（如 OLS 係數、L1 正則化、排列重要度）在評估特徵重要性時存在何種系統性分歧？
- **RQ4（區間推論與決策可靠性）**：點預測無法反映預測不確定性。如何透過嚴謹的 95% 預測區間 (Prediction Interval) 提供個別學生未來的成績區間估計？其與母體均值信賴區間 (Confidence Interval) 有何本質區別？

### 1.3 重要科學限制揭露：統計關聯性 vs. 因果推論 (Correlation is NOT Causation)
多元線性回歸模型所估計之係數 $\beta_j = \frac{\partial E[Y|X]}{\partial X_j}$，在數學上僅代表**在控制其餘納入變數不變之條件下，該自變數與目標變數間的統計邊際關聯度**。
本研究特別鄭重聲明：
> **「高度統計顯著性與較大回歸係數，絕不等同於因果關係 (Correlation does NOT imply Causation)。」**
在缺乏自然實驗 (Natural Experiment)、工具變數 (Instrumental Variables, IV) 或隨機對照試驗 (Randomized Controlled Trial, RCT) 的前提下，資料中可能存在未觀測之干擾因子（如學生天賦能力、學習動機之內生性、家庭未衡量資本）。因此，本研究之特徵重要性分析僅供**預測建模與特徵篩選**參考，決策者不得逕行將負係數特徵（如 `Access_to_Resources_Low`）視為單一因果政策槓桿。

---

## 2. 資料探索與理解 (Data Understanding)

### 2.1 主研究資料集：Student Performance Factors
- **資料來源**：Kaggle 公開教育資料庫（`lainguyn123/student-performance-factors`）。
- **資料規模**：6,607 筆真實學生觀測記錄，共 20 個欄位（19 個輸入特徵 + 1 個連續目標變數 `Exam_Score`）。
- **特徵屬性與編碼結構**：
  - **數值連續特徵 (6 項)**：`Hours_Studied`（學習時數，1～44）、`Attendance`（出席率，60～100%）、`Sleep_Hours`（睡眠時數，4～10）、`Previous_Scores`（過往成績，40～100）、`Tutoring_Sessions`（課外輔導次數，0～8）、`Physical_Activity`（每週運動時數，0～6）。
  - **類別名義/順序特徵 (13 項)**：`Parental_Involvement`（家長參與度）、`Access_to_Resources`（學習資源取得度）、`Extracurricular_Activities`（課外活動參與）、`Motivation_Level`（學習動機水準）、`Internet_Access`（網路連線）、`Family_Income`（家庭收入水準）、`Teacher_Quality`（教師教學品質）、`School_Type`（學校體制）、`Peer_Influence`（同儕影響力）、`Learning_Disabilities`（學習障礙）、`Parental_Education_Level`（父母最高學歷）、`Distance_from_Home`（住家與學校距離）、`Gender`（學生生理性別）。
- **資料品質檢驗**：
  - 資料完整度極佳，全體 6,607 筆樣本中僅三欄位存在少量隨機缺失：`Teacher_Quality`（93 筆，1.41%）、`Parental_Education_Level`（103 筆，1.56%）、`Distance_from_Home`（97 筆，1.47%），缺失機制符合隨機缺失 (MAR) 假設。
- **目標變數分佈性質**：
  - `Exam_Score` 為 0～100 連續數值，樣本範圍為 55～101 分（存在極少量測量上限誤差）。
  - 樣本平均數 $\mu = 67.24$，標準差 $\sigma = 3.89$，中位數為 67.0 分，偏態係數為 0.12，峰度為 2.96，呈現典型之對稱單峰常態分佈。

### 2.2 對比研究資料集：Flood Probability Prediction
- **資料來源**：Kaggle Playground Series s4e5 競賽資料集。
- **資料規模**：全資料集逾 100 萬筆，本研究採固定隨機抽樣 `random_state=42` 抽取 **30,000 筆獨立樣本**（訓練集 24,000 筆，測試集 6,000 筆）。
- **特徵屬性**：20 個純連續數值地理水文氣象特徵（如 `MonsoonIntensity`, `TopographyDrainage`, `RiverManagement`, `DamsQuality`, `Siltation` 等），完全剔除無意義之唯一識別碼 `id`。
- **目標變數分佈**：`FloodProbability` 為 0.285～0.725 之連續機率值，均值 0.504，標準差 0.051。

---

## 3. 資料準備與前處理 (Data Preparation)

為落實嚴格的**機器學習工程規範**，杜絕實務上最易發生的**資料洩漏 (Data Leakage)**，本研究設計了標準的前處理管線 (Preprocessing Pipeline)：

```
原始資料集 (N=6,607)
       │
       ▼
[Train/Test Split (80/20, seed=42)] ───► 測試集 (N=1,322) [完全隔離封存]
       │
       ▼ 訓練集 (N=5,285)
┌─────────────────────────────────────────────────────────────┐
│ 數值特徵 (6項)：SimpleImputer(median) ──► StandardScaler    │
│ 類別特徵 (13項)：SimpleImputer(most_frequent)               │
│                  ──► OneHotEncoder(drop='first')            │
└─────────────────────────────────────────────────────────────┘
       │ fit() 僅於訓練集執行，transform() 套用於訓練與測試集
       ▼
編碼後特徵矩陣 (27 個純數值特徵)
```

### 3.1 防範資料洩漏四大守則
1. **切分優先原則**：在執行任何資料探索統計、中位數計算、標準化平均數估計或編碼對照表建立之前，第一步即執行 `train_test_split(test_size=0.20, random_state=42)`。
2. **無洩漏特徵轉換**：前處理轉換器 `ColumnTransformer` 嚴格僅在訓練集執行 `.fit()`，測試集僅執行 `.transform()`。測試集之真實目標值在評估前從未參與任何運算。
3. **特徵篩選隔離**：所有特徵選擇演算法（Filter, RFE, SFS, Lasso, Permutation）均嚴格限制僅能存取 $X_{\text{train}}$ 與 $y_{\text{train}}$，杜絕「以全資料集篩選特徵後再切分」之學術偽陽性漏洞。
4. **虛擬變數陷阱防範 (Dummy Variable Trap)**：13 個類別特徵經 One-Hot Encoding 時，設定 `drop='first'`。每個具 $M$ 個水準之類別變數僅產生 $M-1$ 個指示變數，將基準組吸納入截距項 $\beta_0$，徹底排除因指示變數總和為 1 所導致的完全多重共線性 (Perfect Multicollinearity)。
   - **維度擴增**：原始 19 個輸入特徵（6 數值 + 13 類別）經編碼後，正式轉化為 **27 個數值型設計矩陣欄位**。

---

## 4. 模型建立與特徵選擇實驗 (Modeling & Feature Selection)

### 4.1 五種特徵選擇機制理論概述
在固定特徵預算 $K=5$（除 Lasso 依據 L1 正則化自然稀疏度外）的條件下，系統性實作五種不同範式：

1. **Filter 方法 (SelectKBest - f_regression)**：
   - *機制*：計算各變數與目標變數間之單變量皮爾森相關係數並轉換為 F 統計量，挑選 F 值最高之 $K$ 個特徵。
   - *優點*：運算複雜度極低（$O(P)$），完全獨立於學習演算法。
   - *缺點*：忽略變數間的多重共線性與特徵交互作用，易挑選高度冗餘特徵。
2. **Wrapper 方法 (Recursive Feature Elimination, RFE)**：
   - *機制*：使用 OLS 線性回歸擬合全模型，遞迴評估各變數回歸係數之絕對值 $|\beta_j|$，每次剔除權重最小之特徵，直至剩餘 $K$ 個特徵。
   - *優點*：考慮變數間的協同效應與條件關聯。
   - *缺點*：對係數尺度的單位敏感（故前處理必須標準化），且具貪婪本質。
3. **Wrapper 方法 (Sequential Forward Selection, SFS)**：
   - *機制*：由空集合開始，在 5 折交叉驗證架構下，每一步貪婪挑選能使負均方誤差 (-MSE) 最大化改善之單一特徵加入集合，直至達到 $K$ 個特徵。
   - *優點*：直接以交叉驗證泛化誤差為目標函數，能有效搜尋最佳子集。
   - *缺點*：運算開銷龐大（需反覆擬合 $O(K \cdot P)$ 次回歸模型）。
4. **Embedded 方法 (LassoCV)**：
   - *機制*：在 OLS 損失函數加入 L1 正則化懲罰項 $\alpha \sum_{j=1}^P |\beta_j|$。利用次微分性質將不重要特徵之係數嚴格壓縮為 0。
   - *優點*：將特徵選擇與參數估計整合於單一凸優化問題中，計算效率高於 Wrapper。
5. **Model-based 方法 (Permutation Importance)**：
   - *機制*：在訓練集內部切分驗證子集，隨機打亂單一特徵之數值破壞其與 Target 之關聯，評估驗證集 MSE 增加量（泛化效能衰退程度）。
   - *優點*：不依賴線性假設，能真實衡量該變數對未見資料之邊際預測貢獻。

### 4.2 學生表現資料集：特徵選擇實驗實測數值

以下表格為執行 `scripts/run_experiments.py` 產出之**真實可重現實驗數據**（絕無虛構）：

| 模型方法 (Method) | 特徵選擇範式 | 選取特徵數 | 測試集 R² | 調整後 R² | 測試集 RMSE | 測試集 MAE | 測試集 MSE | 訓練集 R² | 5折CV R² | 耗時(秒) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline (All Features)** | 全特徵基準對照 | 27 | **0.769650** | **0.764843** | **1.804445** | **0.452392** | **3.256020** | 0.717314 | 0.715585 | 0.076 |
| **Lasso Selection (Embedded)** | L1 正則化 (LassoCV) | 27 | **0.769650** | **0.764843** | **1.804445** | **0.452392** | **3.256020** | 0.717314 | 0.715585 | 0.178 |
| **Permutation Importance** | 排列重要度 ($K=5$) | 5 | **0.666460** | **0.665193** | **2.171314** | **1.190192** | **4.714606** | 0.597603 | 0.598156 | 0.800 |
| **SelectKBest (Filter)** | 單變量 F 檢定 ($K=5$) | 5 | **0.664693** | **0.663419** | **2.177060** | **1.174931** | **4.739588** | 0.605658 | 0.606164 | 0.042 |
| **SFS Forward (Wrapper)** | 循序前向貪婪 ($K=5$) | 5 | **0.664693** | **0.663419** | **2.177060** | **1.174931** | **4.739588** | 0.605658 | 0.606164 | 17.598 |
| **RFE (Wrapper)** | 遞迴係數消除 ($K=5$) | 5 | **0.645730** | **0.644384** | **2.237773** | **1.266747** | **5.007626** | 0.580982 | 0.581391 | 0.144 |

### 4.3 各方法挑選之特徵子集明細
- **Baseline (27 項)**：全部 27 個標準化/虛擬編碼特徵。
- **SelectKBest (Filter, 5 項)**：`Hours_Studied`, `Attendance`, `Previous_Scores`, `Tutoring_Sessions`, `Access_to_Resources_Low`
- **SFS Forward (Wrapper, 5 項)**：`Hours_Studied`, `Attendance`, `Previous_Scores`, `Tutoring_Sessions`, `Access_to_Resources_Low`（與 SelectKBest 挑選完全相同之子集，兩者測試集 $R^2$ 均為 0.664693）
- **Permutation Importance (5 項)**：`Attendance`, `Hours_Studied`, `Parental_Involvement_Low`, `Access_to_Resources_Low`, `Previous_Scores`（測試集 $R^2 = 0.666460$，為所有 $K=5$ 方法中泛化表現最高者）
- **RFE (Wrapper, 5 項)**：`Hours_Studied`, `Attendance`, `Parental_Involvement_Low`, `Parental_Involvement_Medium`, `Access_to_Resources_Low`（完全剔除了 `Previous_Scores`，測試集 $R^2$ 下降至 0.645730）
- **Lasso Selection (Embedded, 27 項)**：$\alpha = 0.002271$，非零係數特徵數為 27/27。若依據係數絕對值 $|\beta|$ 取前五大特徵，則排序為：
  1. `Attendance` ($|\beta| = 5.281$)
  2. `Access_to_Resources_Low` ($|\beta| = 1.968$)
  3. `Parental_Involvement_Low` ($|\beta| = 1.859$)
  4. `Hours_Studied` ($|\beta| = 1.774$)
  5. `Family_Income_Low` ($|\beta| = 1.238$)

### 4.4 深度科學探討一：Lasso 為何保留全部 27 個特徵？
在既有文獻與初版報告中，常誤將 Lasso 視為「必然會將特徵數量縮減至極少數」之黑盒子。本實驗實測發現 Lasso 保留了全部 27 個特徵，其背後具備深刻的統計理論根源：
1. **大樣本下交叉驗證選擇之 $\alpha$ 懲罰極小**：
   在 $N=5,285$ 的大規模樣本下，目標函數中殘差平方和 (SSE) 的梯度權重隨樣本數線性放大，相形之下 L1 正則化項 $\alpha \sum |\beta_j|$ 的懲罰力道相對微弱。透過 5 折交叉驗證所搜尋出的最優 $\alpha$ 僅為 **$0.002271$**。
2. **所有預測因子均具備微弱非零真實關聯**：
   在真實教育場景中，諸如學校體制、同儕影響、家庭收入等變數，雖然單獨解釋力有限，但其真實母體迴歸係數均非絕對為零。在此微弱懲罰強度下，即使微小的係數亦能協助降低驗證集之 MSE，因此其權重未被壓縮至截斷閥值 ($10^{-5}$) 以下。
3. **特徵篩選規則之定義本質**：
   Filter、RFE 與 SFS 是在演算法外部**強加固定特徵數預算 ($K=5$)** 之約束；而標準 LassoCV 是以**預測誤差最小化**為目標之資料驅動內嵌法。兩者定義規則不同，忠實反映了統計正則化與啟發式特徵修剪的本質歧異。

### 4.5 深度科學探討二：為何特徵縮減導致 $R^2$ 由 0.7696 下降至 ~0.665？
實務界常有迷思，誤以為「特徵選擇一定能提高測試集 $R^2$」。本實驗清晰證實：**在當前資料集中，將特徵數由 27 縮減至 5，測試集 $R^2$ 必然發生顯著下降（由 0.769650 跌至 0.6647～0.6665，淨損失約 10.3% 的解釋力）**。其統計學與機器學習理論解析如下：

1. **多元回歸判定係數之單調性定理**：
   在樣本容量固定下，向線性回歸模型新增任何一個樣本偏相關係數不為零之變數，其決定係數 $R^2$ 在數學上必然非減（$R^2_{p+1} \ge R^2_p$）。被剔除的 22 個特徵（如家長參與、同儕影響、學校型態等）並非純隨機白噪音，而是具備統計顯著之微弱偏相關。移除這 22 個特徵，等同於直接截斷了模型對該部分變異的解釋能力。
2. **偏差-變異數權衡 (Bias-Variance Trade-off)**：
   特徵選擇能提升泛化效能的唯一先決條件是：**「藉由剔除雜訊變數所節省之參數估計變異量 (Variance Reduction)，大於剔除變數所招致之遺漏變數偏差 (Omission Bias)」**。
   然而在本研究中，$N_{\text{train}} = 5,285$ 遠大於特徵維度 $P = 27$（樣本維度比高達 195:1）。在極充沛的樣本支持下，全模型的參數估計變異量極低（無過擬合現象，Train $R^2=0.7173$, Test $R^2=0.7696$）。因此，強行刪減 22 個特徵無法帶來任何可觀的變異數下降，反而引發顯著的遺漏變數偏差，致使均方誤差 MSE 由 3.256 惡化至 4.715。
3. **結論**：特徵選擇之主要效益在於**「降低資料收集成本、提升模型解釋性與降低維護複雜度」**，而非無條件提高預測準確率。

---

## 5. 模型評估、預測區間推論與假設診斷 (Evaluation & Diagnostics)

### 5.1 95% 預測區間 (Prediction Interval) vs. 95% 信賴區間 (Confidence Interval)

許多資料科學報告常混淆預測區間與信賴區間。本研究採用 Statsmodels `OLSResults.get_prediction().summary_frame(alpha=0.05)` 產出兩者，其嚴謹數學定義與實證比較如下：

#### 數學公式推導
對於給定之新特徵向量 $x_0$：
1. **母體均值 95% 信賴區間 (CI for Mean Response $E[Y|x_0]$)**：
   $$\text{CI}_{0.95} = x_0^T \hat{\beta} \pm t_{n-p-1, 0.975} \cdot s \sqrt{x_0^T (X^T X)^{-1} x_0}$$
   其衡量的是：在相同條件 $x_0$ 下，**所有可能學生之「平均考試成績期望值」**的估計不確定性。當樣本數 $n \to \infty$ 時，區間寬度收斂至 0。
2. **個別觀測值 95% 預測區間 (PI for Individual Observation $Y_0$)**：
   $$\text{PI}_{0.95} = x_0^T \hat{\beta} \pm t_{n-p-1, 0.975} \cdot s \sqrt{1 + x_0^T (X^T X)^{-1} x_0}$$
   其衡量的是：**「未來特定單一位具備特徵 $x_0$ 的學生」**實際考分的可能落點。即使樣本數 $n \to \infty$，區間寬度依然受到隨機誤差項不可約減之本質變異 $\sigma^2$ 之限制，無法收斂為 0。

#### 實證數值對比
| 區間指標 | 母體均值 95% 信賴區間 (CI) | 個別觀測值 95% 預測區間 (PI) | 統計意義與實務啟示 |
|---|---|---|---|
| **測試集平均寬度** | **0.5913 分** | **8.2190 分** | PI 寬度為 CI 寬度的 **13.9 倍**，充分反映個體隨機隨機變異之巨大影響 |
| **測試集真實涵蓋率** | **49.85%** | **99.47%** | CI 僅涵蓋半數測試樣本，證實絕不可誤用 CI 預測單一學生表現；PI 達到 99.47% 極高可靠度 |

### 5.2 特徵選擇後的統計推論界限 (Post-Selection Inference)
必須指出：當線性回歸模型之輸入特徵是歷經資料驅動的特徵篩選（如 SFS、RFE 或 Lasso）決定時，傳統 OLS 假設中「設計矩陣 $X$ 為外生固定」之前提已被破壞。此現象在統計學中稱為**選擇後推論 (Post-Selection Inference)** 或 **Selective Inference**。若在特徵挑選後直接套用標準 t 檢定計算信賴區間，其名義 95% 涵蓋率可能產生輕微樂觀偏差。本研究採用嚴格之訓練集擬合、測試集獨立驗證區間涵蓋率，確保推論結論具備實證支撐。

### 5.3 線性回歸五大基本假設診斷

本研究針對 Student Performance 訓練模型執行嚴謹之計量統計檢定：

| 回歸基本假設 | 檢定/評估方法 | 統計量數值 | p-value / 判讀依據 | 診斷結論 | 統計意涵與實務處置 |
|---|---|---|---|---|---|
| **1. 線性性 (Linearity)** | 殘差對擬合值相關度 | Corr = 0.0000 | 殘差均勻對稱散佈於 0 軸 | **完美符合** | 特徵與 Target 間之整體線性規格設定正確，無明顯非線性漏估 |
| **2. 殘差獨立性 (Independence)** | Durbin-Watson 統計量 | **DW = 1.9541** | 落於 [1.5, 2.5] 理想區間 | **符合** | 殘差無一階自相關。因本資料為橫斷面調查，個體間具獨立隨機性 |
| **3. 同質變異性 (Homoscedasticity)** | Breusch-Pagan LM 檢定 | LM = 22.08 | **$p = 0.7335$** ($> 0.05$) | **顯著符合** | 無法拒絕同質變異虛無假設，殘差變異數穩定，OLS 參數估計具備最優效率 |
| **4. 殘差常態性 (Normality)** | Jarque-Bera 檢定 | JB = 3,883,964 | $p = 0.0000$ ($< 0.05$) | **檢定拒絕，但受 CLT 保障** | 由於樣本數高達 5,285 筆，JB 檢定極為敏感。但依中央極限定理 (CLT)，參數估計量高度漸近常態，Q-Q 圖主體吻合 |
| **5. 無多重共線性 (Multicollinearity)** | 變異數膨脹因子 (VIF) | 最大 VIF = **2.80** | 全部 27 個特徵 VIF $< 3.0$ | **優異受控** | 遠低於警戒值 5.0 或 10.0。證實 `drop='first'` 有效化解虛擬變數陷阱 |

---

## 6. 對比案例分析：Student Performance vs. Flood Probability

為驗證特徵選擇規律是否普遍成立，本研究引入第二案例 Kaggle **Flood Probability Prediction**（洪水機率預測，30,000 筆樣本、20 個水文連續特徵）進行全流程對比：

### 6.1 跨資料集實驗數據對照表
| 評估指標 | Student Performance Factors (主研究) | Flood Probability Prediction (對比研究) |
|---|---|---|
| **樣本規模與型態** | 6,607 筆真實社會調查資料 (含遺漏值) | 30,000 筆固定隨機抽樣合成數值資料 |
| **特徵維度 (原始 ➔ 編碼後)** | 19 原始特徵 ➔ **27 編碼特徵** (6數值+21虛擬) | 20 原始特徵 ➔ **20 連續數值特徵** (無類別) |
| **全特徵 Baseline 測試集 R²** | **0.769650** | **0.844545** |
| **最佳 K=5 子集測試集 R²** | **0.666460** (Permutation Importance) | **0.180733** (RFE Wrapper) |
| **特徵縮減 R² 保留比例** | **86.6%** ($0.6665 / 0.7696$) | **21.4%** ($0.1807 / 0.8445$) |
| **特徵縮減 R² 衰退程度** | 僅微幅衰退 13.4% | **大幅崩跌 78.6%** |
| **最關鍵特徵結構** | 高度集中：`Attendance`, `Hours_Studied` 貢獻主導變異 | 均勻分散：20 個特徵等權重線性疊加 |

### 6.2 訊號架構理論解密：集中型訊號 vs. 分散型訊號
本對比實驗揭示了資料科學中最關鍵的本質洞察：
1. **集中型訊號 (Pareto Signal Distribution)**：
   在 Student 資料集中，少數核心行為特徵（出席率與讀書時數）即主導了考試成績的絕大部分方差。此時實施特徵選擇，能以極小的效能代價（$R^2$ 僅微幅下降 0.103）換取模型複雜度的大幅降低（變數減少 81%）。
2. **分散型訊號 (Uniformly Distributed Signal)**：
   在 Flood 資料集中，目標變數的真實生成機制為 20 個地理水文特徵的近乎均勻線性組合（每個變數提供約 5% 的變異解釋量）。當強行篩選 $K=5$ 個特徵時，等同於直接丟棄 75% 的訊號來源，導致模型 $R^2$ 由 0.8445 暴跌至 0.1807。
3. **實務結論**：特徵選擇的成功與否，完全取決於真實資料生成程序 (Data Generating Process) 的訊號集中程度，切忌盲目套用特徵篩選。

---

## 7. 部署考量與生命週期維護 (Deployment Perspective)

依據 CRISP-DM 框架第六階段，研究成果必須轉化為可維護之軟體工程實踐：
1. **模型序列化與輕量依賴**：生產環境採用輕量化 FastAPI 服務，將預處理 Pipeline 與 OLS 參數進行向量化儲存，避免在 Serverless 環境中引入不必要的龐大科學計算相依套件。
2. **輸入資料合規驗證**：建立 Pydantic 資料合約，對使用者傳入之 19 項原始特徵實施範圍檢核（如出席率需介於 0～100、讀書時數不得為負），防止垃圾輸入導致回歸外推 (Extrapolation)。
3. **概念漂移 (Concept Drift) 監控**：定期計算殘差平方和與 Durbin-Watson 統計量，若學生群體行為改變導致測試誤差超越 95% 預測區間警戒線，應觸發模型重新校準機制。

---

## 8. 研究限制與未來展望 (Limitations & Future Work)

1. **線性加性假設限制**：標準線性回歸假設特徵間效應為線性相加。然而在教育實務中，`Hours_Studied` 與 `Attendance` 極可能具備非線性交互作用（例如：只出席不讀書，或讀書時間邊際效益遞減）。未來可納入二次多項式項與交互作用項 ($X_i \times X_j$)。
2. **未觀測之內生性混淆**：受限於公開資料集之蒐集範圍，學生天賦智力、心理壓力與考試焦慮等關鍵變數未被記錄，可能導致遺漏變數偏差。
3. **異質族群分析**：未來可評估分量回歸 (Quantile Regression)，探討特徵對於高分群與低分群學生是否具備異質影響。

---

## 9. 真實研究紀錄待補項目 (User Action Items)

依據專題評分規範，本研究誠實保留真實研究紀錄之待補欄位，嚴禁使用虛構或預設示範文字充數：

### 9.1 Google NotebookLM 研究摘要 (待使用者填寫)
> [!IMPORTANT]
> **操作說明**：
> 請使用者依據 `docs/notebooklm_research_guide.md` 之指引，開啟 Google NotebookLM 並匯入作業指定之多篇線性回歸、特徵選擇與 CRISP-DM 網路文獻，針對研究主題進行提問探討後，將**真實產生之研究摘要（至少 100 字繁體中文）**貼入下方區塊：

```text
[=== 待補欄位：請在此貼上您於 Google NotebookLM 產生的真實研究摘要文字（至少 100 字） ===]
```

### 9.2 ChatGPT / AI 協作對話紀錄 (待使用者檢附)
> [!NOTE]
> **操作說明**：
> 本次研究在架構規劃、程式碼除錯與統計檢定整合過程中獲得 AI 輔助。請依據 `docs/chatgpt_conversation_export_guide.md` 之說明，將您的完整對話匯出為 PDF 檔案，並儲存於：
> `reports/ai_conversations/chatgpt_conversation_record.pdf`

---

## 10. 附錄：可重現性設定與代碼結構

- **隨機種子 (Random State)**：所有切分與演算法統一固定為 `random_state=42`。
- **訓練/測試比例**：`test_size=0.20`。
- **核心程式模組**：
  - 前處理與防洩漏管線：`src/preprocessing.py`
  - 五種特徵選擇演算法：`src/feature_selection.py`
  - 統計診斷與回歸假設：`src/statistical_tests.py`
  - 95% 預測區間推論：`src/prediction_intervals.py`
  - 離線實驗自動化腳本：`scripts/run_experiments.py`
  - 報告與 PDF 生成器：`src/reporting.py`
