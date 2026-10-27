# Multiple Linear Regression Dashboard
### Interactive Regression Analysis with CRISP-DM and Feature Selection

[![Python Version](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **GitHub Repository**：[https://github.com/sam052003/Multiple-Linear-regression.git](https://github.com/sam052003/Multiple-Linear-regression.git)  
> **核心設計文件 (Single Source of Truth)**：請參閱 [MyPlan/design.md](MyPlan/design.md)  
> **雲端部署指南**：請參閱 [docs/deployment.md](docs/deployment.md)  

---

## 📌 專案總覽 (Project Overview)

本專案為大學「多元線性回歸 (Multiple Linear Regression, MLR)」課程作業，建立了一個遵循 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 完整生命週期的現代化互動式機器學習與統計分析 Web 應用程式。

本系統不只是展示回歸模型的單一預測分數，而是引導使用者從問題理解、探索性資料分析 (EDA)、嚴格防資料洩漏的前處理 Pipeline、**5 種特徵選擇演算法深度比較**、Scikit-learn 線性回歸訓練、Statsmodels OLS **95% 預測區間 (Prediction Interval)** 推論、**回歸五大統計假設診斷**，到單筆樣本互動式預測與綜合成果匯出。

![CRISP-DM Process](reports/figures/crisp_dm_process.png)

---

## 🌟 核心特色與加分功能 (Core & Bonus Features)

### 1. 三個內建回歸主題切換 (Bonus 1: Problem Selector)
網站提供側邊欄下拉選單，可無縫切換三種不同規模與領域的 Regression Problems：
- 🏢 **50 Startups — Profit Prediction**（**預設展示**）：經典入門教學案例，分析研發、行政、行銷支出與所在州別對淨利的影響。
- 🎓 **Student Performance Factors — Exam Score Prediction**（**主要研究**）：Kaggle 真實教育數據，約 6,607 筆資料、**19 個原始輸入特徵**（精確符合 14~39 個特徵作業規格），Target 為連續數值 `Exam_Score`。
- 🌊 **Flood Probability Prediction**（**第二研究**）：Kaggle Playground s4e5 合成資料集，20 個數值特徵，採用固定種子 (`seed=42`) 抽樣 30,000 筆快取，排除 `id` 欄位。

### 2. 自由自訂資料集支援 (Bonus 2: Custom Dataset)
完整支援使用者載入任意回歸資料集：
- 🌐 **安全 CSV URL 載入**：內建 SSRF 防護機制，強制 HTTPS、DNS 解析檢查、嚴格阻擋本機與內部私有 IP、檔案下載 20MB 上限與逾時保護。
- 💻 **本機 CSV 上傳**：支援 UTF-8、UTF-8 BOM 與 Latin1 編碼自動解析。
- 🎯 **自訂 Target 與特徵過濾**：使用者可自由指定 Target，自動辨識數值與類別特徵，排除 ID 欄位，並對非 14~39 規格給出友善警示。

### 3. 五大特徵選擇演算法深度比較 (Feature Selection Suite)
支援從全特徵 Baseline 到五大方法全面比較（固定在 Training Set 上擬合）：
1. **Baseline**：保留全部前處理後特徵。
2. **SelectKBest**：Filter 方法，以 `f_regression` F 檢定評分。
3. **Recursive Feature Elimination (RFE)**：Wrapper 方法，以 `LinearRegression` 遞迴剃除權重最小特徵。
4. **Sequential Feature Selector (SFS)**：Wrapper 方法，以貪婪前向搜尋 (`forward`) 尋找最佳 CV MSE 子集，支援大資料集 Fast Mode。
5. **Lasso Feature Selection**：Embedded 方法，使用 `LassoCV` 正則化進行特徵收縮，具備非零權重保底機制。
6. **Permutation Importance**：Model-based 方法，於內部驗證集打亂特徵值評估泛化性能衰退量。

### 4. 95% 預測區間推論 (95% Prediction Interval)
老師特別要求預測圖必須具備信賴區間或預測區間。本專案透過 **Statsmodels OLS** 完整實作：
- **95% Confidence Interval for Mean (母體均值信賴區間)**：評估給定條件下平均期望值的落點。
- **95% Prediction Interval for Observation (單一觀測值預測區間)**：評估新觀測值的落點，納入個體隨機誤差 $\sigma^2$，區間必定寬於 CI。
- 揭露特徵選擇後之 Post-selection inference 統計限制。

### 5. 多元線性回歸五大假設診斷 (Regression Diagnostics)
- **線性性 (Linearity)**：殘差 vs. 擬合值散佈圖與零軸正交檢定。
- **殘差獨立性 (Independence)**：Durbin-Watson 統計量（附帶橫斷面資料適用限制說明）。
- **同質變異性 (Homoscedasticity)**：Breusch-Pagan 檢定與 p-value 解讀。
- **殘差常態性 (Normality)**：Q-Q 圖、Jarque-Bera 檢定與大樣本敏感度說明。
- **多重共線性 (Multicollinearity)**：相關熱圖與變異數膨脹因子 (VIF) 計算表。

---

## 📊 真實實驗評估結果 (Empirical Results)

### 主研究：Student Performance Factors ($N=6,607$)
以 80/20 切分（訓練集 5,285 筆，測試集 1,322 筆），特徵選擇固定 $K=5$：

| 特徵選擇方法 (Method) | 特徵數 | 測試集 $R^2$ | Adjusted $R^2$ | 測試集 RMSE | 測試集 MAE | 訓練集 $R^2$ | CV 平均 $R^2$ |
|---|---|---|---|---|---|---|
| **Baseline (All Features)** | 27 | **0.7696** | 0.7648 | **1.8044** | 0.4524 | 0.7173 | 0.7156 |
| **Lasso Selection (Embedded)** | 27 | **0.7696** | 0.7648 | **1.8044** | 0.4524 | 0.7173 | 0.7156 |
| **Permutation Importance** | 5 | **0.6665** | 0.6652 | **2.1713** | 1.1902 | 0.5976 | 0.5982 |
| **SelectKBest (Filter)** | 5 | **0.6647** | 0.6634 | **2.1771** | 1.1749 | 0.6057 | 0.6062 |
| **SFS Forward (Wrapper)** | 5 | **0.6647** | 0.6634 | **2.1771** | 1.1749 | 0.6057 | 0.6062 |
| **RFE (Wrapper)** | 5 | **0.6457** | 0.6444 | **2.2378** | 1.2667 | 0.5810 | 0.5814 |

> **重要洞見**：在學生考試成績預測中，僅需 5 個核心特徵 (`Previous_Scores`, `Hours_Studied`, `Attendance`, `Tutoring_Sessions`, `Access_to_Resources_Low`)，模型即可保留全體 27 個特徵超過 **86% 的預測力** ($R^2 = 0.665$ vs $0.770$)。

![Feature Selection Comparison](reports/figures/feature_selection_comparison.png)

---

## 📂 專案架構 (Project Structure)

```text
HW2-Multiple Linear regression/
├── app.py                            # Streamlit 應用程式入口
├── requirements.txt                  # 相容依賴套件清單
├── README.md                         # 專案說明文件
├── .gitignore                        # Git 忽略清單
├── MyPlan/
│   └── design.md                     # 核心系統架構與設計文件 (Single Source of Truth)
├── src/                              # 核心模組
│   ├── config.py                     # 全域常數與資料集詮釋資料
│   ├── data_loader.py                # 資料讀取與快取 (@st.cache_data)
│   ├── data_validation.py            # SSRF 防護、格式驗證與資料品質報告
│   ├── preprocessing.py              # ColumnTransformer 管線 (防資料洩漏)
│   ├── feature_selection.py          # 5 大特徵選擇演算法與比較引擎
│   ├── modeling.py                   # 線性回歸與係數解析
│   ├── evaluation.py                 # MAE, RMSE, R2, Adjusted R2 與 CV
│   ├── statistical_tests.py          # 5 大回歸假設統計檢定與 VIF
│   ├── prediction_intervals.py       # Statsmodels OLS 95% PI / CI 推論
│   ├── visualization.py              # Plotly 互動式圖表生成
│   └── reporting.py                  # 報告與數據 CSV/PDF 產出
├── components/                       # Streamlit 各分頁 UI 元件
│   ├── sidebar.py                    # 側邊欄控制項
│   ├── overview.py                   # 商業理解與研究動機
│   ├── data_understanding.py         # EDA、分佈與資料品質
│   ├── data_preparation.py           # 前處理與切分展示
│   ├── feature_selection_ui.py       # 特徵選擇比較儀表板
│   ├── evaluation_ui.py              # 模型評估、95% PI 與假設檢定
│   ├── prediction_ui.py              # 互動式單筆特徵預測
│   └── report_ui.py                  # 成果匯出與報告下載
├── scripts/                          # 命令列與實驗腳本
│   ├── prepare_student_data.py       # 學生資料準備
│   ├── prepare_flood_data.py         # 洪水抽樣資料準備
│   ├── run_experiments.py            # 自動化實驗與報告產出
│   └── generate_notebooks.py         # 產生重現 Jupyter Notebooks
├── notebooks/                        # 可重現 Jupyter Notebooks
│   ├── student_performance_analysis.ipynb
│   └── flood_prediction_analysis.ipynb
├── reports/                          # 產出成果與報告
│   ├── final_report.md               # CRISP-DM 正式研究報告 (Markdown)
│   ├── final_report.pdf              # 期末報告 PDF
│   ├── notebooklm_summary.md         # NotebookLM 真實摘要填寫處
│   ├── figures/                      # 靜態論文成果圖表
│   ├── results/                      # 指標與預測 CSV
│   └── ai_conversations/             # ChatGPT 對話 PDF 存放處
├── docs/                             # 指南與方法論
│   ├── methodology.md                # 統計與機器學習理論手冊
│   ├── deployment.md                 # Streamlit Cloud 部署手冊
│   ├── notebooklm_research_guide.md  # NotebookLM 研究指南
│   └── chatgpt_conversation_export_guide.md # 對話 PDF 匯出指南
├── assets/
│   └── screenshots/                  # 系統操作說明與截圖目錄
└── tests/                            # 自動化測試套件 (pytest)
    ├── test_data_loader.py
    ├── test_custom_dataset.py
    ├── test_modeling.py
    ├── test_feature_selection.py
    └── test_prediction_intervals.py
```

---

## 💻 本機執行指南 (Local Execution)

### 1. 環境需求
- Python 3.11、3.12 或 3.14
- Git

### 2. 安裝步驟
```bash
# 複製儲存庫
git clone https://github.com/sam052003/Multiple-Linear-regression.git
cd Multiple-Linear-regression

# 安裝相依套件
pip install -r requirements.txt
```

### 3. 執行自動化測試 (20 項全數通過)
```bash
pytest -v
```

### 4. 啟動 Streamlit 互動儀表板
```bash
streamlit run app.py
```
啟動後在瀏覽器開啟 `http://localhost:8501` 即可進入儀表板。

### 5. 批次執行全套實驗與報告產出
```bash
python scripts/run_experiments.py
```

---

## 🌐 雲端部署 (Deployment)

本專案支援一鍵部署至 **Streamlit Community Cloud**：
1. 進入 [share.streamlit.io](https://share.streamlit.io/) 並登入 GitHub。
2. 選擇 Repository：`sam052003/Multiple-Linear-regression`。
3. Branch 設定為 `main`，Main file path 設定為 `app.py`。
4. 點選 **Deploy** 即可取得公開 Demo URL。
> 詳細步驟請參閱 [docs/deployment.md](docs/deployment.md)。

---

## 🤖 AI 輔助開發揭露 (AI Assistance Disclosure)

- **工具**：Google Antigravity (Gemini 3.8 Flash)
- **輔助範疇**：CRISP-DM 架構規劃、SSRF 網址安全防護設計、Statsmodels OLS 與 Scikit-learn 特徵矩陣對齊、自動化測試撰寫與報告模組整合。
- **對話 PDF 匯出**：請依據 [docs/chatgpt_conversation_export_guide.md](docs/chatgpt_conversation_export_guide.md) 將真實對話匯出並放置於 `reports/ai_conversations/`。
- **NotebookLM 研究**：請依據 [docs/notebooklm_research_guide.md](docs/notebooklm_research_guide.md) 將您的真實研讀心得填入 [reports/notebooklm_summary.md](reports/notebooklm_summary.md)。
