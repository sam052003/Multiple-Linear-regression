# Multiple Linear Regression Dashboard
### Interactive Regression Analysis with CRISP-DM and Feature Selection

[![Python Version](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2%20App%20Router-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Deployment](https://img.shields.io/badge/Vercel-Production%20Ready-black.svg)](https://vercel.com/)
[![Tests](https://img.shields.io/badge/pytest-39%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **GitHub Repository**：[https://github.com/sam052003/Multiple-Linear-regression.git](https://github.com/sam052003/Multiple-Linear-regression.git)  
> **核心設計文件 (Single Source of Truth)**：請參閱 [MyPlan/design.md](MyPlan/design.md)  
> **Vercel 雲端部署指南**：請參閱 [docs/deployment.md](docs/deployment.md)  

---

## 📌 專案成果架構 (Project Deliverables)

本專案成果明確分為兩大部分：
1. **互動式網頁分析平台（Interactive Web Dashboard）**：
   - 全站採用臺灣繁體中文（`zh-Hant-TW`）介面，保留重要英文機器學習術語。
   - 遵循 CRISP-DM 六大步驟之互動分析流程：問題理解、資料理解、資料準備、特徵選擇與建模、模型評估與診斷、互動預測與應用。
   - 支援 3 大內建回歸主題（50 Startups、Student Performance、Flood Probability）與自訂 CSV（HTTPS 網址／本機上傳）。
   - 提供 5 大特徵選擇演算法平行比較、高斯-馬可夫回歸五大假設統計診斷、Statsmodels 95% 預測區間推論與預測紀錄 CSV 匯出。
   - 部署於 Vercel 雲端無伺服器架構（Next.js 14 + FastAPI）。
2. **獨立 CRISP-DM 研究報告（Independent Research Report）**：
   - 正式學術報告為獨立作業繳交文件，完全與互動網站分離，不於公開網站中提供展示或下載入口，公開 API 亦不暴露報告內容。
   - 報告資產由離線腳本與 Notebook 獨立維護，完整保留於 Repository 根目錄下之 `reports/`、`notebooks/` 與 `scripts/` 中。

![CRISP-DM Process](reports/figures/crisp_dm_process.png)

---

## 🌟 網站核心功能 (Interactive Web Dashboard Features)

### 1. 臺灣繁體中文介面 (Traditional Chinese Localization)
- 網站所有導航、標題、說明文字、統計指標、假設檢定解說、圖表標籤與錯誤提示均採自然、專業的臺灣繁體中文。
- 同步保留重要機器學習英文術語（如：多元線性回歸 Multiple Linear Regression、特徵選擇 Feature Selection、資料探索分析 EDA、平均絕對誤差 MAE、均方根誤差 RMSE、決定係數 R²、95% 預測區間 Prediction Interval、95% 信賴區間 Confidence Interval 等）。

### 2. 三個內建回歸主題切換 (Problem Selector)
網站頂端下拉選單提供三個經典與進階 Regression Problems：
- 🏢 **50 Startups — 新創公司利潤預測**（**預設教學展示案例**）：經典入門案例，分析研發、行政、行銷支出與所在州別對企業淨利之影響。
- 🎓 **Student Performance — 學生成績預測**（**主要研究**）：Kaggle 教育數據，約 6,607 筆資料、**19 個原始輸入特徵**（精確符合 14~39 個特徵作業規格），Target 為連續數值 `Exam_Score`。
- 🌊 **Flood Probability — 洪水機率預測**（**比較研究**）：Kaggle 高維度環境觀測合成資料集，20 個數值特徵，採用固定種子抽樣 30,000 筆快取，嚴格排除 `id` 欄位。

### 3. 自由自訂資料集支援 (Custom Dataset)
完整支援使用者載入任意回歸資料集：
- 🌐 **安全 CSV URL 載入**：內建逐跳 SSRF 防護機制，強制 HTTPS、DNS 解析檢查、嚴格阻擋本機與內部私有 IP、檔案下載 4.5MB 限制與逾時保護。
- 💻 **本機 CSV 上傳**：支援直接上傳本機 CSV 並自動進行資料型態判斷與前處理。
- 🎯 **自訂 Target 與特徵過濾**：使用者可自由指定 Target，自動辨識數值與類別特徵，排除 ID 欄位，並對 14~39 特徵規格給出動態合規檢驗。

### 4. 五大特徵選擇演算法深度比較 (Feature Selection Suite)
支援從全特徵 Baseline 到五大方法全面平行比較（固定在 Training Set 上擬合）：
1. **Baseline**：保留全部前處理後特徵。
2. **SelectKBest**：Filter 方法，以 `f_regression` F 檢定評分。
3. **Recursive Feature Elimination (RFE)**：Wrapper 方法，以 `LinearRegression` 遞迴剃除權重最小特徵。
4. **Sequential Feature Selector (SFS)**：Wrapper 方法，以貪婪前向搜尋 (`forward`) 尋找最佳 CV MSE 子集，支援大資料集 Fast Mode。
5. **Lasso Feature Selection**：Embedded 方法，使用 `LassoCV` 正則化進行特徵收縮，具備非零權重保底機制。
6. **Permutation Importance**：Model-based 方法，於內部驗證集打亂特徵值評估泛化性能衰退量。

### 5. 95% 預測區間推論 (95% Prediction Interval)
依作業要求完整呈現預測不確定性，透過 **Statsmodels OLS** 實作：
- **95% Confidence Interval for Mean (母體均值信賴區間)**：評估給定條件下平均期望值的落點 $E(y|X)$。
- **95% Prediction Interval for Observation (單一觀測值預測區間)**：評估新觀測值的落點，納入個體隨機誤差 $\sigma^2_\epsilon$，區間必定寬於 CI。
- 揭露特徵選擇後之 Post-selection inference 統計限制與警示。

### 6. 多元線性回歸五大假設診斷 (Regression Diagnostics)
- **線性性 (Linearity)**：殘差 vs. 擬合值散佈圖與零軸正交檢定。
- **殘差獨立性 (Independence)**：Durbin-Watson 統計量（附帶橫斷面資料適用限制說明）。
- **同質變異性 (Homoscedasticity)**：Breusch-Pagan 檢定與 p-value 解讀。
- **殘差常態性 (Normality)**：Q-Q 圖、Jarque-Bera 檢定與大樣本敏感度說明。
- **多重共線性 (Multicollinearity)**：相關熱圖與變異數膨脹因子 (VIF) 計算表。

---

## 📑 獨立研究報告資產 (Independent Research Report Assets)

正式學術研究報告與分析實驗成果作為 Repository 之獨立產出（不於網頁上展示或下載）：
- 📄 **論文與完整報告**：`reports/final_report.md`、`reports/final_report.pdf`
- 🧠 **文獻研讀心得**：`reports/notebooklm_summary.md`
- 💬 **AI 輔助對話紀錄**：`reports/ai_conversations/`
- 📈 **實驗圖表產出**：`reports/figures/`
- 📊 **數值實驗紀錄**：`reports/results/` (`student_predictions_intervals.csv`、`student_feature_selection_comparison.csv`)
- 🧪 **離線執行腳本**：`scripts/run_experiments.py`、`notebooks/`

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
├── app/                              # Next.js 14 App Router (Frontend, 臺灣繁體中文)
│   ├── layout.tsx                    # 根佈局 (zh-Hant-TW) 與深色主題設定
│   ├── page.tsx                      # 儀表板主入口與 CRISP-DM 六大階段管理器
│   ├── globals.css                   # Tailwind 指令與玻璃擬態樣式
│   ├── types/api.ts                  # 前後端共用 TypeScript 型別
│   ├── lib/api.ts                    # REST API 客戶端
│   └── components/                   # React 現代化元件
│       ├── Navbar.tsx                # 導航列、Problem Selector 與連線狀態
│       ├── CustomDatasetModal.tsx    # 自訂資料集載入視窗 (SSRF 防護)
│       ├── PlotlyChart.tsx           # Plotly.js 客戶端動態圖表
│       └── tabs/                     # CRISP-DM 六大階段分頁
│           ├── OverviewTab.tsx       # Stage 1: 問題理解與資料集摘要
│           ├── DataUnderstandingTab.tsx # Stage 2: 資料理解 EDA 與相關分析
│           ├── DataPreparationTab.tsx   # Stage 3: 資料準備管線與基準模型
│           ├── FeatureSelectionTab.tsx  # Stage 4: 特徵選擇與建模評測
│           ├── EvaluationTab.tsx     # Stage 5: 模型評估與統計假設診斷
│           └── PredictionTab.tsx     # Stage 6: 互動預測與單筆推論
├── api/                              # FastAPI Backend (Vercel Serverless)
│   ├── index.py                      # ASGI 應用進入點 (報告路由已取消掛載)
│   ├── schemas.py                    # Pydantic v2 Request/Response Contracts
│   ├── requirements.txt              # Vercel Serverless 專屬相依套件
│   ├── services/ml_service.py        # 無狀態機器學習服務層
│   └── routers/                      # REST API 路由
│       ├── datasets.py               # 資料集概覽與自訂資料解析
│       ├── analysis.py               # 模型訓練、特徵選擇與診斷
│       └── predict.py                # 單筆樣本 95% PI 推論
├── reports/                          # [獨立成果] 正式研究報告與實驗數據 (不公開於網站)
│   ├── final_report.md               # 論文形式正式報告 Markdown
│   ├── final_report.pdf              # 論文形式正式報告 PDF
│   ├── notebooklm_summary.md         # NotebookLM 文獻研讀心得
│   ├── figures/                      # 靜態論文圖表
│   ├── results/                      # 實驗預測與特徵比較 CSV
│   └── ai_conversations/             # AI 輔助開發真實紀錄
├── src/                              # Python Machine Learning 核心演算法 (100% 沿用)
│   ├── config.py                     # 全域常數與資料集詮釋資料
│   ├── data_loader.py                # 資料讀取與記憶體快取 (@lru_cache)
│   ├── data_validation.py            # SSRF 防護、格式驗證與資料品質報告
│   ├── preprocessing.py              # ColumnTransformer 管線 (防資料洩漏)
│   ├── feature_selection.py          # 5 大特徵選擇演算法與比較引擎
│   ├── modeling.py                   # 線性回歸與係數解析
│   ├── evaluation.py                 # MAE, RMSE, R2, Adjusted R2 與 CV
│   ├── statistical_tests.py          # 5 大回歸假設統計檢定與 VIF
│   ├── prediction_intervals.py       # Statsmodels OLS 95% PI / CI 推論
│   ├── visualization.py              # 靜態論文圖表生成模組
│   └── reporting.py                  # [離線用] 報告與數據 CSV/PDF 產出
├── vercel.json                       # Vercel Monorepo 路由設定
├── package.json                      # Next.js 前端相依套件與腳本
├── requirements.txt                  # Python 完整相依套件清單
├── tests/                            # 自動化測試套件 (pytest, 39 項全通)
│   ├── test_api.py                   # FastAPI REST API 整合測試 (含報告端點 404 檢驗)
│   ├── test_data_loader.py           # 資料載入測試
│   ├── test_custom_dataset.py        # SSRF 防護與資料檢核測試
│   ├── test_modeling.py              # 線性回歸訓練與指標測試
│   ├── test_feature_selection.py     # 5 大特徵選擇演算法測試
│   └── test_prediction_intervals.py  # 95% PI 與假設檢定測試
└── app.py                            # [Legacy] 原 Streamlit 進入點 (保留)
```

---

## 💻 本機執行指南 (Local Execution)

### 1. 環境需求
- Python 3.11+ / 3.12 / 3.14
- Node.js 18+ (推薦 v20 LTS)
- Git

### 2. 安裝步驟
```bash
# 複製儲存庫
git clone https://github.com/sam052003/Multiple-Linear-regression.git
cd Multiple-Linear-regression

# 安裝 Python 相依套件
pip install -r requirements.txt

# 安裝前端 Node.js 相依套件
npm install
```

### 3. 執行自動化測試 (39 項全數通過)
```bash
pytest -v
```

### 4. 啟動開發伺服器

#### 終端機 1：啟動 FastAPI 後端 (Port 8000)
```powershell
python -m uvicorn api.index:app --reload --port 8000
```
- API 文件：`http://127.0.0.1:8000/api/docs`

#### 終端機 2：啟動 Next.js 前端 (Port 3000)
```powershell
npm run dev
```
- 瀏覽器開啟：`http://localhost:3000`

---

## 🌐 Vercel 雲端部署 (Vercel Deployment)

本專案支援一鍵部署至 **Vercel**：
1. 進入 [Vercel Dashboard](https://vercel.com/dashboard) 並連結 GitHub。
2. 匯入 Repository：`sam052003/Multiple-Linear-regression`。
3. Framework Preset 保持為 `Next.js`，Root Directory 保持為 `./`。
4. 點選 **Deploy**，Vercel 將自動編譯 Next.js 前端並以 Serverless Function 架設 `api/index.py`。
> 詳細步驟與資源限制說明請參閱 [docs/deployment.md](docs/deployment.md)。

---

## 🤖 AI 輔助開發揭露 (AI Assistance Disclosure)

- **工具**：Google Antigravity (Gemini 3.8 Flash)
- **輔助範疇**：CRISP-DM 架構規劃、Next.js 14 + FastAPI 遷移、SSRF 網址安全防護設計、Statsmodels OLS 與 Scikit-learn 特徵矩陣對齊、自動化測試撰寫與報告模組整合。
- **對話 PDF 匯出**：請依據 [docs/chatgpt_conversation_export_guide.md](docs/chatgpt_conversation_export_guide.md) 將真實對話匯出並放置於 `reports/ai_conversations/`。
- **NotebookLM 研究**：請依據 [docs/notebooklm_research_guide.md](docs/notebooklm_research_guide.md) 將您的真實研讀心得填入 [reports/notebooklm_summary.md](reports/notebooklm_summary.md)。
