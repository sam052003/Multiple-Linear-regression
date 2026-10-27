# Vercel & Cloud 部署完整指南 (Deployment Guide)

> **專案儲存庫**：[https://github.com/sam052003/Multiple-Linear-regression.git](https://github.com/sam052003/Multiple-Linear-regression.git)  
> **目標架構**：Next.js 14 App Router (Frontend) + FastAPI (Python Serverless API)  
> **部署平台**：Vercel (Monorepo 一鍵部署)  
> **分支**：`main` (正式生產分支) / `migration/vercel` (遷移驗證分支)

---

## 1. Vercel 部署架構與特點

本專案採用現代化解耦架構，將 **Next.js 14 App Router** 與 **FastAPI Serverless Functions** 整合於同一個 GitHub Repository 中，完全符合 Vercel 官方的最佳實踐：

- **Frontend (Next.js 14)**：TypeScript, React, Tailwind CSS, Plotly.js, Lucide Icons。提供零全頁重載、響應式排版與互動式資料視覺化。
- **Backend (FastAPI in `api/index.py`)**：100% 沿用 `src/` 中的 Python 機器學習核心（Scikit-learn, Statsmodels, Pandas, NumPy, SciPy）。
- **Routing & Proxy**：透過 `vercel.json` 與 `next.config.js`，將所有 `/api/*` 流量自動導向 `api/index.py` Serverless Function。

---

## 2. Vercel 一鍵部署步驟 (Step-by-Step)

### 步驟 1：匯入 GitHub Repository
1. 開啟瀏覽器造訪 [Vercel Dashboard](https://vercel.com/dashboard)。
2. 點擊 **「Add New...」** -> **「Project」**。
3. 在 GitHub 儲存庫清單中搜尋並選擇 `sam052003/Multiple-Linear-regression`，點擊 **「Import」**。

### 步驟 2：確認專案設定 (Project Settings)
Vercel 會自動偵測到 Next.js 專案：
- **Framework Preset**：`Next.js`
- **Root Directory**：`./` (專案根目錄)
- **Build Command**：`npm run build` (預設)
- **Output Directory**：`.next` (預設)
- **Install Command**：`npm install` (預設)

### 步驟 3：Serverless Python Function 自動偵測
Vercel 部署引擎會自動辨識根目錄下的 `api/index.py` 與 `api/requirements.txt`：
- 自動以 Python 3.11/3.12 容器建置 Serverless Function。
- 記憶體預設配置 1024 MB。
- 無需手動設定複雜的 Dockerfile。

### 步驟 4：點擊部署 (Deploy)
1. 點擊 **「Deploy」** 按鈕。
2. Vercel 將在約 1~2 分鐘內完成前端編譯、Python 相依套件安裝與 Edge 網路分發。
3. 部署完成後，即可獲得正式公開 Production 網址（例如 `https://multiple-linear-regression-dashboard.vercel.app`）。

---

## 3. Vercel 效能與資源限制對策 (Resource Constraints & Mitigation)

| 限制項目 | Vercel 規格限制 | 本專案因應對策 |
|---|---|---|
| **Function 執行時限 (Timeout)** | Hobby 方案上限 10~15 秒 | 循序特徵選擇 (SFS) 提供 Fast Mode（子採樣挑選特徵）；Flood 資料集預先抽取 30,000 筆固定種子樣本，全運算控制在 3 秒內。 |
| **Request Body 大小限制** | 最大 4.5 MB | Custom CSV 上傳在前端與後端均設置 4.5 MB 上限檢查；若需要超大型資料，支援輸入公開 HTTPS 網址由後端串流解析。 |
| **Serverless Bundle Size** | 解壓後上限 250 MB | `api/requirements.txt` 僅安裝純後端運算庫 (numpy, scikit-learn, statsmodels, fastapi)，排除 Streamlit 等重型套件。 |
| **Cold Start 延遲** | Python Serverless 初次喚醒約 1~2 秒 | API 採輕量級按需載入，核心資料集使用 `@functools.lru_cache` 保持暖機狀態。 |
| **SSRF 安全防護** | 防止利用 Serverless 探測內網 | `src/data_validation.py` 與 `api/routers/datasets.py` 嚴格解析 DNS IP，全面封鎖 Loopback、RFC 1918 私有網段與重定向。 |

---

## 4. 本機開發與雙向代理 (Local Development)

若要在本機同時啟動 FastAPI 後端與 Next.js 前端開發伺服器：

### 終端機 1：啟動 FastAPI 後端 (Port 8000)
```powershell
python -m uvicorn api.index:app --reload --port 8000
```
- API 交互式文件：`http://127.0.0.1:8000/api/docs`
- 系統健康檢查：`http://127.0.0.1:8000/api/health`

### 終端機 2：啟動 Next.js 前端 (Port 3000)
```powershell
npm run dev
```
- 前端儀表板：`http://localhost:3000`
- `next.config.js` 會自動將 `/api/*` 代理導向 `http://127.0.0.1:8000/api/*`。

---

## 5. [Legacy] Streamlit Community Cloud 部署指引 (歷史保留)

若仍需要以原本的 Streamlit 架構進行備用展示：
1. 進入點：`app.py`
2. 啟動指令：`streamlit run app.py`
3. 部署平台：[Streamlit Community Cloud](https://share.streamlit.io/)
4. 相依套件庫：`requirements.txt`
