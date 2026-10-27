# Streamlit Community Cloud 部署完整指南 (Deployment Guide)

> **專案儲存庫**：[https://github.com/sam052003/Multiple-Linear-regression.git](https://github.com/sam052003/Multiple-Linear-regression.git)  
> **進入點檔案**：`app.py`  
> **預設分支**：`main`  

---

## 1. 部署前準備檢查 (Pre-flight Checklist)
本專案已完全針對 Streamlit Community Cloud 進行輕量化與穩定性最佳化：
- [x] **進入點確認**：專案根目錄之 `app.py` 具備完整 CRISP-DM 介面。
- [x] **相依套件庫**：`requirements.txt` 已鎖定相容無原生衝突之 Python 輪子版本。
- [x] **資料自給自足**：已包含處理後之資料集快取 (`data/processed/`)，無需使用者設定本機路徑或外部金鑰即可完整執行。
- [x] **安全防護**：Custom Dataset 模組已內建嚴格之 SSRF 防護與檔案下載限制。
- [x] **無私密金鑰洩漏**：`.gitignore` 已嚴格排除所有暫存與敏感設定。

---

## 2. Streamlit Community Cloud 逐步部署步驟

### 步驟 1：登入 Streamlit Cloud
1. 開啟瀏覽器訪問 [Streamlit Community Cloud 官方入口](https://share.streamlit.io/)。
2. 點選「Continue with GitHub」並使用您的 GitHub 帳號登入授權。

### 步驟 2：建立新應用程式 (New App)
1. 進入 Streamlit 管理儀表板，點選右上角藍色按鈕 **「Create app」**。
2. 選擇 **「Yup, I have an existing app」**。

### 步驟 3：設定專案資訊
在彈出的部署設定視窗中依序填入：
- **Repository**：`sam052003/Multiple-Linear-regression`
- **Branch**：`main`
- **Main file path**：`app.py`
- **App URL** (可選)：自訂您偏好的次網域（例如 `mlr-dashboard-crispdm.streamlit.app`）。

### 步驟 4：進階設定 (Advanced Settings)
- **Python Version**：選擇 `3.11` 或 `3.12`。
- **Secrets**：本系統預設不需任何外部 API 金鑰即可執行；若未來需要額外設定，可於此處以 TOML 格式設定。

### 步驟 5：點擊部署 (Deploy)
1. 點擊 **「Deploy!」**。
2. Streamlit 將會自動執行：
   - 抓取 GitHub 最新程式碼
   - 解析 `requirements.txt` 並自動安裝所有相依套件
   - 啟動 `app.py`
3. 部署完成後，您即可取得正式公開網址 (Public Live Demo URL)。

---

## 3. 常見疑難排解 (Troubleshooting)

### Q1: 雲端部署出現記憶體不足 (OOM - Out of Memory)？
- **原因**：大型 Flood 資料集原始資料超過 100 萬筆，若未經抽樣直接載入會瞬間吃光雲端 1GB RAM 配額。
- **解決對策**：本系統已預設採用固定隨機種子 (`random_state=42`) 抽樣之代表性子集 (`flood_sample_30k.csv`)，既確保回歸統計顯著性，又能保證記憶體佔用小於 200MB。

### Q2: 自訂 CSV URL 下載逾時或報錯？
- **原因**：部分公網伺服器頻寬不足或目標 URL 涉及私有網段 (SSRF)。
- **解決對策**：系統已內建 12 秒連線逾時保護，並自動阻擋私有 IP。請確認輸入的是直接返回 `text/csv` 之公開 HTTPS 網址。
