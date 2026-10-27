# ChatGPT / AI 對話紀錄匯出 PDF 操作指南

> **作業規格要求**：
> 老師要求所有 ChatGPT / AI 輔助開發的對話紀錄必須匯出成 PDF 繳交。
> 本文件說明如何將真實對話轉成 PDF，並放入專案中指定之 `reports/ai_conversations/` 目錄。

---

## 1. 匯出步驟說明

### 方法 A：透過瀏覽器內建「列印成 PDF」(最推薦、排版最完整)
1. 在瀏覽器中開啟您與 ChatGPT 或 AI 助理的完整開發對話頁面。
2. 展開所有被摺疊的程式碼區塊或長回應。
3. 按下快速鍵：
   - **Windows**：`Ctrl + P`
   - **Mac**：`Cmd + P`
4. 在列印對話框中：
   - **目的地 (Destination)**：選擇「另存為 PDF (Save as PDF)」。
   - **版面配置 (Layout)**：縱向 (Portrait)。
   - **背景圖形 (Background graphics)**：務必**勾選**，以保留語法突顯與色彩。
5. 點擊「儲存 (Save)」，檔案名稱建議命名為：
   - `chatgpt_hw2_development_session_1.pdf`
   - `chatgpt_hw2_development_session_2.pdf`
6. 將儲存之 PDF 移動至本專案目錄：
   `reports/ai_conversations/`

---

### 方法 B：使用 ChatGPT 官方「共用連結 (Share Link)」與匯出工具
1. 在 ChatGPT 對話視窗右上角點選「分享 (Share)」。
2. 複製產生的公開對話網址。
3. 在新分頁開啟該連結後，依上述「Ctrl + P」另存為 PDF。

---

## 2. 隱私與敏感資訊檢查清單 (Privacy Checklist)

在將對話 PDF 提交至公開 GitHub Repository 之前，請務必快速檢查確認：
- [ ] 無個人密碼、金鑰或敏感 Token (例如 Kaggle API Key、GitHub Personal Access Token)
- [ ] 無個人真實電話、住址或不願公開之隱私資訊
- [ ] 截圖與文字清晰無缺漏

---

## 3. 目標存放資料夾確認
- 存放路徑：`reports/ai_conversations/`
- 本專案已在 `.gitignore` 中妥善保留該目錄結構，您可安心放入真實對話 PDF 供期末評分查核。
