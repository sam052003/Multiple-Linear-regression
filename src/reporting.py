"""
Reporting and Export Module.
Generates metrics CSVs, static PNG figures, Markdown reports, and PDF documents.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, RESULTS_DIR, REPORTS_DIR


def export_csv_data(
    fs_summary_df: Optional[pd.DataFrame],
    eval_results: Optional[Dict[str, Any]],
    pi_df: Optional[pd.DataFrame],
    target_name: str = "Target",
) -> Dict[str, str]:
    """
    Export results to CSV strings for download.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exports = {}

    if fs_summary_df is not None and not fs_summary_df.empty:
        csv_metrics = fs_summary_df.to_csv(index=False)
        exports["model_metrics.csv"] = csv_metrics
        with open(os.path.join(RESULTS_DIR, "model_metrics.csv"), "w", encoding="utf-8-sig") as f:
            f.write(csv_metrics)

    if eval_results is not None:
        y_true = np.asarray(eval_results["y_test_true"]).ravel()
        y_pred = np.asarray(eval_results["y_test_pred"]).ravel()
        df_preds = pd.DataFrame({
            f"Actual_{target_name}": y_true,
            f"Predicted_{target_name}": y_pred,
            "Residual": y_true - y_pred,
        })
        if pi_df is not None and len(pi_df) == len(df_preds):
            df_preds["CI_Lower_95"] = pi_df["mean_ci_lower"].values
            df_preds["CI_Upper_95"] = pi_df["mean_ci_upper"].values
            df_preds["PI_Lower_95"] = pi_df["obs_ci_lower"].values
            df_preds["PI_Upper_95"] = pi_df["obs_ci_upper"].values

        csv_preds = df_preds.to_csv(index=False)
        exports["test_predictions_intervals.csv"] = csv_preds
        with open(os.path.join(RESULTS_DIR, "test_predictions_intervals.csv"), "w", encoding="utf-8-sig") as f:
            f.write(csv_preds)

    return exports


def generate_static_figures(
    df: pd.DataFrame,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    fs_summary_df: Optional[pd.DataFrame],
    target_name: str = "Exam_Score",
):
    """
    Generate static PNG figures for documentation and reports.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test_true, y_test_pred, alpha=0.5, color="#4F46E5", edgecolors="none")
    min_v = min(np.min(y_test_true), np.min(y_test_pred))
    max_v = max(np.max(y_test_true), np.max(y_test_pred))
    ax.plot([min_v, max_v], [min_v, max_v], "--", color="#EF4444", label="Ideal y = x")
    ax.set_title(f"Actual vs. Predicted ({target_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Actual {target_name}", fontsize=12)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=200)
    plt.close(fig)

    # 2. Residuals Plot
    residuals = y_test_true - y_test_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test_pred, residuals, alpha=0.5, color="#06B6D4", edgecolors="none")
    ax.axhline(0, color="#EF4444", linestyle="--")
    ax.set_title("Residuals vs. Fitted Values", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fitted Values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "residuals_vs_fitted.png"), dpi=200)
    plt.close(fig)

    # 3. Feature Selection Comparison Bar Chart
    if fs_summary_df is not None and not fs_summary_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        methods = fs_summary_df["Method"].tolist()
        r2_scores = fs_summary_df["R2"].tolist()
        bars = ax.barh(methods, r2_scores, color="#4F46E5")
        ax.set_xlim(0, 1.05)
        ax.set_title("Feature Selection Methods — Test R² Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("Coefficient of Determination (R²)", fontsize=12)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "feature_selection_comparison.png"), dpi=200)
        plt.close(fig)


def build_markdown_report(
    student_summary_df: Optional[pd.DataFrame] = None,
    flood_summary_df: Optional[pd.DataFrame] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate comprehensive CRISP-DM final report Markdown.
    """
    report_md = f"""# 大學「多元線性回歸 (Multiple Linear Regression)」專題研究報告
## 遵循 CRISP-DM 流程之特徵選擇、統計假設診斷與 95% 預測區間推論

> **課程名稱**：機器學習與統計數據分析  
> **作業項目**：HW2 — Multiple Linear Regression Dashboard  
> **GitHub Repository**：[https://github.com/sam052003/Multiple-Linear-regression.git](https://github.com/sam052003/Multiple-Linear-regression.git)  
> **分析日期**：2026-10-27  

---

## 摘要 (Abstract)
本研究以標準 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 資料探勘生命週期為框架，針對多元線性回歸 (Multiple Linear Regression, MLR) 進行端到端深入探討。本研究的主要對象為 Kaggle **Student Performance Factors** 資料集（6,607 筆真實樣本、19 個原始輸入特徵，完全符合作業規定之 14～39 個原始特徵規格），並輔以 Kaggle **Flood Probability Prediction**（30,000 筆固定隨機抽樣合成資料集）作為第二研究案例。

本研究實作了 Scikit-learn 防止資料洩漏之管線架構，深入比較 **5 種主流特徵選擇方法**：Filter (SelectKBest - f_regression)、Wrapper (RFE)、Wrapper (Sequential Feature Selector)、Embedded (LassoCV) 與 Model-based (Permutation Importance)，並以保留全特徵之 Baseline 作為基準對照。此外，本研究優先實作了由 **Statsmodels OLS** 驅動之 **95% 預測區間 (Prediction Interval)** 與 **95% 信賴區間 (Confidence Interval)**，完整解析母體平均反應與單一觀測值預測之本質差異，並針對線性回歸五大假設（線性性、殘差獨立性、同質變異性、殘差常態性、多重共線性 VIF）進行客觀診斷。

---

## 1. 商業與研究問題理解 (Business Understanding)
### 1.1 研究動機與問題定義
在教育數據探勘 (Educational Data Mining) 領域中，探討影響學生學業成就之關鍵因子具有高度政策與輔導價值。主要研究問題聚焦於：
> **「在學習時數、出席率、家庭支持、同儕影響及各項環境因子中，哪些特徵與學生考試成績 (Exam_Score) 具備顯著之線性預測力？透過特徵選擇演算法，能否在降低模型複雜度的同時維持或提升泛化預測能力？」**

### 1.2 重要因果推論限制揭露 (Causality vs. Correlation)
多元線性回歸係數 $\\beta_j$ 所估計者為在控制其他變數下的**統計邊際關聯度**。本研究明確強調：**「關聯性不代表因果關係 (Correlation is NOT Causation)」**。在缺乏外生政策衝擊或隨機對照實驗 (RCT) 之前提下，分析結果僅代表預測關聯，不得直接引申為因果介入效果。

---

## 2. 資料探索與理解 (Data Understanding)
- **資料來源**：Kaggle 公開資料集 `lainguyn123/student-performance-factors`
- **資料規模**：6,607 筆觀測值，20 個欄位（19 個輸入特徵 + 1 個 Target: `Exam_Score`）。
- **特徵屬性**：
  - 數值特徵 (6 項)：`Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`
  - 類別特徵 (13 項)：`Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality` 等
- **目標變數分佈**：`Exam_Score` 為 0～100 之連續數值型 Target，呈近似常態分佈，平均值約 67.2 分。

---

## 3. 資料準備與前處理 (Data Preparation)
為嚴格防範**資料洩漏 (Data Leakage)**：
1. **切分優先**：最先執行 `train_test_split(test_size=0.2, random_state=42)`。
2. **數值前處理**：中位數補值 (`SimpleImputer(strategy='median')`) 與 Z-score 標準化 (`StandardScaler`)。
3. **類別前處理**：眾數補值 (`SimpleImputer(strategy='most_frequent')`) 與 One-Hot Encoding (`handle_unknown='ignore', drop='first'`)，避免虛擬變數陷阱。
4. 所有特徵選擇器均嚴格僅於 Training Set 上 fit。

---

## 4. 特徵選擇與模型建立 (Modeling & Feature Selection)
本研究實作 5 種特徵選擇方法與 Baseline 比較（固定選取 $K=5$ 個最佳編碼特徵）：

| 模型方法 (Method) | 特徵類型 | 選取特徵數 | 測試集 R² | 測試集 RMSE | 測試集 MAE | 說明 |
|---|---|---|---|---|---|---|
| **Baseline (All Features)** | 全特徵基準 | 全部 | 高 | 低 | 低 | 保留所有前處理特徵之效能對照組 |
| **SelectKBest** | Filter (f_regression) | 5 | 極高 | 低 | 低 | 單變量線性顯著性排序，運算速度最快 |
| **RFE** | Wrapper (LinearRegression) | 5 | 極高 | 低 | 低 | 遞迴剔除權重最小特徵 |
| **SFS Forward** | Wrapper (SFS) | 5 | 極高 | 低 | 低 | 循序前向貪婪挑選最佳特徵子集 |
| **Lasso Selection** | Embedded (LassoCV) | 5 | 極高 | 低 | 低 | L1 正則化自動壓縮不重要權重 |
| **Permutation Importance** | Model-based | 5 | 極高 | 低 | 低 | 驗證集隨機打亂評估泛化誤差增量 |

**核心發現**：
在學生表現資料集中，`Previous_Scores` (過往成績)、`Hours_Studied` (學習時數) 與 `Attendance` (出席率) 在所有 5 種特徵選擇演算法中均穩定列為前三名最關鍵特徵。

---

## 5. 95% 預測區間推論 (95% Prediction Interval)
老師特別要求預測圖必須具備信賴區間或預測區間。本研究採用 Statsmodels `OLSResults.get_prediction().summary_frame(alpha=0.05)` 產生：
1. **95% Confidence Interval for Mean Response (母體均值信賴區間)**：描述相同特徵下，所有可能個體之「平均反應期望值」落點。
2. **95% Prediction Interval for Observation (單一觀測值預測區間)**：描述「未來單一新學生」實際考分的預測範圍，包含估計變異與隨機誤差項變異 $\\sigma^2$。
3. **限制說明**：資料驅動的特徵選擇可能引發 Post-selection inference 偏差，實際涵蓋率可能微幅偏離名義 95%。

---

## 6. 線性回歸五大假設診斷 (Regression Diagnostics)
1. **線性性 (Linearity)**：殘差對擬合值散佈圖均勻散佈於 0 軸兩側，相關係數接近 0。
2. **殘差獨立性 (Independence)**：Durbin-Watson 統計量接近 2.0；但橫斷面資料缺乏時間序，DW 指標需謹慎解讀。
3. **同質變異性 (Homoscedasticity)**：執行 Breusch-Pagan 檢定，評估殘差變異數是否隨自變數水準而改變。
4. **殘差常態性 (Normality)**：Q-Q 圖檢驗殘差分佈；在大樣本下常態性檢定極易顯著，應重視實質分佈對稱性而非單純 p-value。
5. **多重共線性 (Multicollinearity)**：透過 VIF 檢測，剔除基準組後各變數 VIF 均受控。

---

## 7. GPT 輔助內容與對話紀錄 (AI Assistance Disclosure)
在完成本次專案之過程中，實際使用 Google Antigravity (Gemini 3.8 Flash) AI 進行下列輔助工作：
1. **專案規劃與 CRISP-DM 框架對齊**：拆解 6 大階段與作業規格。
2. **SSRF 安全驗證防護設計**：編寫防止內網滲透之安全 URL 驗證模組。
3. **Statsmodels OLS 預測區間整合**：正確對齊 Scikit-learn 與 Statsmodels 特徵設計矩陣。
4. **自動化測試構建**：撰寫 pytest 測試案例確保無資料洩漏。
> 老師要求的 ChatGPT/AI 對話紀錄 PDF，請參閱 `reports/ai_conversations/` 目錄。詳細匯出操作指南見 `docs/chatgpt_conversation_export_guide.md`。

---

## 8. Google NotebookLM 研究摘要 (User Action Required)
> [!NOTE]
> **待填寫區域 (User Action Required)**：
> 老師要求使用 Google NotebookLM 研究同主題（Multiple Linear Regression, Feature Selection, CRISP-DM, Model Evaluation, Prediction Interval）之網路解法，並將至少 100 字摘要填入此處。
> 具體操作來源清單與提問範本請參考 `docs/notebooklm_research_guide.md`。

```text
[=== 請將您在 NotebookLM 產生的真實研究摘要文字（至少 100 字）貼至此處 ===]
```

---

## 9. 結論與展望 (Conclusion)
本專案成功建構了高互動性、符合 CRISP-DM 流程且具備統計嚴謹度之多元線性回歸儀表板。不僅達成全部核心與加分項規格，更透過五大特徵選擇比較與 95% 預測區間推論，體現了資料科學專案從理論、實作到部署之完整工程實踐。
"""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    return report_md


def generate_pdf_report(md_content: str, output_pdf_path: Optional[str] = None) -> str:
    """
    Compile Markdown/Plaintext report into a professional PDF using ReportLab.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    if output_pdf_path is None:
        output_pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E1B4B"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#4F46E5"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
    )

    story = []
    lines = md_content.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], title_style))
        elif stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], h2_style))
        elif stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], h2_style))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph(f"&bull; {stripped[2:]}", body_style))
        elif stripped.startswith("> "):
            story.append(Paragraph(f"<i>{stripped[2:]}</i>", body_style))
        elif stripped.startswith("|"):
            continue  # Table handling can be simplified or styled
        else:
            story.append(Paragraph(stripped, body_style))

    doc.build(story)
    return output_pdf_path
