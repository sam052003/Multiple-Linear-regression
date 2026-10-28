"""
Unit and Integration Tests for Dynamic Reporting, CSV Consistency, and Side-Effect Prevention.
Guarantees:
1. build_markdown_report is pure by default and NEVER mutates reports/final_report.md.
2. generate_pdf_report requires an explicit output path and never mutates production reports/final_report.pdf.
3. pytest execution does not alter production reports or CSVs (verified by SHA-256 hashes).
4. Official reports contain all 6 feature selection methods exactly matching the CSV results.
5. Best K=5 model and retention rates are dynamically calculated from full results.
6. Diagnostics dynamically respond to values and p-values (p < 0.05 vs p >= 0.05).
7. PDF generation verified via text extraction and layout inspection using pdfplumber.
"""

import os
import hashlib
import pandas as pd
import pytest
import pdfplumber

from src.reporting import build_markdown_report, generate_pdf_report
from src.config import RESULTS_DIR, REPORTS_DIR


def get_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file if it exists, else empty string."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def verify_production_deliverables_unmodified_by_pytest():
    """
    Session-level autouse fixture: records SHA-256 hashes of all production deliverables
    before tests run and asserts they remain 100% identical after all tests finish.
    """
    target_files = [
        os.path.join(REPORTS_DIR, "final_report.md"),
        os.path.join(REPORTS_DIR, "final_report.pdf"),
        os.path.join(REPORTS_DIR, "notebooklm_summary.md"),
        os.path.join(RESULTS_DIR, "student_feature_selection_comparison.csv"),
        os.path.join(RESULTS_DIR, "flood_feature_selection_comparison.csv"),
        os.path.join(RESULTS_DIR, "student_predictions_intervals.csv"),
    ]
    initial_hashes = {f: get_file_hash(f) for f in target_files if os.path.exists(f)}

    yield

    for f, h_before in initial_hashes.items():
        assert os.path.exists(f), f"Production file '{f}' was deleted during pytest session!"
        h_after = get_file_hash(f)
        assert h_before == h_after, (
            f"Production deliverable '{f}' was modified during pytest session!\n"
            f"Hash before: {h_before}\nHash after:  {h_after}"
        )


@pytest.fixture
def sample_student_summary():
    return pd.DataFrame([
        {
            "Method": "Baseline (All Features)",
            "Features": 27,
            "R2": 0.769650,
            "Adj_R2": 0.764843,
            "RMSE": 1.804445,
            "MAE": 0.452392,
            "MSE": 3.256020,
            "Train_R2": 0.717314,
            "CV_R2_Mean": 0.715585,
            "Time_Sec": 0.082,
        },
        {
            "Method": "SelectKBest (Filter)",
            "Features": 5,
            "R2": 0.664693,
            "Adj_R2": 0.663419,
            "RMSE": 2.177060,
            "MAE": 1.174931,
            "MSE": 4.739588,
            "Train_R2": 0.605658,
            "CV_R2_Mean": 0.606164,
            "Time_Sec": 0.045,
        },
    ])


@pytest.fixture
def sample_flood_summary():
    return pd.DataFrame([
        {
            "Method": "Baseline (All Features)",
            "Features": 20,
            "R2": 0.844545,
            "Adj_R2": 0.844025,
            "RMSE": 0.019983,
            "MAE": 0.015939,
            "MSE": 0.000399,
            "Train_R2": 0.845726,
            "CV_R2_Mean": 0.845357,
            "Time_Sec": 0.155,
        },
        {
            "Method": "RFE (Wrapper)",
            "Features": 5,
            "R2": 0.180733,
            "Adj_R2": 0.180050,
            "RMSE": 0.045873,
            "MAE": 0.037177,
            "MSE": 0.002104,
            "Train_R2": 0.190346,
            "CV_R2_Mean": 0.189762,
            "Time_Sec": 0.277,
        },
    ])


@pytest.fixture
def sample_diagnostics():
    return {
        "linearity": {"metric_name": "Corr(Residuals, Fitted)", "value": 0.0},
        "independence": {"metric_name": "DW Statistic", "value": 1.9541},
        "homoscedasticity": {"metric_name": "LM Statistic", "value": 22.08, "p_value": 0.7335},
        "normality": {"metric_name": "JB Stat", "value": 3883964.01, "p_value": 0.0},
        "multicollinearity": {"metric_name": "Max VIF", "value": 2.80},
    }


def test_build_markdown_report_purity_and_zero_side_effects(sample_student_summary, sample_flood_summary, sample_diagnostics):
    """
    Test 1: Ensure build_markdown_report is pure by default.
    Calling it with output_path=None MUST NEVER modify reports/final_report.md or any production artifact.
    """
    prod_md_path = os.path.join(REPORTS_DIR, "final_report.md")
    hash_before = get_file_hash(prod_md_path)

    # Invoke with mock fixtures without output_path
    md = build_markdown_report(
        student_summary_df=sample_student_summary,
        flood_summary_df=sample_flood_summary,
        diagnostics=sample_diagnostics,
    )
    assert isinstance(md, str)
    assert len(md) > 1000

    # Invoke with empty inputs
    md_empty = build_markdown_report(None, None, None)
    assert isinstance(md_empty, str)

    hash_after = get_file_hash(prod_md_path)
    assert hash_before == hash_after, "build_markdown_report unexpectedly modified production reports/final_report.md!"


def test_dynamic_report_with_custom_fixtures_and_no_stale_data():
    """
    Test 2: When fixture data changes, the report dynamically updates its numbers
    and retains zero stale metrics or old results.
    """
    custom_df_a = pd.DataFrame([
        {"Method": "Baseline (All Features)", "Features": 27, "R2": 0.812345, "Adj_R2": 0.810000, "RMSE": 1.234567, "MAE": 0.345678, "MSE": 1.524155, "Train_R2": 0.80, "CV_R2_Mean": 0.79, "Time_Sec": 0.111},
        {"Method": "RFE (Wrapper)", "Features": 5, "R2": 0.712345, "Adj_R2": 0.710000, "RMSE": 1.567890, "MAE": 0.456789, "MSE": 2.458279, "Train_R2": 0.70, "CV_R2_Mean": 0.69, "Time_Sec": 0.222},
    ])
    md_a = build_markdown_report(student_summary_df=custom_df_a)
    dynamic_body_a = md_a.split("## 9.")[0]
    assert "0.812345" in dynamic_body_a
    assert "0.712345" in dynamic_body_a
    assert "0.769650" not in dynamic_body_a  # Dynamic modeling sections must not retain stale default number

    custom_df_b = pd.DataFrame([
        {"Method": "Baseline (All Features)", "Features": 27, "R2": 0.923456, "Adj_R2": 0.920000, "RMSE": 0.987654, "MAE": 0.234567, "MSE": 0.975460, "Train_R2": 0.91, "CV_R2_Mean": 0.90, "Time_Sec": 0.333},
        {"Method": "SelectKBest (Filter)", "Features": 5, "R2": 0.823456, "Adj_R2": 0.820000, "RMSE": 1.112233, "MAE": 0.334455, "MSE": 1.237062, "Train_R2": 0.81, "CV_R2_Mean": 0.80, "Time_Sec": 0.044},
    ])
    md_b = build_markdown_report(student_summary_df=custom_df_b)
    assert "0.923456" in md_b
    assert "0.823456" in md_b
    assert "0.812345" not in md_b  # Must not retain previous test numbers

    # Test changing diagnostics: DW=1.2000 (positive autocorrelation) and VIF=8.5
    custom_diag = {
        "independence": {"metric_name": "DW", "value": 1.2000},
        "multicollinearity": {"metric_name": "VIF", "value": 8.50},
    }
    md_diag = build_markdown_report(diagnostics=custom_diag)
    assert "DW = 1.2000" in md_diag
    assert "存在正自相關" in md_diag
    assert "目前排序無相依" not in md_diag
    assert "1.9541" not in md_diag
    assert "8.50" in md_diag
    assert "存在共線性疑慮" in md_diag
    assert "優異受控" not in md_diag
    assert "2.80" not in md_diag


def test_diagnostics_p_value_responsiveness_and_dynamic_conclusions():
    """
    Test 3: Diagnostic formatting must correctly respond to p-value values:
    If p < 0.05, it must not display (>= 0.05) and must indicate rejection.
    If p >= 0.05, it must indicate failure to reject H0.
    """
    # Case A: Significant rejection of homoscedasticity (p < 0.05)
    diag_sig = {
        "homoscedasticity": {"value": 55.4, "p_value": 0.0012},
    }
    md_sig = build_markdown_report(diagnostics=diag_sig)
    assert "p = 0.0012 (< 0.05)" in md_sig
    assert ">= 0.05" not in md_sig.split("同質變異性")[1].split("\n")[0]
    assert "≥ 0.05" not in md_sig.split("同質變異性")[1].split("\n")[0]
    assert "顯著拒絕同質變異" in md_sig

    # Case B: Failure to reject homoscedasticity (p >= 0.05)
    diag_nonsig = {
        "homoscedasticity": {"value": 22.08, "p_value": 0.7335},
    }
    md_nonsig = build_markdown_report(diagnostics=diag_nonsig)
    assert "p = 0.7335 (>= 0.05)" in md_nonsig
    assert "未拒絕同質變異虛無假設" in md_nonsig
    assert "顯著拒絕同質變異" not in md_nonsig


def test_official_production_markdown_and_csv_consistency():
    """
    Test 4: Verify production final_report.md contains all 6 methods for both Student and Flood datasets,
    matching the official result CSVs exactly.
    """
    s_csv_path = os.path.join(RESULTS_DIR, "student_feature_selection_comparison.csv")
    f_csv_path = os.path.join(RESULTS_DIR, "flood_feature_selection_comparison.csv")
    md_path = os.path.join(REPORTS_DIR, "final_report.md")

    if not os.path.exists(md_path) or not os.path.exists(s_csv_path) or not os.path.exists(f_csv_path):
        pytest.skip("Production report or CSVs not generated yet; run scripts/run_experiments.py first.")

    s_df = pd.read_csv(s_csv_path)
    f_df = pd.read_csv(f_csv_path)
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Check 6 methods in Student table and verify all metric numbers match CSV
    assert len(s_df) == 6, f"Expected 6 methods in Student CSV, got {len(s_df)}"
    for _, r in s_df.iterrows():
        method = str(r["Method"])
        assert method in md_text, f"Student method '{method}' missing from final_report.md!"
        assert f"{r['R2']:.6f}" in md_text, f"Student R2 {r['R2']:.6f} for {method} missing from final_report.md!"
        assert f"{r['RMSE']:.6f}" in md_text, f"Student RMSE {r['RMSE']:.6f} for {method} missing from final_report.md!"

    # Check 6 methods in Flood table and verify all metric numbers match CSV
    assert len(f_df) == 6, f"Expected 6 methods in Flood CSV, got {len(f_df)}"
    for _, r in f_df.iterrows():
        method = str(r["Method"])
        assert method in md_text, f"Flood method '{method}' missing from final_report.md!"
        assert f"{r['R2']:.6f}" in md_text, f"Flood R2 {r['R2']:.6f} for {method} missing from final_report.md!"
        assert f"{r['RMSE']:.6f}" in md_text, f"Flood RMSE {r['RMSE']:.6f} for {method} missing from final_report.md!"

    # Student Best K=5 must be Permutation Importance
    s_k5 = s_df[s_df["Features"] == 5].sort_values("R2", ascending=False).iloc[0]
    assert "Permutation Importance" in str(s_k5["Method"])
    assert f"{s_k5['R2']:.6f}" in md_text

    # Flood Best K=5 must be RFE
    f_k5 = f_df[f_df["Features"] == 5].sort_values("R2", ascending=False).iloc[0]
    assert "RFE" in str(f_k5["Method"])
    assert f"{f_k5['R2']:.6f}" in md_text

    # There should be no erroneous "尚未計算" for completed metrics
    assert "{{S_K5_R2}}" not in md_text
    assert "{{F_K5_R2}}" not in md_text
    assert "{{STUDENT_TABLE}}" not in md_text
    assert "{{FLOOD_TABLE}}" not in md_text
    assert "{{DIAGNOSTICS_TABLE}}" not in md_text


def test_academic_rigor_phrasing_in_report():
    """
    Test 5: Verify required academic phrasing and epistemological rigor.
    """
    md = build_markdown_report()

    # 1. Synthetic benchmark disclaimer
    assert "公開模擬／合成教育基準資料集" in md or "模擬" in md
    assert "田野調查" in md

    # 2. Missing data MAR
    assert "不逕行假設其符合統計上的隨機缺失" in md or "隨機缺失" in md

    # 3. Linearity & OLS orthogonality
    assert "正交" in md or "常態方程" in md

    # 4. DW order dependence
    assert "排列次序" in md or "橫斷面" in md

    # 5. Breusch-Pagan fail to reject
    assert "未拒絕同質變異" in md or "未有充份證據拒絕" in md

    # 6. Normality & CLT prediction interval limit
    assert "CLT" in md
    assert "漸近常態" in md

    # 7. Deployment future recommendations
    assert "未來" in md or "建議" in md


def test_pdf_report_compilation_and_layout_via_pdfplumber(tmp_path, sample_student_summary, sample_flood_summary, sample_diagnostics):
    """
    Test 6: Verify PDF generation uses tmp_path, does not modify production PDF,
    and extracts text cleanly via pdfplumber without LaTeX leaks or missing glyphs.
    """
    prod_pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")
    hash_before = get_file_hash(prod_pdf_path)

    md = build_markdown_report(
        student_summary_df=sample_student_summary,
        flood_summary_df=sample_flood_summary,
        diagnostics=sample_diagnostics,
    )
    test_pdf = os.path.join(tmp_path, "test_academic_report.pdf")
    out = generate_pdf_report(md, output_pdf_path=test_pdf)

    assert os.path.exists(out)
    assert os.path.getsize(out) > 50000

    # Verify production PDF was NOT touched
    hash_after = get_file_hash(prod_pdf_path)
    assert hash_before == hash_after, "generate_pdf_report unexpectedly modified production reports/final_report.pdf!"

    # Verify with pdfplumber
    with pdfplumber.open(out) as pdf:
        assert len(pdf.pages) >= 5
        full_test_text = ""
        for p_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            full_test_text += "\n" + text
            assert "\x00" not in text, f"Missing glyph (null char) found on page {p_idx + 1}"
            assert text.count("$") == 0, f"Raw $ delimiter found on page {p_idx + 1}"
            assert "\\text{" not in text, f"Raw \\text found on page {p_idx + 1}"
            assert "\\hat{\\beta}" not in text, f"Raw \\hat{{\\beta}} found on page {p_idx + 1}"
            assert "\\sqrt" not in text, f"Raw \\sqrt found on page {p_idx + 1}"

        # Mathematical formulas and key terms must be extracted properly without raw LaTeX
        assert "CI" in full_test_text
        assert "PI" in full_test_text
        assert "β_hat" in full_test_text
        assert "x_0" in full_test_text


def test_official_production_pdf_matches_markdown_and_csv():
    """
    Test 7: Verify official reports/final_report.pdf matches CSV and Markdown deliverables,
    containing all 6 methods and correct metrics extracted via pdfplumber.
    """
    prod_pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")
    s_csv_path = os.path.join(RESULTS_DIR, "student_feature_selection_comparison.csv")
    f_csv_path = os.path.join(RESULTS_DIR, "flood_feature_selection_comparison.csv")

    if not os.path.exists(prod_pdf_path) or not os.path.exists(s_csv_path) or not os.path.exists(f_csv_path):
        pytest.skip("Production deliverables missing.")

    s_df = pd.read_csv(s_csv_path)
    f_df = pd.read_csv(f_csv_path)

    all_table_methods = []
    with pdfplumber.open(prod_pdf_path) as pdf:
        assert len(pdf.pages) >= 10, f"Expected publication PDF >= 10 pages, got {len(pdf.pages)}"
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row and row[0]:
                        cell_clean = " ".join(row[0].split())
                        all_table_methods.append(cell_clean)

    # Check 6 methods in PDF table cells
    for method in s_df["Method"]:
        assert any(method in cell for cell in all_table_methods), f"Student method '{method}' not found in official PDF tables!"
    for method in f_df["Method"]:
        assert any(method in cell for cell in all_table_methods), f"Flood method '{method}' not found in official PDF tables!"

    # Check best K=5 R2 in PDF text
    s_k5 = s_df[s_df["Features"] == 5].sort_values("R2", ascending=False).iloc[0]
    assert f"{s_k5['R2']:.6f}" in full_text, f"Student best K=5 R2 {s_k5['R2']:.6f} missing from official PDF!"

    f_k5 = f_df[f_df["Features"] == 5].sort_values("R2", ascending=False).iloc[0]
    assert f"{f_k5['R2']:.6f}" in full_text, f"Flood best K=5 R2 {f_k5['R2']:.6f} missing from official PDF!"

    # Verify no raw LaTeX leaks
    assert "\\text{" not in full_text
    assert "\\hat{\\beta}" not in full_text
    assert "\\sqrt{" not in full_text
    assert "\x00" not in full_text


def test_notebooklm_summary_file_completed_and_valid():
    """
    Test 8: Verify reports/notebooklm_summary.md contains the actual user NotebookLM research,
    marks status as completed, removes placeholders/samples, and lists the exact 6 sources.
    """
    summary_path = os.path.join(REPORTS_DIR, "notebooklm_summary.md")
    assert os.path.exists(summary_path), "reports/notebooklm_summary.md must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "已完成 NotebookLM 實際研究" in text
    assert "待填寫" not in text
    assert "範例填寫結構參考" not in text
    assert "[===" not in text

    # Key research reflection statements
    assert "本研究透過 Google NotebookLM 研讀 CRISP-DM、多元線性回歸及特徵選擇相關文獻" in text
    assert "LassoCV 結合 L1 正則化與交叉驗證選擇懲罰參數 α，並不保證固定選取五個特徵" in text
    assert "0.769650" in text and "0.666460" in text
    assert "0.844545" in text and "0.180733" in text
    assert "將前處理與特徵選擇整合進 Pipeline，使交叉驗證各折獨立擬合，降低資料洩漏風險" in text
    assert "95% 信賴區間（CI）用於估計條件平均值，95% 預測區間（PI）則針對個別新觀測值" in text
    assert "本研究的 Student 資料屬於合成資料，無法直接推論真實教育情境中的因果關係" in text

    # Exact 6 references
    expected_refs = [
        "IBM CRISP-DM",
        "https://www.ibm.com/docs/zh-tw/spss-modeler/saas?topic=dm-crisp-help-overview",
        "Scikit-learn Feature Selection",
        "https://scikit-learn.org/stable/modules/feature_selection.html",
        "Scikit-learn Linear Models",
        "https://scikit-learn.org/stable/modules/linear_model.html",
        "Scikit-learn Common Pitfalls",
        "https://scikit-learn.org/stable/common_pitfalls.html",
        "Statsmodels get_prediction",
        "https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_prediction.html",
        "專案內部研究成果",
        "reports/final_report.pdf",
    ]
    for item in expected_refs:
        assert item in text, f"Expected reference item '{item}' missing from reports/notebooklm_summary.md"


def test_build_markdown_report_integrates_real_notebooklm_and_references(
    sample_student_summary, sample_flood_summary, sample_diagnostics
):
    """
    Test 9: Verify build_markdown_report dynamically embeds Section 9.1 and Section 10 References
    from reports/notebooklm_summary.md without side effects, removes '待填寫' for NotebookLM,
    avoids circular self-citation, and keeps Section 9.2 ChatGPT PDF pending.
    """
    md = build_markdown_report(
        student_summary_df=sample_student_summary,
        flood_summary_df=sample_flood_summary,
        diagnostics=sample_diagnostics,
    )

    assert "### 9.1" in md
    assert "已完成 NotebookLM 實際研究" in md
    assert "本研究透過 Google NotebookLM 研讀 CRISP-DM、多元線性回歸及特徵選擇相關文獻" in md
    assert "LassoCV 結合 L1 正則化與交叉驗證選擇懲罰參數 α" in md
    assert "95% 信賴區間（CI）用於估計條件平均值，95% 預測區間（PI）則針對個別新觀測值" in md
    assert "待填寫" not in md
    assert "[=== 待補欄位" not in md

    # Section 9.2 must still honestly remain pending (待補)
    sec_9_2 = md.split("### 9.2")[1].split("## 10.")[0]
    assert "待補" in sec_9_2
    assert "chatgpt_conversation_record.pdf" in sec_9_2

    # Section 10 References must contain all 5 URLs and internal empirical cross-reference without self-citation loop
    sec_10 = md.split("## 10. 參考文獻")[1].split("## 11.")[0]
    assert "https://www.ibm.com/docs/zh-tw/spss-modeler/saas?topic=dm-crisp-help-overview" in sec_10
    assert "https://scikit-learn.org/stable/modules/feature_selection.html" in sec_10
    assert "https://scikit-learn.org/stable/modules/linear_model.html" in sec_10
    assert "https://scikit-learn.org/stable/common_pitfalls.html" in sec_10
    assert (
        "https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_prediction.html"
        in sec_10
    )
    assert "專案內部研究成果" in sec_10
    assert "final_report.pdf" not in sec_10  # Must not circularly self-cite final_report.pdf inside final_report


def test_missing_notebooklm_summary_fallback_and_anti_loop_guards(tmp_path):
    """
    Test 10: Verify missing NotebookLM file displays explicit pending notice,
    and self-reference / full-report duplication attempts are blocked.
    """
    missing_path = str(tmp_path / "does_not_exist_notebooklm.md")
    md_missing = build_markdown_report(notebooklm_summary_path=missing_path)
    sec_9_1 = md_missing.split("### 9.1")[1].split("### 9.2")[0]
    assert "待補" in sec_9_1
    assert "尚未" in sec_9_1

    # Self-reference guard: summary_path == output_path
    same_path = str(tmp_path / "same_report.md")
    with pytest.raises(ValueError, match="self-reference"):
        build_markdown_report(output_path=same_path, notebooklm_summary_path=same_path)

    # Self-reference guard: pointing to final_report.md
    fake_final = tmp_path / "final_report.md"
    fake_final.write_text("dummy", encoding="utf-8")
    with pytest.raises(ValueError, match="self-reference"):
        build_markdown_report(notebooklm_summary_path=str(fake_final))

    # Full-report duplication guard
    duplicated_report_file = tmp_path / "bad_summary.md"
    duplicated_report_file.write_text(md_missing, encoding="utf-8")
    with pytest.raises(ValueError, match="full research report"):
        build_markdown_report(notebooklm_summary_path=str(duplicated_report_file))


def test_repeated_report_generation_preserves_summary_without_loss_or_duplication(
    tmp_path, sample_student_summary, sample_flood_summary, sample_diagnostics
):
    """
    Test 11: Running report generation multiple times must never lose the NotebookLM summary
    and must produce identical output without recursive growth.
    """
    summary_path = os.path.join(REPORTS_DIR, "notebooklm_summary.md")
    summary_hash_before = get_file_hash(summary_path)

    out_md = str(tmp_path / "repeated_final_report.md")
    out_pdf = str(tmp_path / "repeated_final_report.pdf")

    outputs = []
    for _ in range(3):
        md_out = build_markdown_report(
            student_summary_df=sample_student_summary,
            flood_summary_df=sample_flood_summary,
            diagnostics=sample_diagnostics,
            output_path=out_md,
        )
        generate_pdf_report(md_out, output_pdf_path=out_pdf)
        outputs.append(md_out)

    assert outputs[0] == outputs[1] == outputs[2]
    assert "本研究透過 Google NotebookLM 研讀 CRISP-DM" in outputs[-1]
    assert outputs[-1].count("### 9.1") == 1
    assert get_file_hash(summary_path) == summary_hash_before


def test_official_pdf_extracts_notebooklm_summary_and_references():
    """
    Test 12: Verify official reports/final_report.md and reports/final_report.pdf contain
    the real NotebookLM research reflection, accurate numbered references with URLs,
    no '待填寫', and pending Section 9.2.
    """
    prod_md_path = os.path.join(REPORTS_DIR, "final_report.md")
    prod_pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")
    assert os.path.exists(prod_md_path)
    assert os.path.exists(prod_pdf_path)

    with open(prod_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "已完成 NotebookLM 實際研究" in md_text
    assert "本研究透過 Google NotebookLM 研讀 CRISP-DM" in md_text
    assert "待填寫" not in md_text

    with pdfplumber.open(prod_pdf_path) as pdf:
        full_pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        # Normalize line breaks within extracted text for URL verification
        compact_pdf_text = "".join((page.extract_text() or "").replace("\n", "") for page in pdf.pages)

    # Verify NotebookLM summary content in PDF
    assert "Google NotebookLM" in full_pdf_text
    assert "已完成 NotebookLM 實際研究" in full_pdf_text
    assert "LassoCV" in full_pdf_text
    assert "0.769650" in full_pdf_text
    assert "0.666460" in full_pdf_text
    assert "0.844545" in full_pdf_text
    assert "0.180733" in full_pdf_text
    assert "Pipeline" in full_pdf_text
    assert "95% 信賴區間" in full_pdf_text
    assert "95% 預測區間" in full_pdf_text
    assert "合成資料" in full_pdf_text
    assert "待填寫" not in full_pdf_text

    # Verify Section 9.2 still retains 待補
    assert "9.2 ChatGPT / AI 協作對話紀錄" in full_pdf_text
    assert "待補" in full_pdf_text

    # Verify all 5 external URLs and internal study item in PDF references
    expected_urls = [
        "https://www.ibm.com/docs/zh-tw/spss-modeler/saas?topic=dm-crisp-help-overview",
        "https://scikit-learn.org/stable/modules/feature_selection.html",
        "https://scikit-learn.org/stable/modules/linear_model.html",
        "https://scikit-learn.org/stable/common_pitfalls.html",
        "https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_prediction.html",
    ]
    for url in expected_urls:
        assert url in compact_pdf_text, f"Reference URL '{url}' missing from official PDF!"
    assert "專案內部研究成果" in full_pdf_text


def test_graduate_course_identification_and_legacy_guard(tmp_path):
    """
    Test 13: Verify graduate-level course identification (碩士班課程 HW2 / Graduate-Level Course Project)
    across build_markdown_report(), official Markdown, official PDF (cover + headers + footers on all pages),
    zero fabricated university/course names or wrong academic level terms, and full-report duplication guard
    compatibility with both new and legacy report titles.
    """
    md_dynamic = build_markdown_report()
    assert md_dynamic.startswith("# 碩士班課程 HW2 — 多元線性回歸專題研究報告\n")
    assert "碩士班課程作業（Graduate-Level Course Project）" in md_dynamic

    prod_md_path = os.path.join(REPORTS_DIR, "final_report.md")
    prod_pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")
    assert os.path.exists(prod_md_path)
    assert os.path.exists(prod_pdf_path)

    with open(prod_md_path, "r", encoding="utf-8") as f:
        prod_md = f.read()

    assert prod_md.startswith("# 碩士班課程 HW2 — 多元線性回歸專題研究報告")
    assert "碩士班課程作業（Graduate-Level Course Project）" in prod_md

    forbidden_terms = [
        "國立大學",
        "機器學習與統計數據分析",
        "大學「多元線性回歸",
        "大學『多元線性回歸",
        "大學部",
        "Bachelor",
        "Undergraduate",
        "Master's Thesis",
        "碩士學位論文",
        "碩士論文",
    ]
    for term in forbidden_terms:
        assert term not in md_dynamic, f"Forbidden academic term '{term}' found in generated Markdown!"
        assert term not in prod_md, f"Forbidden academic term '{term}' found in official final_report.md!"

    with pdfplumber.open(prod_pdf_path) as pdf:
        assert len(pdf.pages) >= 10
        page1_text = pdf.pages[0].extract_text() or ""
        assert "碩士班課程 HW2 — 多元線性回歸專題研究報告" in page1_text
        assert "Graduate-Level Course Project" in page1_text

        for p_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            assert (
                "碩士班課程｜多元線性回歸專題研究" in page_text
            ), f"Graduate-level footer missing on PDF page {p_idx + 1}!"
            if p_idx >= 1:
                assert (
                    "碩士班課程 HW2 — CRISP-DM 多元線性回歸專題研究報告" in page_text
                ), f"Graduate-level running header missing on PDF page {p_idx + 1}!"
            for term in forbidden_terms:
                assert (
                    term not in page_text
                ), f"Forbidden academic term '{term}' found on PDF page {p_idx + 1}!"

    # Verify full-report duplication guard blocks BOTH new and legacy report titles
    for idx, header_variant in enumerate(
        [
            "# 碩士班課程 HW2 — 多元線性回歸專題研究報告\n\n內容段落",
            "# 大學「多元線性回歸 (Multiple Linear Regression)」專題研究報告\n\n舊版內容",
            "# 大學『多元線性回歸』專題研究報告\n\n舊版內容",
        ]
    ):
        dup_path = tmp_path / f"dup_variant_{idx}.md"
        dup_path.write_text(header_variant, encoding="utf-8")
        with pytest.raises(ValueError, match="full research report"):
            build_markdown_report(notebooklm_summary_path=str(dup_path))

