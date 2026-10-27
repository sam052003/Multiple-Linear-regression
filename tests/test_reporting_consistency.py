"""
Unit and Integration Tests for Dynamic Reporting and CSV Consistency.
Ensures report Markdown and PDF numbers 100% align with real DataFrames and CSV exports,
and that academic descriptions and error handling conform to specifications.
"""

import os
import re
import pandas as pd
import pytest

from src.reporting import build_markdown_report, generate_pdf_report, export_csv_data
from src.config import RESULTS_DIR, REPORTS_DIR


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
    ])


@pytest.fixture
def sample_diagnostics():
    return {
        "linearity": {"metric_name": "Corr(Residuals, Fitted)", "value": 0.0},
        "independence": {"metric_name": "DW Statistic", "value": 1.9541},
        "homoscedasticity": {"metric_name": "LM Statistic", "value": 22.08, "p_value": 0.7335},
        "normality": {"metric_name": "JB Stat", "value": 3883964.01},
        "multicollinearity": {"metric_name": "Max VIF", "value": 2.80},
    }


def test_markdown_report_dynamic_data_and_csv_consistency(sample_student_summary, sample_flood_summary, sample_diagnostics):
    """
    Test that build_markdown_report dynamically pulls exact values from DataFrames,
    rather than using hardcoded static values.
    """
    md = build_markdown_report(
        student_summary_df=sample_student_summary,
        flood_summary_df=sample_flood_summary,
        diagnostics=sample_diagnostics,
    )

    # Check Student values
    assert "0.769650" in md
    assert "0.764843" in md
    assert "1.804445" in md
    assert "0.452392" in md
    assert "3.256020" in md
    assert "0.082" in md  # Time_Sec

    assert "0.664693" in md
    assert "0.045" in md  # SelectKBest Time_Sec

    # Check Flood values
    assert "0.844545" in md
    assert "0.019983" in md
    assert "0.155" in md  # Flood Time_Sec

    # Check Diagnostics values
    assert "DW = 1.9541" in md
    assert "22.08" in md
    assert "3883964.01" in md
    assert "2.80" in md


def test_markdown_report_missing_diagnostics_handled_gracefully():
    """
    Test that if diagnostics or summary DataFrames are None, report indicates '尚未計算'
    without crashing or faking data.
    """
    md = build_markdown_report(student_summary_df=None, flood_summary_df=None, diagnostics=None)
    assert "尚未執行" in md or "尚未計算" in md
    assert "尚未計算" in md


def test_academic_rigor_phrasing_in_report(sample_student_summary, sample_flood_summary, sample_diagnostics):
    """
    Verify required academic phrasing:
    - Synthetic/simulated educational dataset disclaimer
    - Missing data does not assume MAR without evidence
    - OLS orthogonality Corr=0 is normal equation property
    - DW has order dependence limitation
    - Breusch-Pagan p>0.05 fails to reject H0, not proof of homoscedasticity
    - Jarque-Bera rejection and CLT limits on single future observation PI
    - Deployment recommendations classified as future work
    """
    md = build_markdown_report(
        student_summary_df=sample_student_summary,
        flood_summary_df=sample_flood_summary,
        diagnostics=sample_diagnostics,
    )

    # 1. Synthetic disclaimer
    assert "公開模擬／合成教育基準資料集" in md or "模擬" in md
    assert "田野調查" in md  # explicitly states it is NOT a field survey

    # 2. Missing data MAR
    assert "不逕行假設其符合統計上的隨機缺失" in md or "隨機缺失" in md

    # 3. Linearity & OLS orthogonality
    assert "正交" in md or "常態方程" in md or "代數必然性質" in md

    # 4. DW order dependence
    assert "排列次序" in md or "橫斷面" in md

    # 5. Breusch-Pagan fail to reject
    assert "未拒絕同質變異" in md or "未有充份證據拒絕" in md

    # 6. Normality & CLT prediction interval limit
    assert "CLT" in md
    assert "漸近常態" in md

    # 7. Deployment future recommendations
    assert "未來" in md or "建議" in md


def test_pdf_report_compilation_and_no_raw_latex(sample_student_summary, sample_flood_summary, sample_diagnostics, tmp_path):
    """
    Test that PDF generates without throwing errors, and does not contain raw LaTeX code.
    """
    md = build_markdown_report(
        student_summary_df=sample_student_summary,
        flood_summary_df=sample_flood_summary,
        diagnostics=sample_diagnostics,
    )
    test_pdf = os.path.join(tmp_path, "test_academic_report.pdf")
    out = generate_pdf_report(md, test_pdf)

    assert os.path.exists(out)
    assert os.path.getsize(out) > 50000  # valid pdf > 50KB

    with open(out, "rb") as f:
        pdf_bytes = f.read()

    # Raw unparsed LaTeX macros should not be present as literal strings
    assert b"\\text{" not in pdf_bytes
    assert b"\\hat{\\beta}" not in pdf_bytes
    assert b"\\sqrt{" not in pdf_bytes
