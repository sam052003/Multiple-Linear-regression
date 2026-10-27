"""
Reports Router.
Provides endpoints for retrieving final Markdown reports, metric CSVs, and PDF downloads.
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from src.config import REPORTS_DIR, RESULTS_DIR
from src.reporting import build_markdown_report, generate_pdf_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/markdown", response_class=PlainTextResponse)
def get_markdown_report():
    """Retrieve the comprehensive final analysis report Markdown."""
    report_file = os.path.join(REPORTS_DIR, "final_report.md")
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            return f.read()
    # If not generated yet, build on the fly
    return build_markdown_report()


@router.get("/pdf")
def download_pdf_report():
    """Download the compiled final analysis report PDF."""
    pdf_file = os.path.join(REPORTS_DIR, "final_report.pdf")
    if not os.path.exists(pdf_file):
        md = build_markdown_report()
        generate_pdf_report(md, pdf_file)

    if not os.path.exists(pdf_file):
        raise HTTPException(status_code=404, detail="找不到 final_report.pdf")

    return FileResponse(
        pdf_file,
        media_type="application/pdf",
        filename="Multiple_Linear_Regression_Final_Report.pdf",
    )


@router.get("/download/{filename}")
def download_report_file(filename: str):
    """Download generated CSV result files (e.g. student_feature_selection_comparison.csv)."""
    # Security check: prevent path traversal
    safe_name = os.path.basename(filename)
    file_path = os.path.join(RESULTS_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"找不到請求之檔案: {safe_name}")

    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=safe_name,
    )
