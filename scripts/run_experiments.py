"""
Script: run_experiments.py
Description: Execute full CRISP-DM experiments on Student Performance (primary)
and Flood Probability (comparative) datasets. Generates metrics CSVs, static figures,
and compiles publication-quality Markdown and PDF reports.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure utf-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import statsmodels.api as sm

from src.config import (
    FIGURES_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
)
from src.data_loader import load_student_performance, load_flood_prediction
from src.preprocessing import prepare_dataset
from src.modeling import train_baseline_model
from src.evaluation import evaluate_regression_model
from src.feature_selection import run_feature_selection_comparison
from src.prediction_intervals import OLSPredictionModel
from src.statistical_tests import diagnose_regression_assumptions
from src.reporting import generate_static_figures, build_markdown_report, generate_pdf_report


def run_all_experiments():
    print("==================================================")
    print("[START] Starting CRISP-DM Automated Experiment Suite")
    print("==================================================")

    # 1. Main Research: Student Performance Factors
    print("\n[1/3] Running Experiments on Student Performance Factors...")
    df_student = load_student_performance()
    print(f"Loaded Student dataset: {df_student.shape[0]} rows, {df_student.shape[1]} columns")

    target_student = "Exam_Score"
    student_artifacts = prepare_dataset(
        df=df_student,
        target_col=target_student,
        test_size=0.20,
        random_state=42,
    )
    print(f"Train samples: {len(student_artifacts.X_train_proc)}, Test samples: {len(student_artifacts.X_test_proc)}")
    print(f"Encoded feature count: {len(student_artifacts.feature_names_out)}")

    # Run Feature Selection comparison (K=5, 5-fold CV)
    print("Running 5 Feature Selection methods on Student Performance (K=5)...")
    student_fs_df, student_fs_res, _ = run_feature_selection_comparison(
        X_train=student_artifacts.X_train_proc,
        y_train=student_artifacts.y_train,
        X_test=student_artifacts.X_test_proc,
        y_test=student_artifacts.y_test,
        k=5,
        cv_folds=5,
        random_state=42,
        fast_mode=False,
    )

    print("\n--- Student Performance Feature Selection Summary ---")
    print(student_fs_df[["Method", "Features", "R2", "Adj_R2", "RMSE", "MAE"]].to_string(index=False))

    # Fit Statsmodels OLS for 95% CI and PI
    ols_student = OLSPredictionModel()
    ols_student.fit(student_artifacts.X_train_proc, student_artifacts.y_train)
    student_pi_df = ols_student.predict_intervals(student_artifacts.X_test_proc, alpha=0.05)

    # Run 5 Regression Assumption Diagnostics
    print("\nRunning Regression Assumption Diagnostics on Student Model...")
    diagnostics = diagnose_regression_assumptions(
        ols_results=ols_student.ols_results,
        X_train=student_artifacts.X_train_proc,
        y_train=student_artifacts.y_train,
    )
    for diag_key, diag_val in diagnostics.items():
        print(f"  [{diag_val['diagnostic']}]: {diag_val.get('value')} ({diag_val.get('interpretation', '')[:50]}...)")

    # Save Student Results to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    student_fs_path = os.path.join(RESULTS_DIR, "student_feature_selection_comparison.csv")
    student_fs_df.to_csv(student_fs_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved Student FS comparison CSV to: {student_fs_path}")

    X_test_design = sm.add_constant(student_artifacts.X_test_proc, has_constant="add")
    y_test_pred_student = ols_student.ols_results.predict(X_test_design)
    student_preds_df = pd.DataFrame({
        "Actual_Exam_Score": student_artifacts.y_test.values,
        "Predicted_Exam_Score": y_test_pred_student.values,
        "Residual": student_artifacts.y_test.values - y_test_pred_student.values,
        "CI_Lower_95": student_pi_df["mean_ci_lower"].values,
        "CI_Upper_95": student_pi_df["mean_ci_upper"].values,
        "PI_Lower_95": student_pi_df["obs_ci_lower"].values,
        "PI_Upper_95": student_pi_df["obs_ci_upper"].values,
    })
    student_preds_path = os.path.join(RESULTS_DIR, "student_predictions_intervals.csv")
    student_preds_df.to_csv(student_preds_path, index=False, encoding="utf-8-sig")
    print(f"Saved Student predictions with 95% PI to: {student_preds_path}")

    # 2. Second Research: Flood Probability Prediction
    print("\n[2/3] Running Comparative Experiments on Flood Probability Prediction...")
    df_flood = load_flood_prediction(sample_size=30000, random_state=42)
    print(f"Loaded Flood sample: {df_flood.shape[0]} rows, {df_flood.shape[1]} columns")

    flood_artifacts = prepare_dataset(
        df=df_flood,
        target_col="FloodProbability",
        excluded_cols=["id"],
        test_size=0.20,
        random_state=42,
    )
    print(f"Train samples: {len(flood_artifacts.X_train_proc)}, Test samples: {len(flood_artifacts.X_test_proc)}")

    print("Running 5 Feature Selection methods on Flood Probability (K=5)...")
    flood_fs_df, _, _ = run_feature_selection_comparison(
        X_train=flood_artifacts.X_train_proc,
        y_train=flood_artifacts.y_train,
        X_test=flood_artifacts.X_test_proc,
        y_test=flood_artifacts.y_test,
        k=5,
        cv_folds=5,
        random_state=42,
        fast_mode=True,
    )
    flood_fs_path = os.path.join(RESULTS_DIR, "flood_feature_selection_comparison.csv")
    flood_fs_df.to_csv(flood_fs_path, index=False, encoding="utf-8-sig")
    print(f"Saved Flood FS comparison CSV to: {flood_fs_path}")

    print("\n--- Flood Probability Feature Selection Summary ---")
    print(flood_fs_df[["Method", "Features", "R2", "Adj_R2", "RMSE", "MAE"]].to_string(index=False))

    # 3. Generate Publication Figures
    print("\nGenerating static publication figures in reports/figures/...")
    generate_static_figures(
        df=df_student,
        y_test_true=student_artifacts.y_test.values,
        y_test_pred=y_test_pred_student.values,
        fs_summary_df=student_fs_df,
        target_name="Exam_Score",
        pi_df=student_pi_df,
        flood_summary_df=flood_fs_df,
    )
    print("Static PNG figures created successfully.")

    # 4. Build Final Report Markdown and PDF
    print("\n[3/3] Compiling Markdown and PDF Reports...")
    report_md_path = os.path.join(REPORTS_DIR, "final_report.md")
    md_content = build_markdown_report(
        student_summary_df=student_fs_df,
        flood_summary_df=flood_fs_df,
        diagnostics=diagnostics,
        output_path=report_md_path,
    )
    print(f"Saved final Markdown report to: {report_md_path}")
    pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")
    generate_pdf_report(md_content, output_pdf_path=pdf_path)
    print(f"Saved final PDF report to: {pdf_path}")
    print(f"PDF File Size: {os.path.getsize(pdf_path):,} bytes")

    print("\n==================================================")
    print("[DONE] All Experiments and Reports Completed Successfully!")
    print("==================================================")


def rebuild_reports_only():
    """
    Rebuild reports/final_report.md and reports/final_report.pdf from existing
    verified experiment CSVs, deterministic OLS diagnostics, and reports/notebooklm_summary.md
    without altering existing experiment CSV files.
    """
    print("==================================================")
    print("[START] Rebuilding Official Markdown & PDF Reports")
    print("==================================================")

    student_fs_path = os.path.join(RESULTS_DIR, "student_feature_selection_comparison.csv")
    flood_fs_path = os.path.join(RESULTS_DIR, "flood_feature_selection_comparison.csv")
    if not os.path.exists(student_fs_path) or not os.path.exists(flood_fs_path):
        raise FileNotFoundError(
            "Saved experiment CSVs not found in reports/results/. Run full scripts/run_experiments.py first."
        )

    student_fs_df = pd.read_csv(student_fs_path)
    flood_fs_df = pd.read_csv(flood_fs_path)

    df_student = load_student_performance()
    student_artifacts = prepare_dataset(
        df=df_student,
        target_col="Exam_Score",
        test_size=0.20,
        random_state=42,
    )
    ols_student = OLSPredictionModel()
    ols_student.fit(student_artifacts.X_train_proc, student_artifacts.y_train)
    diagnostics = diagnose_regression_assumptions(
        ols_results=ols_student.ols_results,
        X_train=student_artifacts.X_train_proc,
        y_train=student_artifacts.y_train,
    )

    report_md_path = os.path.join(REPORTS_DIR, "final_report.md")
    md_content = build_markdown_report(
        student_summary_df=student_fs_df,
        flood_summary_df=flood_fs_df,
        diagnostics=diagnostics,
        output_path=report_md_path,
    )
    print(f"Saved final Markdown report to: {report_md_path}")

    pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")
    generate_pdf_report(md_content, output_pdf_path=pdf_path)
    print(f"Saved final PDF report to: {pdf_path}")
    print(f"PDF File Size: {os.path.getsize(pdf_path):,} bytes")
    print("==================================================")
    print("[DONE] Official Markdown & PDF Reports Updated!")
    print("==================================================")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        rebuild_reports_only()
    else:
        run_all_experiments()
