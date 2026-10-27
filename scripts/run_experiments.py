"""
Script: run_experiments.py
Description: Execute full CRISP-DM experiments on Student Performance and Flood datasets,
generate figures, CSV results, and build final Markdown and PDF reports.
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
    PROBLEM_STUDENT,
    PROBLEM_FLOOD,
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
    print("\n[1/2] Running Experiments on Student Performance Factors...")
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

    # Run Feature Selection comparison
    print("Running 5 Feature Selection methods on Student Performance...")
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

    # Fit OLS for Prediction Interval
    ols_student = OLSPredictionModel()
    ols_student.fit(student_artifacts.X_train_proc, student_artifacts.y_train)
    student_pi_df = ols_student.predict_intervals(student_artifacts.X_test_proc, alpha=0.05)

    # Save Student Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    student_fs_path = os.path.join(RESULTS_DIR, "student_feature_selection_comparison.csv")
    student_fs_df.to_csv(student_fs_path, index=False, encoding="utf-8-sig")
    print(f"Saved Student FS comparison to: {student_fs_path}")

    X_test_design = sm.add_constant(student_artifacts.X_test_proc, has_constant="add")
    y_test_pred_student = ols_student.ols_results.predict(X_test_design)
    student_preds_df = pd.DataFrame({
        "Actual_Exam_Score": student_artifacts.y_test.values,
        "Predicted_Exam_Score": y_test_pred_student.values,
        "CI_Lower_95": student_pi_df["mean_ci_lower"].values,
        "CI_Upper_95": student_pi_df["mean_ci_upper"].values,
        "PI_Lower_95": student_pi_df["obs_ci_lower"].values,
        "PI_Upper_95": student_pi_df["obs_ci_upper"].values,
    })
    student_preds_path = os.path.join(RESULTS_DIR, "student_predictions_intervals.csv")
    student_preds_df.to_csv(student_preds_path, index=False, encoding="utf-8-sig")
    print(f"Saved Student predictions with 95% PI to: {student_preds_path}")

    # Generate Figures
    generate_static_figures(
        df=df_student,
        y_test_true=student_artifacts.y_test.values,
        y_test_pred=y_test_pred_student.values,
        fs_summary_df=student_fs_df,
        target_name="Exam_Score",
    )
    print("Generated static publication figures in reports/figures/")

    # 2. Second Research: Flood Probability Prediction
    print("\n[2/2] Running Experiments on Flood Probability Prediction...")
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

    print("Running 5 Feature Selection methods on Flood Probability...")
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
    print(f"Saved Flood FS comparison to: {flood_fs_path}")

    # 3. Build Final Report & PDF
    print("\n[3/3] Compiling Markdown and PDF Reports...")
    md_content = build_markdown_report(student_summary_df=student_fs_df, flood_summary_df=flood_fs_df)
    pdf_path = os.path.join(REPORTS_DIR, "final_report.pdf")
    generate_pdf_report(md_content, pdf_path)
    print(f"Saved final PDF report to: {pdf_path}")

    print("\n==================================================")
    print("[DONE] All Experiments Completed Successfully!")
    print("==================================================")


if __name__ == "__main__":
    run_all_experiments()
