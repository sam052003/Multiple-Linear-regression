"""
Script: prepare_student_data.py
Description: Download and prepare the Student Performance Factors dataset.
Source: Kaggle (lainguyn123/student-performance-factors)
Destination: data/processed/student_performance.csv
"""

import os
import sys
import pandas as pd
import requests

STUDENT_DATA_URLS = [
    "https://raw.githubusercontent.com/grugg1233/StudentPerformanceFactors/main/StudentPerformanceFactors.csv",
    "https://raw.githubusercontent.com/Shivi2599/Student_Performance_Factors_Kaggle/main/StudentPerformanceFactors.csv",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "student_performance.csv")


def prepare_student_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"[OK] student_performance.csv already exists at {OUTPUT_FILE} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df

    print("Downloading Student Performance Factors dataset...")
    df = None
    for url in STUDENT_DATA_URLS:
        try:
            print(f"Trying {url} ...")
            df = pd.read_csv(url)
            if df.shape[0] > 1000 and "Exam_Score" in df.columns:
                print(f"Successfully downloaded from {url}!")
                break
        except Exception as e:
            print(f"Failed from {url}: {e}")

    if df is None:
        raise RuntimeError("Failed to download Student Performance Factors dataset from all sources.")

    # Validate schema
    expected_target = "Exam_Score"
    if expected_target not in df.columns:
        raise ValueError(f"Missing target column {expected_target}!")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Saved to {OUTPUT_FILE} with shape {df.shape}")
    return df


if __name__ == "__main__":
    prepare_student_data()
