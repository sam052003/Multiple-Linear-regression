"""
Data Loader Module.
Handles downloading, caching, and loading of built-in and custom datasets.
"""

import os
import io
from typing import Optional, Tuple
import pandas as pd
import requests

from src.config import (
    PROBLEM_50_STARTUPS,
    PROBLEM_STUDENT,
    PROBLEM_FLOOD,
    PROBLEM_METADATA,
    PROCESSED_DATA_DIR,
)
from src.data_validation import fetch_csv_safely

try:
    import streamlit as st
    cache_decorator = st.cache_data
except ImportError:
    def cache_decorator(func):
        return func


@cache_decorator
def load_50_startups() -> pd.DataFrame:
    """
    Load the classic 50 Startups dataset.
    """
    url = PROBLEM_METADATA[PROBLEM_50_STARTUPS]["source"]
    local_path = os.path.join(PROCESSED_DATA_DIR, "50_Startups.csv")

    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    try:
        df = pd.read_csv(url)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        df.to_csv(local_path, index=False)
        return df
    except Exception as e:
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        raise RuntimeError(f"無法載入 50 Startups 資料集: {e}")


@cache_decorator
def load_student_performance() -> pd.DataFrame:
    """
    Load the Student Performance Factors dataset (Main Research).
    """
    local_path = os.path.join(PROCESSED_DATA_DIR, "student_performance.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    fallback_url = PROBLEM_METADATA[PROBLEM_STUDENT]["fallback_url"]
    df = pd.read_csv(fallback_url)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(local_path, index=False)
    return df


@cache_decorator
def load_flood_prediction(sample_size: int = 30000, random_state: int = 42) -> pd.DataFrame:
    """
    Load the Flood Probability Prediction dataset (Sampled to 30k rows).
    """
    local_path = os.path.join(PROCESSED_DATA_DIR, "flood_sample_30k.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        if len(df) > sample_size:
            return df.sample(n=sample_size, random_state=random_state)
        return df

    # If missing, run prepare script logic
    from scripts.prepare_flood_data import prepare_flood_data
    return prepare_flood_data(sample_size=sample_size, random_state=random_state)


def load_dataset_by_name(problem_name: str) -> pd.DataFrame:
    """
    Dispatch function to load built-in dataset by problem identifier.
    """
    if problem_name == PROBLEM_50_STARTUPS:
        return load_50_startups()
    elif problem_name == PROBLEM_STUDENT:
        return load_student_performance()
    elif problem_name == PROBLEM_FLOOD:
        return load_flood_prediction()
    else:
        raise ValueError(f"未知之問題名稱: {problem_name}")


def load_custom_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse uploaded CSV file from Streamlit file_uploader.
    """
    if uploaded_file is None:
        return None, "未選擇檔案。"

    try:
        content = uploaded_file.getvalue()
        df = None
        for enc in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                break
            except Exception:
                continue

        if df is None or df.empty:
            return None, "檔案解析失敗或檔案為空。"

        return df, None
    except Exception as e:
        return None, f"解析上傳 CSV 時發生錯誤: {e}"


def load_custom_url(url: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load custom dataset from a public HTTPS URL with safety verification.
    """
    return fetch_csv_safely(url)
