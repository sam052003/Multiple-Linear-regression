"""
Tests for Data Loader and Built-in Datasets.
"""

import pytest
import pandas as pd
from src.config import PROBLEM_50_STARTUPS, PROBLEM_STUDENT, PROBLEM_FLOOD
from src.data_loader import load_dataset_by_name, load_50_startups, load_student_performance, load_flood_prediction


def test_load_50_startups():
    df = load_50_startups()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "Profit" in df.columns
    # Check expected 4 features
    expected_cols = {"R&D Spend", "Administration", "Marketing Spend", "State", "Profit"}
    assert expected_cols.issubset(set(df.columns))


def test_load_student_performance():
    df = load_student_performance()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6607
    assert "Exam_Score" in df.columns
    # Check that there are 19 original input features (20 columns total including Exam_Score)
    assert df.shape[1] == 20


def test_load_flood_prediction():
    df = load_flood_prediction(sample_size=1000)  # fast sample for test
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "FloodProbability" in df.columns
    assert "id" in df.columns
    # 22 columns total: id + 20 features + FloodProbability
    assert df.shape[1] == 22


def test_load_dataset_by_name_dispatch():
    df_startup = load_dataset_by_name(PROBLEM_50_STARTUPS)
    assert len(df_startup) == 50

    with pytest.raises(ValueError):
        load_dataset_by_name("NonExistentProblem")
