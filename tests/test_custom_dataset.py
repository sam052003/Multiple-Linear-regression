"""
Tests for Custom Dataset, SSRF Security, and Data Validation.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_validation import (
    is_safe_public_url,
    validate_target_column,
    validate_feature_count,
    generate_data_quality_report,
)


def test_ssrf_prevention_blocks_private_ips():
    # Loopback
    is_safe, msg = is_safe_public_url("http://127.0.0.1/data.csv")
    assert not is_safe
    assert "HTTPS" in msg or "私有" in msg

    # Localhost
    is_safe, msg = is_safe_public_url("https://localhost/secret.csv")
    assert not is_safe

    # Private 192.168.x.x
    is_safe, msg = is_safe_public_url("https://192.168.1.1/data.csv")
    assert not is_safe
    assert "私有" in msg or "受限" in msg

    # Kaggle HTML webpage
    is_safe, msg = is_safe_public_url("https://www.kaggle.com/datasets/user/dataset")
    assert not is_safe
    assert "Kaggle" in msg


def test_safe_public_https_url_passes():
    url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
    is_safe, msg = is_safe_public_url(url)
    assert is_safe


def test_target_column_validation():
    df_valid = pd.DataFrame({
        "feat1": [1, 2, 3, 4, 5],
        "target": [10.5, 20.1, 30.2, 40.5, 50.0],
    })
    is_valid, msg, series = validate_target_column(df_valid, "target")
    assert is_valid

    # Constant target
    df_const = pd.DataFrame({"target": [5, 5, 5, 5]})
    is_valid, msg, _ = validate_target_column(df_const, "target")
    assert not is_valid
    assert "無變異性" in msg

    # Non-numeric target
    df_text = pd.DataFrame({"target": ["dog", "cat", "bird", "fish"]})
    is_valid, msg, _ = validate_target_column(df_text, "target")
    assert not is_valid


def test_feature_count_specification_compliance():
    # 4 features: non-compliant (too few)
    compliant, msg = validate_feature_count(4)
    assert not compliant
    assert "< 14" in msg

    # 19 features: compliant
    compliant, msg = validate_feature_count(19)
    assert compliant
    assert "符合正式作業要求" in msg

    # 20 features: compliant
    compliant, msg = validate_feature_count(20)
    assert compliant

    # 45 features: non-compliant (too many)
    compliant, msg = validate_feature_count(45)
    assert not compliant
    assert "> 39" in msg


def test_data_quality_report():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "A": [1.0, np.nan, 3.0, 4.0],
        "B": ["cat", "dog", "cat", "dog"],
        "C": [10, 10, 10, 10],  # constant
    })
    report = generate_data_quality_report(df, target_col="A")
    assert report["n_rows"] == 4
    assert report["total_missing_cells"] == 1
    assert "C" in report["constant_cols"]
    assert "id" in report["potential_id_cols"]
