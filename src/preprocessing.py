"""
Preprocessing Module.
Builds robust, leakage-free Scikit-learn preprocessing pipelines.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class PreprocessingArtifacts:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_proc: pd.DataFrame
    X_test_proc: pd.DataFrame
    feature_names_out: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    preprocessor: ColumnTransformer
    test_size: float
    random_state: int
    n_total_samples: int = 0
    n_valid_target_samples: int = 0
    n_dropped_target_samples: int = 0


def prepare_dataset(
    df: pd.DataFrame,
    target_col: str,
    excluded_cols: Optional[List[str]] = None,
    selected_feature_cols: Optional[List[str]] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    scale_numeric: bool = True,
) -> PreprocessingArtifacts:
    """
    Execute data preparation and train/test split strictly preventing data leakage.
    All transformers are fitted ONLY on training data.
    """
    if target_col not in df.columns:
        raise ValueError(f"模型訓練失敗：資料集中找不到目標變數欄位 '{target_col}'。")

    n_total = len(df)
    if n_total == 0:
        raise ValueError("模型訓練失敗：提供之資料集筆數為 0，無法進行分析。")

    if excluded_cols is None:
        excluded_cols = []

    # Drop target and excluded columns
    drop_cols = set([target_col] + excluded_cols)
    available_cols = [c for c in df.columns if c not in drop_cols]

    if selected_feature_cols is not None:
        feature_cols = [c for c in selected_feature_cols if c in available_cols]
    else:
        feature_cols = available_cols

    if not feature_cols:
        raise ValueError("模型訓練失敗：沒有可用的輸入特徵欄位，無法建立多元線性回歸模型。")

    # Analyze and validate target column
    y_raw = df[target_col]
    y_num = pd.to_numeric(y_raw, errors="coerce")
    n_valid = int(y_num.notna().sum())
    n_dropped = n_total - n_valid

    # Check 1: 0 valid samples
    if n_valid == 0:
        raise ValueError(
            f"模型訓練失敗：目前選取的目標欄位無法轉換為有效數值，{n_total} 筆資料中有 0 筆可用。請選擇適合回歸分析的數值欄位。"
        )

    # Check 2: Categorical text masquerading as target or insufficient numeric ratio
    if not pd.api.types.is_numeric_dtype(y_raw):
        valid_ratio = n_valid / n_total
        if valid_ratio < 0.8:
            raise ValueError(
                f"模型訓練失敗：此欄位屬於類別型目標變數，不適合目前的多元線性回歸模型。請選擇連續型數值目標欄位。（有效數值僅佔 {n_valid}/{n_total} 筆）"
            )

    # Filter to valid target rows
    valid_idx = y_num.notna()
    X = df.loc[valid_idx, feature_cols].copy()
    y = y_num.loc[valid_idx].astype(float).copy()

    # Check 3: Variance check on valid target values
    unique_targets = y.nunique()
    if unique_targets <= 1:
        raise ValueError(f"模型訓練失敗：目標欄位 '{target_col}' 數值皆為常數（無變異性），無法建立回歸模型。")

    if unique_targets == 2:
        raise ValueError(
            f"模型訓練失敗：此欄位屬於類別型目標變數（僅有 2 個唯一值，疑似二元分類標籤），不適合目前的多元線性回歸模型。請選擇連續型數值目標欄位。"
        )

    # Check 4: Minimum sample size
    if n_valid < 10:
        raise ValueError(
            f"模型訓練失敗：有效目標變數筆數過少（僅 {n_valid} 筆可用，共 {n_total} 筆），排除無效 Target 後無法滿足回歸模型切分與訓練需求（至少需要 10 筆）。"
        )

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # 1. Train/Test Split (MUST BE FIRST)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Build Pipeline Transformers
    transformers = []

    if numeric_features:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        # Cast categorical features to string to handle mixed types cleanly
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # 3. Fit ONLY on X_train, transform both
    # Prepare string representations for categorical training data
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    for col in categorical_features:
        X_train_clean[col] = X_train_clean[col].astype(str)
        X_test_clean[col] = X_test_clean[col].astype(str)

    X_train_arr = preprocessor.fit_transform(X_train_clean)
    X_test_arr = preprocessor.transform(X_test_clean)

    # 4. Extract feature names
    feature_names_out = []
    if numeric_features:
        feature_names_out.extend(numeric_features)
    if categorical_features:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_encoded_names = encoder.get_feature_names_out(categorical_features).tolist()
        feature_names_out.extend(cat_encoded_names)

    # Convert to clean pandas DataFrames
    X_train_proc = pd.DataFrame(X_train_arr, columns=feature_names_out, index=X_train.index)
    X_test_proc = pd.DataFrame(X_test_arr, columns=feature_names_out, index=X_test.index)

    return PreprocessingArtifacts(
        X_train_raw=X_train,
        X_test_raw=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_proc=X_train_proc,
        X_test_proc=X_test_proc,
        feature_names_out=feature_names_out,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        preprocessor=preprocessor,
        test_size=test_size,
        random_state=random_state,
        n_total_samples=n_total,
        n_valid_target_samples=n_valid,
        n_dropped_target_samples=n_dropped,
    )
