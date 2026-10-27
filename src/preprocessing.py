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
        raise ValueError("沒有可用的輸入特徵欄位！")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target itself is NaN (cannot train or evaluate without ground truth)
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Convert y to float
    y = pd.to_numeric(y, errors="coerce")
    valid_y_idx = y.notna()
    X = X.loc[valid_y_idx]
    y = y.loc[valid_y_idx]

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
    )
