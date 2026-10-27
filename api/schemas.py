"""
Pydantic Schemas for FastAPI Endpoints.
Defines strict Request and Response models for the Multiple Linear Regression API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# -------------------- Health & Common --------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    message: str = "Multiple Linear Regression API is running"


class ErrorResponse(BaseModel):
    detail: str


# -------------------- Datasets --------------------
class DatasetItem(BaseModel):
    id: str
    title: str
    tag: str
    source: str
    default_target: str
    excluded_cols: List[str]
    expected_features: int
    is_formal_study: bool
    description: str
    limitations: str


class DatasetOverviewResponse(BaseModel):
    id: str
    title: str
    n_rows: int
    n_cols: int
    columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    target_column: str
    excluded_columns: List[str]
    available_features: List[str]
    is_compliant_14_39: bool
    compliance_message: str
    preview_head: List[Dict[str, Any]]
    quality_summary: Dict[str, Any]
    descriptive_stats: Dict[str, Dict[str, float]]
    target_correlation: Dict[str, float] = Field(default_factory=dict)
    target_stats: Dict[str, float] = Field(default_factory=dict)
    custom_data: Optional[List[Dict[str, Any]]] = None
    n_valid_target_rows: Optional[int] = None
    n_dropped_target_rows: Optional[int] = None
    candidate_targets: Optional[List[Dict[str, Any]]] = None
    recommended_target: Optional[str] = None
    is_target_valid: Optional[bool] = True
    target_validation_message: Optional[str] = None


class DatasetOverviewRequest(BaseModel):
    target_column: Optional[str] = None
    excluded_columns: Optional[List[str]] = None
    custom_data: Optional[List[Dict[str, Any]]] = None


class CustomDatasetUrlRequest(BaseModel):
    url: str
    target_column: Optional[str] = None
    excluded_columns: Optional[List[str]] = Field(default_factory=list)


# -------------------- Metric Set --------------------
class MetricSet(BaseModel):
    MAE: float
    MSE: float
    RMSE: float
    R2: float
    Adj_R2: Optional[float] = None
    n_samples: int
    n_features: int


# -------------------- Analysis & Training --------------------
class TrainRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    excluded_columns: Optional[List[str]] = Field(default_factory=list)
    test_size: float = Field(default=0.20, ge=0.10, le=0.50)
    random_state: int = 42
    scale_numeric: bool = True
    custom_data: Optional[List[Dict[str, Any]]] = None  # for user-uploaded data session


class CoefficientItem(BaseModel):
    feature: str
    coefficient: float
    abs_coefficient: float


class TrainResponse(BaseModel):
    dataset_id: str
    model_type: str = "LinearRegression"
    intercept: float
    coefficients: List[CoefficientItem]
    training_time_ms: float
    n_train_samples: int
    n_test_samples: int
    encoded_feature_names: List[str]
    train_metrics: Optional[MetricSet] = None
    test_metrics: Optional[MetricSet] = None
    target_column: Optional[str] = None
    n_features: Optional[int] = None
    n_total_samples: Optional[int] = None
    n_valid_target_samples: Optional[int] = None
    n_dropped_target_samples: Optional[int] = None


# -------------------- Feature Selection --------------------
class FeatureSelectionRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    excluded_columns: Optional[List[str]] = Field(default_factory=list)
    k_features: int = Field(default=5, ge=1)
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = Field(default=5, ge=2, le=10)
    fast_mode: bool = True
    custom_data: Optional[List[Dict[str, Any]]] = None


class MethodComparisonItem(BaseModel):
    method: str
    features: int
    r2: float
    adj_r2: Optional[float] = None
    rmse: float
    mae: float
    mse: float
    train_r2: float
    cv_r2_mean: Optional[float] = None
    time_sec: float
    selected_features: List[str]


class FeatureRankItem(BaseModel):
    feature: str
    rank: int
    score: float
    is_selected: bool


class FeatureSelectionResponse(BaseModel):
    dataset_id: str
    k: int
    best_method: str
    comparison_table: List[MethodComparisonItem]
    method_rankings: Dict[str, List[FeatureRankItem]]


# -------------------- Evaluation & Assumptions --------------------
class EvaluationRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    excluded_columns: Optional[List[str]] = Field(default_factory=list)
    selected_features: Optional[List[str]] = None
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5
    custom_data: Optional[List[Dict[str, Any]]] = None


class AssumptionDetail(BaseModel):
    diagnostic: str
    method: str
    metric_name: str
    value: Any
    interpretation: str
    limitations: str
    extra: Optional[Dict[str, Any]] = None


class EvaluationResponse(BaseModel):
    train_metrics: MetricSet
    test_metrics: MetricSet
    cv_metrics: Dict[str, Any]
    actual_vs_predicted_sample: List[Dict[str, float]]  # [actual, predicted, residual]
    prediction_intervals_sample: List[Dict[str, float]]  # [actual, predicted, mean_ci_lower, mean_ci_upper, obs_ci_lower, obs_ci_upper]
    assumptions: Dict[str, AssumptionDetail]


# -------------------- Interactive Prediction --------------------
class PredictRequest(BaseModel):
    dataset_id: str
    input_features: Dict[str, Any]
    target_column: Optional[str] = None
    excluded_columns: Optional[List[str]] = Field(default_factory=list)
    selected_features: Optional[List[str]] = None
    test_size: float = 0.20
    random_state: int = 42
    custom_data: Optional[List[Dict[str, Any]]] = None


class PredictResponse(BaseModel):
    dataset_id: str
    target_name: str
    predicted_value: float
    ci_lower_95: float
    ci_upper_95: float
    pi_lower_95: float
    pi_upper_95: float
    input_features_summary: Dict[str, Any]
