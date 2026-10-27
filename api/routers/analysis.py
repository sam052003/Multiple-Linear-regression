"""
Analysis Router.
Provides endpoints for Model Training, 5 Feature Selection Comparisons, and Evaluation/Diagnostics.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    TrainRequest,
    TrainResponse,
    FeatureSelectionRequest,
    FeatureSelectionResponse,
    EvaluationRequest,
    EvaluationResponse,
)
from api.services.ml_service import MLService

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    """Train Baseline Multiple Linear Regression on training split and return coefficients."""
    try:
        res = MLService.train_model(
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            excluded_columns=req.excluded_columns,
            test_size=req.test_size,
            random_state=req.random_state,
            scale_numeric=req.scale_numeric,
            custom_data=req.custom_data,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"模型訓練失敗: {e}")


@router.post("/feature-selection", response_model=FeatureSelectionResponse)
def run_feature_selection(req: FeatureSelectionRequest):
    """Execute and compare all 5 Feature Selection methods against Baseline."""
    try:
        res = MLService.run_feature_selection(
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            excluded_columns=req.excluded_columns,
            k_features=req.k_features,
            test_size=req.test_size,
            random_state=req.random_state,
            cv_folds=req.cv_folds,
            fast_mode=req.fast_mode,
            custom_data=req.custom_data,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"特徵選擇執行失敗: {e}")


@router.post("/evaluation", response_model=EvaluationResponse)
def evaluate_model(req: EvaluationRequest):
    """Compute test metrics, Statsmodels 95% Prediction Interval, and 5 Regression Assumptions."""
    try:
        res = MLService.evaluate_model_and_diagnose(
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            excluded_columns=req.excluded_columns,
            selected_features=req.selected_features,
            test_size=req.test_size,
            random_state=req.random_state,
            cv_folds=req.cv_folds,
            custom_data=req.custom_data,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"模型評估與診斷失敗: {e}")
