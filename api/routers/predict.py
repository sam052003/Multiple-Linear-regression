"""
Predict Router.
Handles interactive single-observation predictions with 95% Confidence and Prediction Intervals.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse
from api.services.ml_service import MLService

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictResponse)
def predict_single(req: PredictRequest):
    """Predict single observation target value and compute 95% CI for Mean and 95% PI for Observation."""
    try:
        res = MLService.predict_single(
            dataset_id=req.dataset_id,
            input_features=req.input_features,
            target_column=req.target_column,
            excluded_columns=req.excluded_columns,
            selected_features=req.selected_features,
            test_size=req.test_size,
            random_state=req.random_state,
            custom_data=req.custom_data,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"單筆預測失敗: {e}")
