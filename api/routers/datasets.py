"""
Datasets Router.
Handles querying built-in datasets, overview statistics, and safe custom CSV ingestion.
"""

from typing import List, Optional
import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from api.schemas import (
    DatasetItem,
    DatasetOverviewResponse,
    CustomDatasetUrlRequest,
)
from api.services.ml_service import MLService
from src.data_validation import fetch_csv_safely

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=List[DatasetItem])
def list_datasets():
    """List all supported built-in regression problems."""
    return MLService.get_supported_datasets()


@router.get("/{dataset_id}/overview", response_model=DatasetOverviewResponse)
def get_dataset_overview(
    dataset_id: str,
    target_column: Optional[str] = None,
):
    """Get complete overview, EDA summary, and column metadata for a dataset."""
    try:
        overview = MLService.get_dataset_overview(dataset_id, target_column=target_column)
        return overview
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"資料集解析失敗: {e}")


@router.post("/custom-url", response_model=DatasetOverviewResponse)
def load_custom_url_dataset(req: CustomDatasetUrlRequest):
    """Safely fetch and inspect a custom CSV dataset from a verified public HTTPS URL."""
    df, err = fetch_csv_safely(req.url)
    if err or df is None:
        raise HTTPException(status_code=400, detail=err or "無法自該 URL 讀取有效 CSV")

    target = req.target_column or df.columns[-1]
    excluded = req.excluded_columns if req.excluded_columns is not None else []

    all_cols = list(df.columns)
    avail_features = [c for c in all_cols if c != target and c not in excluded]

    from src.data_validation import validate_feature_count, generate_data_quality_report
    is_compliant, feat_msg = validate_feature_count(len(avail_features))
    quality = generate_data_quality_report(df, target_col=target)

    num_stats = {}
    for c in quality["numeric_cols"]:
        s = df[c].describe()
        num_stats[c] = {
            "mean": float(s["mean"]),
            "std": float(s["std"]) if not np.isnan(s["std"]) else 0.0,
            "min": float(s["min"]),
            "25%": float(s["25%"]),
            "50%": float(s["50%"]),
            "75%": float(s["75%"]),
            "max": float(s["max"]),
        }

    return {
        "id": "custom_url",
        "title": "Custom Dataset (URL)",
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": all_cols,
        "numeric_columns": quality["numeric_cols"],
        "categorical_columns": quality["categorical_cols"],
        "target_column": target,
        "excluded_columns": excluded,
        "available_features": avail_features,
        "is_compliant_14_39": is_compliant,
        "compliance_message": feat_msg,
        "preview_head": df.head(10).to_dict(orient="records"),
        "quality_summary": quality,
        "descriptive_stats": num_stats,
    }


@router.post("/upload", response_model=DatasetOverviewResponse)
async def upload_custom_csv(file: UploadFile = File(...)):
    """Upload a local CSV file (under 20MB) and inspect schema."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="僅接受 .csv 格式檔案")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="檔案過大，超過 20MB 上限")

    df = None
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding=enc)
            break
        except Exception:
            continue

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="CSV 檔案無法解析或內容為空")

    target = df.columns[-1]
    excluded = [c for c in df.columns if c.lower() in ["id", "index", "unnamed: 0"] and c != target]

    all_cols = list(df.columns)
    avail_features = [c for c in all_cols if c != target and c not in excluded]

    from src.data_validation import validate_feature_count, generate_data_quality_report
    import numpy as np
    is_compliant, feat_msg = validate_feature_count(len(avail_features))
    quality = generate_data_quality_report(df, target_col=target)

    num_stats = {}
    for c in quality["numeric_cols"]:
        s = df[c].describe()
        num_stats[c] = {
            "mean": float(s["mean"]),
            "std": float(s["std"]) if not np.isnan(s["std"]) else 0.0,
            "min": float(s["min"]),
            "25%": float(s["25%"]),
            "50%": float(s["50%"]),
            "75%": float(s["75%"]),
            "max": float(s["max"]),
        }

    return {
        "id": "custom_upload",
        "title": file.filename,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": all_cols,
        "numeric_columns": quality["numeric_cols"],
        "categorical_columns": quality["categorical_cols"],
        "target_column": target,
        "excluded_columns": excluded,
        "available_features": avail_features,
        "is_compliant_14_39": is_compliant,
        "compliance_message": feat_msg,
        "preview_head": df.head(10).to_dict(orient="records"),
        "quality_summary": quality,
        "descriptive_stats": num_stats,
    }
