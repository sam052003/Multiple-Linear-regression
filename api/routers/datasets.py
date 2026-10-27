"""
Datasets Router.
Handles querying built-in datasets, overview statistics, and safe custom CSV ingestion.
"""

from typing import List, Optional
import io
import uuid
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from api.schemas import (
    DatasetItem,
    DatasetOverviewResponse,
    CustomDatasetUrlRequest,
)
from api.services.ml_service import MLService
from src.config import MAX_UPLOAD_SIZE_MB, MAX_CUSTOM_DATASET_ROWS
from src.data_validation import fetch_csv_safely, validate_feature_count, generate_data_quality_report

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

    custom_id = f"custom_{uuid.uuid4().hex[:8]}"
    MLService.store_custom_dataset(custom_id, df)

    overview = MLService.get_dataset_overview(
        dataset_id=custom_id,
        target_column=req.target_column,
        excluded_columns=req.excluded_columns,
    )
    return overview


@router.post("/upload", response_model=DatasetOverviewResponse)
async def upload_custom_csv(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
):
    """Upload a local CSV file (under 4.5MB for Vercel Serverless Function compatibility) and inspect schema."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="僅接受 .csv 格式檔案")

    contents = await file.read()
    max_bytes = int(MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    if len(contents) > max_bytes:
        raise HTTPException(status_code=400, detail=f"檔案過大 ({len(contents)/(1024*1024):.1f}MB)，超過 {MAX_UPLOAD_SIZE_MB}MB Vercel 限制")

    # Reject HTML masquerading as CSV
    head_snippet = contents[:1024].lower()
    if b"<!doctype html" in head_snippet or b"<html" in head_snippet or b"<head" in head_snippet:
        raise HTTPException(status_code=400, detail="上傳檔案格式為 HTML 網頁而非 CSV，請確認上傳有效的 CSV 檔案")

    df = None
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding=enc)
            break
        except Exception:
            continue

    if df is None or df.empty or df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="CSV 檔案無法解析、特徵數不足或內容為空")

    # Strictly reject datasets exceeding safe row limit rather than silent truncation
    if df.shape[0] > MAX_CUSTOM_DATASET_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"自訂資料集筆數 ({df.shape[0]} 筆) 超出 Vercel Serverless 無狀態架構安全上限 ({MAX_CUSTOM_DATASET_ROWS} 筆)。"
                f"為確保全 CRISP-DM 流程資料 100% 一致與防止 Request Payload 溢出，請精簡資料集或使用內建研究案例。"
            ),
        )

    custom_id = f"custom_{uuid.uuid4().hex[:8]}"
    MLService.store_custom_dataset(custom_id, df)

    overview = MLService.get_dataset_overview(
        dataset_id=custom_id,
        target_column=target_column,
    )
    return overview
