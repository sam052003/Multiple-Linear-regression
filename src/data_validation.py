"""
Data Validation and Security Module.
Includes SSRF protection, URL safety checks, schema verification, and data quality checks.
"""

import ipaddress
import socket
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import requests

from src.config import (
    MAX_UPLOAD_SIZE_MB,
    MAX_CUSTOM_DATASET_ROWS,
    MAX_DOWNLOAD_TIMEOUT_SECONDS,
    BLOCKED_IP_PREFIXES,
)


def validate_ip_address(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> Tuple[bool, str]:
    """
    Check if an IP address belongs to private, loopback, link-local,
    reserved, multicast, carrier-grade NAT, or cloud metadata ranges.
    """
    # If IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1), unwrap to IPv4
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
        ip_obj = ip_obj.ipv4_mapped

    if ip_obj.is_private:
        return False, f"安全性阻擋: 私有網路 IP ({ip_obj})，禁止存取。"
    if ip_obj.is_loopback:
        return False, f"安全性阻擋: 本機環回 IP ({ip_obj})，禁止存取。"
    if ip_obj.is_link_local:
        return False, f"安全性阻擋: Link-Local / Metadata IP ({ip_obj})，禁止存取。"
    if ip_obj.is_reserved or ip_obj.is_multicast or ip_obj.is_unspecified:
        return False, f"安全性阻擋: 保留或非單播 IP ({ip_obj})，禁止存取。"

    # Check Cloud metadata address 169.254.169.254 specifically
    if str(ip_obj) == "169.254.169.254":
        return False, "安全性阻擋: 雲端實例 Metadata IP (169.254.169.254) 被禁止存取。"

    # Carrier-grade NAT 100.64.0.0/10
    try:
        cgnat = ipaddress.ip_network("100.64.0.0/10")
        if ip_obj in cgnat:
            return False, f"安全性阻擋: 電信級私有 NAT IP ({ip_obj})，禁止存取。"
    except Exception:
        pass

    return True, "IP 驗證通過。"


def is_safe_public_url(url: str) -> Tuple[bool, str]:
    """
    Validate that a URL is a secure, public HTTPS URL and does not target internal/private networks (SSRF prevention).
    """
    if not url or not isinstance(url, str):
        return False, "URL 不能為空。"

    url_clean = url.strip()
    try:
        parsed = urlparse(url_clean)
    except Exception as e:
        return False, f"URL 解析失敗: {e}"

    if parsed.scheme.lower() != "https":
        return False, "基於安全性考量，僅支援安全的 HTTPS 網址協定 (https://)。"

    hostname = parsed.hostname
    if not hostname:
        return False, "無效的網域名稱或主機位置。"

    # Check for direct localhost / IP prefixes
    for prefix in BLOCKED_IP_PREFIXES:
        if hostname.lower().startswith(prefix) or hostname.lower() == prefix:
            return False, f"存取受限: 禁止存取本機或內部私有網路位址 ({hostname})。"

    # Check if hostname is an IP literal directly
    try:
        ip_obj = ipaddress.ip_address(hostname)
        is_safe_ip, ip_msg = validate_ip_address(ip_obj)
        if not is_safe_ip:
            return False, ip_msg
    except ValueError:
        # Not a direct IP literal, proceed to DNS resolution
        pass

    # Check for Kaggle web HTML page vs raw file
    if "kaggle.com" in hostname.lower() and not ("raw" in parsed.path.lower() or "download" in parsed.path.lower()):
        return False, "偵測到 Kaggle 網頁網址。Kaggle 頁面為 HTML 網頁而非直接 CSV 檔案，請下載 CSV 後使用「上傳 CSV」功能。"

    # DNS Resolution Check
    try:
        resolved_ips = socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
        if not resolved_ips:
            return False, f"無法解析主機名稱: {hostname}"
        for entry in resolved_ips:
            ip_str = entry[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            is_safe_ip, ip_msg = validate_ip_address(ip_obj)
            if not is_safe_ip:
                return False, f"安全性阻擋: 主機解析至危險位址 ({ip_str})，已終止連線以防範 SSRF。"
    except socket.gaierror:
        return False, f"無法解析主機名稱: {hostname}，請確認網址是否正確。"
    except Exception as e:
        return False, f"網址安全性驗證時發生錯誤: {e}"

    return True, "URL 安全驗證通過。"


def fetch_csv_safely(url: str, max_redirects: int = 3) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch a CSV file from a verified URL with timeout, byte limit, and hop-by-hop redirect verification.
    """
    current_url = url
    session = requests.Session()

    for hop in range(max_redirects + 1):
        is_safe, msg = is_safe_public_url(current_url)
        if not is_safe:
            return None, msg

        try:
            headers = {"User-Agent": "Multiple-Linear-Regression-Dashboard/2.0"}
            response = session.get(
                current_url,
                headers=headers,
                timeout=MAX_DOWNLOAD_TIMEOUT_SECONDS,
                stream=True,
                allow_redirects=False,  # Strictly disable uncontrolled automatic redirects!
            )

            # Check redirect status codes (301, 302, 303, 307, 308)
            if response.status_code in (301, 302, 303, 307, 308):
                if hop >= max_redirects:
                    return None, f"轉址次數過多 (超過 {max_redirects} 次)，已終止連線。"

                location = response.headers.get("Location")
                if not location:
                    return None, "轉址回應中遺漏 Location 標頭。"

                from urllib.parse import urljoin
                next_url = urljoin(current_url, location)

                # Re-validate redirect destination URL & resolved IP hop-by-hop!
                is_next_safe, next_err = is_safe_public_url(next_url)
                if not is_next_safe:
                    return None, f"轉址安全性阻擋: 跳轉目標網址受限 ({next_err})"

                current_url = next_url
                continue

            if response.status_code != 200:
                return None, f"下載失敗，HTTP 狀態碼: {response.status_code}"

            # 200 OK: Verify Content-Type
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "application/xhtml+xml" in content_type:
                return None, "下載內容為 HTML 網頁而非 CSV 檔案，請確認提供的是直接的 Raw CSV 檔案連結。"

            # Check content length if available
            cl = response.headers.get("Content-Length")
            if cl and int(cl) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                return None, f"檔案過大 ({int(cl) / (1024*1024):.1f} MB)，超過 {MAX_UPLOAD_SIZE_MB} MB 限制。"

            # Stream reading with size accumulator
            max_bytes = int(MAX_UPLOAD_SIZE_MB * 1024 * 1024)
            chunks = []
            total_bytes = 0
            for chunk in response.iter_content(chunk_size=65536):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    return None, f"下載中斷: 檔案大小超過上限 {MAX_UPLOAD_SIZE_MB} MB。"
                chunks.append(chunk)

            import io
            raw_bytes = b"".join(chunks)

            # Inspect initial bytes for HTML tags (reject HTML masquerading as text/plain)
            head_snippet = raw_bytes[:1024].lower()
            if b"<!doctype html" in head_snippet or b"<html" in head_snippet or b"<head" in head_snippet:
                return None, "下載內容格式為 HTML 網頁而非有效 CSV，請確認提供的是 Raw CSV 檔案網址。"

            # Try UTF-8, then UTF-8-sig (BOM), then latin1
            df = None
            for enc in ["utf-8", "utf-8-sig", "latin1"]:
                try:
                    df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
                    break
                except Exception:
                    continue

            if df is None or df.empty or df.shape[1] < 2:
                return None, "讀取失敗: 下載之檔案無法解析為有效 CSV（特徵數不足或內容為空）。"

            # Check maximum rows constraint for Vercel stateless integrity
            if df.shape[0] > MAX_CUSTOM_DATASET_ROWS:
                return None, (
                    f"自訂資料集筆數 ({df.shape[0]} 筆) 超出 Vercel Serverless 安全上限 ({MAX_CUSTOM_DATASET_ROWS} 筆)。"
                    f"為確保全 CRISP-DM 流程資料 100% 一致與防止 Request Payload 溢出，請精簡資料集或使用內建研究案例。"
                )

            return df, None

        except requests.exceptions.Timeout:
            return None, f"連線逾時 (超過 {MAX_DOWNLOAD_TIMEOUT_SECONDS} 秒)，請確認網路或更換檔案來源。"
        except Exception as e:
            return None, f"讀取遠端 CSV 時發生未預期錯誤: {e}"

    return None, f"轉址次數過多 (超過 {max_redirects} 次)。"


def validate_target_column(df: pd.DataFrame, target_col: str) -> Tuple[bool, str, Optional[pd.Series]]:
    """
    Validate that the specified target column exists and is a valid continuous numeric target suitable for regression.
    """
    if target_col not in df.columns:
        return False, f"資料集中找不到目標變數欄位: '{target_col}'", None

    series = df[target_col]
    # Check if numeric
    if not pd.api.types.is_numeric_dtype(series):
        # Attempt conversion
        converted = pd.to_numeric(series, errors="coerce")
        valid_ratio = converted.notna().mean()
        if valid_ratio < 0.8:
            return False, f"欄位 '{target_col}' 為非數值形態，無法作為線性回歸目標 (有效數值佔比僅 {valid_ratio*100:.1f}%)。本系統嚴禁將分類文字隨意編碼後作線性回歸。", None
        series = converted

    # Check variance
    clean_series = series.dropna()
    if clean_series.nunique() <= 1:
        return False, f"目標欄位 '{target_col}' 的所有數值均相同，無變異性，無法建立回歸模型。", None

    if clean_series.nunique() <= 2:
        return False, f"目標欄位 '{target_col}' 僅有 2 個唯一值 (可能是二元分類變數)，線性回歸應使用連續型數值變數。", None

    return True, "目標變數驗證通過。", series


def validate_feature_count(original_feature_count: int) -> Tuple[bool, str]:
    """
    Check if the original input feature count satisfies the assignment requirement (14 <= p <= 39).
    Returns (is_compliant, message).
    """
    if 14 <= original_feature_count <= 39:
        return True, f"符合正式作業要求: 原始輸入特徵數為 {original_feature_count} (介於 14 與 39 之間)。"
    elif original_feature_count < 14:
        return False, f"特徵數提醒: 目前原始特徵數為 {original_feature_count} 個 (< 14 個)。本系統仍可進行回歸分析與特徵選擇，但不符合本次作業 14～39 個原始特徵的正式研究規格。"
    else:
        return False, f"特徵數提醒: 目前原始特徵數為 {original_feature_count} 個 (> 39 個)。本系統仍可進行回歸分析與特徵選擇，但不符合本次作業 14～39 個原始特徵的正式研究規格。"


def generate_data_quality_report(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report: missing values, duplicates, constant columns, high correlations.
    """
    n_rows, n_cols = df.shape
    missing_counts = df.isnull().sum()
    missing_ratios = missing_counts / n_rows

    cols_with_missing = {col: {"count": int(missing_counts[col]), "ratio": float(missing_ratios[col])}
                         for col in df.columns if missing_counts[col] > 0}

    duplicate_rows = int(df.duplicated().sum())

    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    # Potential ID columns (unique values close to row count or named id)
    potential_id_cols = []
    for col in df.columns:
        if col.lower() in ["id", "index", "unnamed: 0"] or (df[col].nunique() == n_rows and n_rows > 30):
            potential_id_cols.append(col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # High correlation pairs among numeric columns
    high_corr_pairs = []
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                val = corr_matrix.loc[c1, c2]
                if val >= 0.85:
                    high_corr_pairs.append({"col1": c1, "col2": c2, "correlation": round(float(val), 4)})

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_summary": cols_with_missing,
        "total_missing_cells": int(missing_counts.sum()),
        "duplicate_rows": duplicate_rows,
        "constant_cols": constant_cols,
        "potential_id_cols": potential_id_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "high_corr_pairs": high_corr_pairs,
    }
