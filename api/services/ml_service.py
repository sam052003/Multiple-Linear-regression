"""
Machine Learning Service for FastAPI Backend.
Wraps core src/ modules to provide stateless, cacheable ML operations without Streamlit dependency.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import tempfile
import numpy as np
import pandas as pd
from functools import lru_cache

from src.config import (
    PROBLEM_50_STARTUPS,
    PROBLEM_STUDENT,
    PROBLEM_FLOOD,
    PROBLEM_METADATA,
)
from src.data_loader import (
    load_50_startups,
    load_student_performance,
    load_flood_prediction,
    load_custom_url,
)
from src.data_validation import (
    validate_target_column,
    validate_feature_count,
    generate_data_quality_report,
)
from src.preprocessing import prepare_dataset, PreprocessingArtifacts
from src.modeling import train_baseline_model, RegressionModelWrapper
from src.feature_selection import run_feature_selection_comparison
from src.evaluation import evaluate_regression_model
from src.prediction_intervals import OLSPredictionModel
from src.statistical_tests import diagnose_regression_assumptions


ID_TO_PROBLEM_MAP = {
    "50_startups": PROBLEM_50_STARTUPS,
    "student_performance": PROBLEM_STUDENT,
    "flood_probability": PROBLEM_FLOOD,
    "flood_prediction": PROBLEM_FLOOD,  # Alias for backward compatibility
}

PROBLEM_TO_ID_MAP = {v: k for k, v in ID_TO_PROBLEM_MAP.items()}


class MLService:
    # Memory cache for custom uploaded/URL datasets
    _CUSTOM_DATASETS_CACHE: Dict[str, pd.DataFrame] = {}

    @classmethod
    def store_custom_dataset(cls, dataset_id: str, df: pd.DataFrame) -> None:
        """Store custom dataset in memory cache and disk tempfile."""
        cls._CUSTOM_DATASETS_CACHE[dataset_id] = df
        try:
            temp_path = os.path.join(tempfile.gettempdir(), f"{dataset_id}.csv")
            df.to_csv(temp_path, index=False)
        except Exception:
            pass

    @classmethod
    def merge_excluded_columns(cls, def_excluded: List[str], excluded_columns: Optional[List[str]]) -> List[str]:
        """
        Merge default excluded columns (like 'id' for Flood) with user-provided excluded columns.
        Prevents an empty list [] from accidentally un-excluding default excluded columns.
        """
        merged = list(def_excluded)
        if excluded_columns:
            for col in excluded_columns:
                if col not in merged:
                    merged.append(col)
        return merged

    @staticmethod
    def get_supported_datasets() -> List[Dict[str, Any]]:
        datasets = []
        canonical_ids = ["50_startups", "student_performance", "flood_probability"]
        for prob_id in canonical_ids:
            prob_name = ID_TO_PROBLEM_MAP[prob_id]
            meta = PROBLEM_METADATA[prob_name]
            datasets.append({
                "id": prob_id,
                "title": meta["title"],
                "tag": meta["tag"],
                "source": meta["source"],
                "default_target": meta["default_target"],
                "excluded_cols": meta["excluded_cols"],
                "expected_features": meta["expected_features"],
                "is_formal_study": meta["is_formal_study"],
                "description": meta["description"],
                "limitations": meta["limitations"],
            })
        return datasets

    @staticmethod
    def pick_default_target(df: pd.DataFrame, requested_target: Optional[str] = None) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
        """
        Evaluate all columns, return:
        (chosen_target, recommended_target, candidate_targets_list)
        """
        candidate_targets = []
        recommended_target = None

        # Search from last column backwards for recommended continuous numeric target
        for col in reversed(df.columns):
            if col.lower() in ["id", "index", "unnamed: 0"]:
                continue
            s = df[col]
            s_num = pd.to_numeric(s, errors="coerce")
            valid_count = int(s_num.notna().sum())
            valid_ratio = float(valid_count / len(df)) if len(df) > 0 else 0.0
            n_unique = int(s_num.dropna().nunique())
            is_num = bool(pd.api.types.is_numeric_dtype(s) or valid_ratio >= 0.8)
            is_suitable = bool(is_num and n_unique > 2 and valid_ratio >= 0.8)
            if is_suitable and recommended_target is None:
                recommended_target = col

        for col in df.columns:
            s = df[col]
            s_num = pd.to_numeric(s, errors="coerce")
            valid_count = int(s_num.notna().sum())
            valid_ratio = float(valid_count / len(df)) if len(df) > 0 else 0.0
            n_unique = int(s_num.dropna().nunique())
            is_num = bool(pd.api.types.is_numeric_dtype(s) or valid_ratio >= 0.8)
            is_suitable = bool(is_num and n_unique > 2 and valid_ratio >= 0.8)
            candidate_targets.append({
                "column": col,
                "is_numeric": is_num,
                "valid_ratio": round(valid_ratio, 3),
                "unique_values": n_unique,
                "is_suitable_for_regression": is_suitable,
            })

        if requested_target and requested_target in df.columns:
            chosen = requested_target
        elif recommended_target:
            chosen = recommended_target
        else:
            chosen = df.columns[-1]

        return chosen, recommended_target, candidate_targets

    @classmethod
    def resolve_dataset(
        cls,
        dataset_id: str,
        custom_data: Optional[List[Dict[str, Any]]] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str, List[str]]:
        """
        Retrieve DataFrame, default target, and default excluded columns.
        Prioritizes:
        1. custom_data passed in request body
        2. in-memory cache _CUSTOM_DATASETS_CACHE
        3. disk cache tempfile
        4. built-in datasets
        """
        # 1. custom_data from request body
        if custom_data is not None and len(custom_data) > 0:
            df = pd.DataFrame(custom_data)
            cls.store_custom_dataset(dataset_id, df)
            target, _, _ = cls.pick_default_target(df, target_column)
            excluded = [c for c in df.columns if c.lower() in ["id", "index", "unnamed: 0"] and c != target]
            return df, target, excluded

        # 2. in-memory cache
        if dataset_id in cls._CUSTOM_DATASETS_CACHE:
            df = cls._CUSTOM_DATASETS_CACHE[dataset_id]
            target, _, _ = cls.pick_default_target(df, target_column)
            excluded = [c for c in df.columns if c.lower() in ["id", "index", "unnamed: 0"] and c != target]
            return df, target, excluded

        # 3. disk tempfile cache
        temp_path = os.path.join(tempfile.gettempdir(), f"{dataset_id}.csv")
        if os.path.exists(temp_path):
            try:
                df = pd.read_csv(temp_path)
                cls._CUSTOM_DATASETS_CACHE[dataset_id] = df
                target, _, _ = cls.pick_default_target(df, target_column)
                excluded = [c for c in df.columns if c.lower() in ["id", "index", "unnamed: 0"] and c != target]
                return df, target, excluded
            except Exception:
                pass

        # 4. built-in datasets
        if dataset_id == "50_startups":
            df = load_50_startups()
            meta = PROBLEM_METADATA[PROBLEM_50_STARTUPS]
        elif dataset_id == "student_performance":
            df = load_student_performance()
            meta = PROBLEM_METADATA[PROBLEM_STUDENT]
        elif dataset_id in ["flood_probability", "flood_prediction"]:
            df = load_flood_prediction(sample_size=30000, random_state=42)
            meta = PROBLEM_METADATA[PROBLEM_FLOOD]
        elif dataset_id.startswith("custom_"):
            raise ValueError(
                f"找不到自訂資料集 '{dataset_id}'。由於 Vercel 雲端採用無狀態 Serverless 架構，"
                f"跨執行個體呼叫時請務必在 API Request 中附帶完整 'custom_data' 資料陣列。"
            )
        else:
            raise ValueError(f"不支援的資料集 ID: {dataset_id}")

        return df, meta["default_target"], meta["excluded_cols"]

    @classmethod
    def get_dataset_overview(
        cls,
        dataset_id: str,
        target_column: Optional[str] = None,
        excluded_columns: Optional[List[str]] = None,
        custom_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        df, def_target, def_excluded = cls.resolve_dataset(dataset_id, custom_data, target_column)
        target = target_column or def_target
        excluded = cls.merge_excluded_columns(def_excluded, excluded_columns)

        chosen_target, recommended_target, candidate_targets = cls.pick_default_target(df, target)
        target = chosen_target

        all_cols = list(df.columns)
        avail_features = [c for c in all_cols if c != target and c not in excluded]

        is_compliant, feat_msg = validate_feature_count(len(avail_features))
        quality = generate_data_quality_report(df, target_col=target)

        # Target sample accounting and validation status
        n_total_rows = int(df.shape[0])
        y_num = pd.to_numeric(df[target], errors="coerce") if target in df.columns else pd.Series(dtype=float)
        n_valid_target = int(y_num.notna().sum())
        n_dropped_target = n_total_rows - n_valid_target

        is_target_valid, target_val_msg, _ = validate_target_column(df, target)

        # Descriptive stats for numeric columns
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

        # Target statistics
        target_stats = {}
        if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            s_t = df[target].describe()
            target_stats = {
                "mean": float(s_t["mean"]),
                "std": float(s_t["std"]) if not np.isnan(s_t["std"]) else 0.0,
                "min": float(s_t["min"]),
                "max": float(s_t["max"]),
            }

        # Pearson correlation with target for numeric columns
        target_correlation = {}
        if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            for c in quality["numeric_cols"]:
                if c != target and c not in excluded:
                    val = df[c].corr(df[target])
                    if not np.isnan(val):
                        target_correlation[c] = round(float(val), 4)

        preview = df.head(10).to_dict(orient="records")

        # For custom datasets, provide full rows so frontend can retain in state for 100% data consistency
        custom_payload = None
        if dataset_id.startswith("custom_") or custom_data is not None:
            custom_payload = df.to_dict(orient="records")

        return {
            "id": dataset_id,
            "title": ID_TO_PROBLEM_MAP.get(dataset_id, "Custom Dataset"),
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
            "preview_head": preview,
            "quality_summary": quality,
            "descriptive_stats": num_stats,
            "target_correlation": target_correlation,
            "target_stats": target_stats,
            "custom_data": custom_payload,
            "n_valid_target_rows": n_valid_target,
            "n_dropped_target_rows": n_dropped_target,
            "candidate_targets": candidate_targets,
            "recommended_target": recommended_target,
            "is_target_valid": is_target_valid,
            "target_validation_message": target_val_msg,
        }

    @classmethod
    def train_model(
        cls,
        dataset_id: str,
        target_column: Optional[str] = None,
        excluded_columns: Optional[List[str]] = None,
        test_size: float = 0.20,
        random_state: int = 42,
        scale_numeric: bool = True,
        custom_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        df, def_target, def_excluded = cls.resolve_dataset(dataset_id, custom_data, target_column)
        target = target_column or def_target
        excluded = cls.merge_excluded_columns(def_excluded, excluded_columns)

        artifacts = prepare_dataset(
            df=df,
            target_col=target,
            excluded_cols=excluded,
            test_size=test_size,
            random_state=random_state,
            scale_numeric=scale_numeric,
        )

        model_wrapper = train_baseline_model(artifacts.X_train_proc, artifacts.y_train)
        coef_table = model_wrapper.get_coefficient_table()

        coef_items = [
            {
                "feature": row["Feature"],
                "coefficient": float(row["Coefficient"]),
                "abs_coefficient": float(row["Abs_Coefficient"]),
            }
            for _, row in coef_table.iterrows()
        ]

        # Evaluate model to get train_metrics and test_metrics
        eval_res = evaluate_regression_model(
            model=model_wrapper.estimator,
            X_train=artifacts.X_train_proc,
            y_train=artifacts.y_train,
            X_test=artifacts.X_test_proc,
            y_test=artifacts.y_test,
            cv_folds=5,
            random_state=random_state,
        )

        return {
            "dataset_id": dataset_id,
            "model_type": model_wrapper.model_type,
            "intercept": model_wrapper.intercept_,
            "coefficients": coef_items,
            "training_time_ms": float(model_wrapper.training_time_sec * 1000),
            "n_train_samples": int(len(artifacts.X_train_proc)),
            "n_test_samples": int(len(artifacts.X_test_proc)),
            "n_total_samples": int(artifacts.n_total_samples),
            "n_valid_target_samples": int(artifacts.n_valid_target_samples),
            "n_dropped_target_samples": int(artifacts.n_dropped_target_samples),
            "encoded_feature_names": artifacts.feature_names_out,
            "train_metrics": eval_res["train"],
            "test_metrics": eval_res["test"],
            "target_column": target,
            "n_features": len(artifacts.feature_names_out),
        }

    @classmethod
    def run_feature_selection(
        cls,
        dataset_id: str,
        target_column: Optional[str] = None,
        excluded_columns: Optional[List[str]] = None,
        k_features: int = 5,
        test_size: float = 0.20,
        random_state: int = 42,
        cv_folds: int = 5,
        fast_mode: bool = True,
        custom_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        df, def_target, def_excluded = cls.resolve_dataset(dataset_id, custom_data, target_column)
        target = target_column or def_target
        excluded = cls.merge_excluded_columns(def_excluded, excluded_columns)

        artifacts = prepare_dataset(
            df=df,
            target_col=target,
            excluded_cols=excluded,
            test_size=test_size,
            random_state=random_state,
            scale_numeric=True,
        )

        # Ensure k does not exceed available features
        actual_k = min(k_features, max(1, len(artifacts.feature_names_out) - 1))

        summary_df, results_map, _ = run_feature_selection_comparison(
            X_train=artifacts.X_train_proc,
            y_train=artifacts.y_train,
            X_test=artifacts.X_test_proc,
            y_test=artifacts.y_test,
            k=actual_k,
            cv_folds=cv_folds,
            random_state=random_state,
            fast_mode=fast_mode,
        )

        comp_items = []
        for _, r in summary_df.iterrows():
            comp_items.append({
                "method": str(r["Method"]),
                "features": int(r["Features"]),
                "r2": float(r["R2"]),
                "adj_r2": float(r["Adj_R2"]) if pd.notna(r["Adj_R2"]) else None,
                "rmse": float(r["RMSE"]),
                "mae": float(r["MAE"]),
                "mse": float(r["MSE"]),
                "train_r2": float(r["Train_R2"]),
                "cv_r2_mean": float(r["CV_R2_Mean"]) if pd.notna(r["CV_R2_Mean"]) else None,
                "time_sec": float(r["Time_Sec"]),
                "selected_features": [f.strip() for f in str(r["Selected_Feature_List"]).split(",") if f.strip()],
            })

        best_method = str(summary_df.iloc[0]["Method"])

        rankings_dict = {}
        for m_name, res in results_map.items():
            if res.feature_scores:
                rank_items = []
                for feat in res.all_features:
                    rank_items.append({
                        "feature": feat,
                        "rank": int(res.feature_rankings.get(feat, 99)),
                        "score": float(res.feature_scores.get(feat, 0.0)),
                        "is_selected": bool(feat in res.selected_features),
                    })
                rank_items.sort(key=lambda x: x["rank"])
                rankings_dict[m_name] = rank_items

        return {
            "dataset_id": dataset_id,
            "k": actual_k,
            "best_method": best_method,
            "comparison_table": comp_items,
            "method_rankings": rankings_dict,
        }

    @classmethod
    def evaluate_model_and_diagnose(
        cls,
        dataset_id: str,
        target_column: Optional[str] = None,
        excluded_columns: Optional[List[str]] = None,
        selected_features: Optional[List[str]] = None,
        test_size: float = 0.20,
        random_state: int = 42,
        cv_folds: int = 5,
        custom_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        df, def_target, def_excluded = cls.resolve_dataset(dataset_id, custom_data, target_column)
        target = target_column or def_target
        excluded = cls.merge_excluded_columns(def_excluded, excluded_columns)

        artifacts = prepare_dataset(
            df=df,
            target_col=target,
            excluded_cols=excluded,
            test_size=test_size,
            random_state=random_state,
            scale_numeric=True,
        )

        X_tr = artifacts.X_train_proc
        X_te = artifacts.X_test_proc
        if selected_features:
            feats = [f for f in selected_features if f in X_tr.columns]
            if feats:
                X_tr = X_tr[feats]
                X_te = X_te[feats]

        # Fit Scikit-learn Linear Regression
        model_wrapper = train_baseline_model(X_tr, artifacts.y_train)
        eval_res = evaluate_regression_model(
            model=model_wrapper.estimator,
            X_train=X_tr,
            y_train=artifacts.y_train,
            X_test=X_te,
            y_test=artifacts.y_test,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        # Fit Statsmodels OLS for 95% Confidence & Prediction Intervals
        ols_model = OLSPredictionModel()
        ols_model.fit(X_tr, artifacts.y_train)
        pi_df = ols_model.predict_intervals(X_te, alpha=0.05)

        # Diagnose 5 Assumptions
        diag = diagnose_regression_assumptions(ols_model.ols_results, X_tr, artifacts.y_train)

        # Downsample actual vs predicted for fast JSON response
        y_true = np.asarray(eval_res["y_test_true"]).ravel()
        y_pred = np.asarray(eval_res["y_test_pred"]).ravel()
        n_pts = min(len(y_true), 100)
        idx = np.arange(n_pts)

        act_pred_sample = [
            {"actual": float(y_true[i]), "predicted": float(y_pred[i]), "residual": float(y_true[i] - y_pred[i])}
            for i in idx
        ]

        # 95% PI sample sorted by predicted value
        sort_idx = np.argsort(y_pred)[:60]
        pi_sample = []
        for i in sort_idx:
            pi_sample.append({
                "actual": float(y_true[i]),
                "predicted": float(y_pred[i]),
                "mean_ci_lower": float(pi_df["mean_ci_lower"].iloc[i]),
                "mean_ci_upper": float(pi_df["mean_ci_upper"].iloc[i]),
                "obs_ci_lower": float(pi_df["obs_ci_lower"].iloc[i]),
                "obs_ci_upper": float(pi_df["obs_ci_upper"].iloc[i]),
            })

        # Format assumptions
        assumptions_formatted = {}
        for key, info in diag.items():
            extra = {}
            if "vif_table" in info and isinstance(info["vif_table"], pd.DataFrame):
                extra["vif_table"] = info["vif_table"].to_dict(orient="records")
            if "p_value" in info:
                extra["p_value"] = float(info["p_value"])

            assumptions_formatted[key] = {
                "diagnostic": info["diagnostic"],
                "method": info["method"],
                "metric_name": info["metric_name"],
                "value": info["value"],
                "interpretation": info["interpretation"],
                "limitations": info["limitations"],
                "extra": extra,
            }

        return {
            "train_metrics": eval_res["train"],
            "test_metrics": eval_res["test"],
            "cv_metrics": eval_res["cv"],
            "actual_vs_predicted_sample": act_pred_sample,
            "prediction_intervals_sample": pi_sample,
            "assumptions": assumptions_formatted,
        }

    @classmethod
    def predict_single(
        cls,
        dataset_id: str,
        input_features: Dict[str, Any],
        target_column: Optional[str] = None,
        excluded_columns: Optional[List[str]] = None,
        selected_features: Optional[List[str]] = None,
        test_size: float = 0.20,
        random_state: int = 42,
        custom_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        df, def_target, def_excluded = cls.resolve_dataset(dataset_id, custom_data, target_column)
        target = target_column or def_target
        excluded = cls.merge_excluded_columns(def_excluded, excluded_columns)

        artifacts = prepare_dataset(
            df=df,
            target_col=target,
            excluded_cols=excluded,
            test_size=test_size,
            random_state=random_state,
            scale_numeric=True,
        )

        # Construct 1-row DataFrame from user input
        user_df = pd.DataFrame([input_features])
        for c in artifacts.categorical_features:
            if c in user_df.columns:
                user_df[c] = user_df[c].astype(str)

        user_proc_arr = artifacts.preprocessor.transform(user_df)
        user_proc_df = pd.DataFrame(user_proc_arr, columns=artifacts.feature_names_out, index=[0])

        X_tr = artifacts.X_train_proc
        if selected_features:
            feats = [f for f in selected_features if f in X_tr.columns]
            if feats:
                X_tr = X_tr[feats]
                user_proc_df = user_proc_df[feats]

        # Model and OLS fit
        model = train_baseline_model(X_tr, artifacts.y_train)
        pred_val = float(model.predict(user_proc_df)[0])

        ols_model = OLSPredictionModel()
        ols_model.fit(X_tr, artifacts.y_train)
        pi_df = ols_model.predict_intervals(user_proc_df, alpha=0.05)

        return {
            "dataset_id": dataset_id,
            "target_name": target,
            "predicted_value": pred_val,
            "ci_lower_95": float(pi_df["mean_ci_lower"].iloc[0]),
            "ci_upper_95": float(pi_df["mean_ci_upper"].iloc[0]),
            "pi_lower_95": float(pi_df["obs_ci_lower"].iloc[0]),
            "pi_upper_95": float(pi_df["obs_ci_upper"].iloc[0]),
            "input_features_summary": input_features,
        }
