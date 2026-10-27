"""
Feature Selection Module.
Implements Baseline, SelectKBest, RFE, SFS, Lasso Selection, and Permutation Importance.
All selectors are strictly fitted ONLY on training data.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    RFE,
    SequentialFeatureSelector,
    SelectFromModel,
)
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


@dataclass
class FeatureSelectionResult:
    method_name: str
    selected_features: List[str]
    all_features: List[str]
    feature_scores: Optional[Dict[str, float]]
    feature_rankings: Optional[Dict[str, int]]
    execution_time_sec: float
    details: Dict[str, Any]


def select_k_best_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int = 5,
) -> FeatureSelectionResult:
    """
    Method 1 — SelectKBest (Filter Method using f_regression).
    """
    start_time = time.perf_counter()
    k = min(k, X_train.shape[1])

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_train, y_train)

    feature_names = list(X_train.columns)
    scores = selector.scores_
    pvalues = selector.pvalues_

    # Clean nan scores
    scores_clean = np.nan_to_num(scores, nan=0.0)

    # Rank features by score descending
    sorted_indices = np.argsort(scores_clean)[::-1]
    rankings = {feature_names[idx]: rank + 1 for rank, idx in enumerate(sorted_indices)}
    scores_dict = {feature_names[i]: float(scores_clean[i]) for i in range(len(feature_names))}

    support = selector.get_support()
    selected = [feature_names[i] for i in range(len(feature_names)) if support[i]]

    return FeatureSelectionResult(
        method_name="SelectKBest (Filter - f_regression)",
        selected_features=selected,
        all_features=feature_names,
        feature_scores=scores_dict,
        feature_rankings=rankings,
        execution_time_sec=time.perf_counter() - start_time,
        details={"pvalues": {feature_names[i]: float(pvalues[i]) for i in range(len(feature_names))}},
    )


def select_rfe_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features_to_select: int = 5,
) -> FeatureSelectionResult:
    """
    Method 2 — Recursive Feature Elimination (Wrapper Method using LinearRegression).
    """
    start_time = time.perf_counter()
    n_features = min(n_features_to_select, X_train.shape[1])

    base_estimator = LinearRegression()
    rfe = RFE(estimator=base_estimator, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)

    feature_names = list(X_train.columns)
    rankings = {feat: int(rank) for feat, rank in zip(feature_names, rfe.ranking_)}

    support = rfe.support_
    selected = [feature_names[i] for i in range(len(feature_names)) if support[i]]

    # Importance score derived as inverse ranking
    scores = {feat: float(1.0 / rank) for feat, rank in rankings.items()}

    return FeatureSelectionResult(
        method_name="Recursive Feature Elimination (RFE - Wrapper)",
        selected_features=selected,
        all_features=feature_names,
        feature_scores=scores,
        feature_rankings=rankings,
        execution_time_sec=time.perf_counter() - start_time,
        details={"n_features_selected": len(selected)},
    )


def select_sfs_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features_to_select: int = 5,
    direction: str = "forward",
    fast_mode: bool = True,
    max_fast_samples: int = 5000,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """
    Method 3 — Sequential Feature Selector (Wrapper Method).
    Includes fast_mode with sub-sampling for large datasets to prevent browser hang.
    """
    start_time = time.perf_counter()
    n_features = min(n_features_to_select, X_train.shape[1])

    # Workset for feature selection
    if fast_mode and len(X_train) > max_fast_samples:
        X_work, _, y_work, _ = train_test_split(
            X_train, y_train, train_size=max_fast_samples, random_state=random_state
        )
    else:
        X_work, y_work = X_train, y_train

    base_estimator = LinearRegression()
    sfs = SequentialFeatureSelector(
        estimator=base_estimator,
        n_features_to_select=n_features,
        direction=direction,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )
    sfs.fit(X_work, y_work)

    feature_names = list(X_train.columns)
    support = sfs.get_support()
    selected = [feature_names[i] for i in range(len(feature_names)) if support[i]]

    rankings = {feat: (1 if feat in selected else 2) for feat in feature_names}
    scores = {feat: (1.0 if feat in selected else 0.0) for feat in feature_names}

    return FeatureSelectionResult(
        method_name=f"Sequential Feature Selector ({direction.capitalize()} - Wrapper)",
        selected_features=selected,
        all_features=feature_names,
        feature_scores=scores,
        feature_rankings=rankings,
        execution_time_sec=time.perf_counter() - start_time,
        details={"direction": direction, "fast_mode": fast_mode, "workset_samples": len(X_work)},
    )


def select_lasso_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_k: Optional[int] = None,
    cv: int = 5,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """
    Method 4 — Lasso Feature Selection (Embedded Method using LassoCV).
    Handles zero-feature edge case gracefully with coefficient-magnitude fallback.
    """
    start_time = time.perf_counter()
    feature_names = list(X_train.columns)

    lasso_cv = LassoCV(cv=cv, random_state=random_state, max_iter=5000)
    lasso_cv.fit(X_train, y_train)

    coef = np.asarray(lasso_cv.coef_).ravel()
    abs_coef = np.abs(coef)
    scores = {feat: float(abs_c) for feat, abs_c in zip(feature_names, abs_coef)}

    # Standard SelectFromModel with threshold > 1e-5
    selector = SelectFromModel(lasso_cv, threshold=1e-5, prefit=True)
    support = selector.get_support()
    selected = [feature_names[i] for i in range(len(feature_names)) if support[i]]

    # Fallback if Lasso shrank ALL coefficients to zero
    fallback_used = False
    if not selected:
        fallback_used = True
        k_fallback = target_k if target_k is not None else min(5, len(feature_names))
        top_indices = np.argsort(abs_coef)[::-1][:k_fallback]
        selected = [feature_names[idx] for idx in top_indices]

    # Rank features by absolute coefficient
    sorted_indices = np.argsort(abs_coef)[::-1]
    rankings = {feature_names[idx]: rank + 1 for rank, idx in enumerate(sorted_indices)}

    return FeatureSelectionResult(
        method_name="Lasso Feature Selection (Embedded - LassoCV)",
        selected_features=selected,
        all_features=feature_names,
        feature_scores=scores,
        feature_rankings=rankings,
        execution_time_sec=time.perf_counter() - start_time,
        details={
            "alpha": float(lasso_cv.alpha_),
            "nonzero_count": int(np.sum(abs_coef > 1e-5)),
            "fallback_used": fallback_used,
        },
    )


def select_permutation_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int = 5,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """
    Method 5 — Permutation Importance (Model-based).
    Splits training data internally into sub-train and sub-val to compute permutation importance without test leakage.
    """
    start_time = time.perf_counter()
    k = min(k, X_train.shape[1])
    feature_names = list(X_train.columns)

    # Internal validation split to prevent leakage
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_sub_tr, y_sub_tr)

    perm = permutation_importance(
        model,
        X_sub_val,
        y_sub_val,
        scoring="neg_mean_squared_error",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )

    importances = perm.importances_mean
    importances_clean = np.nan_to_num(importances, nan=0.0)

    sorted_indices = np.argsort(importances_clean)[::-1]
    top_k_indices = sorted_indices[:k]
    selected = [feature_names[idx] for idx in top_k_indices]

    rankings = {feature_names[idx]: rank + 1 for rank, idx in enumerate(sorted_indices)}
    scores = {feature_names[i]: float(importances_clean[i]) for i in range(len(feature_names))}

    return FeatureSelectionResult(
        method_name="Permutation Importance (Model-based on Validation)",
        selected_features=selected,
        all_features=feature_names,
        feature_scores=scores,
        feature_rankings=rankings,
        execution_time_sec=time.perf_counter() - start_time,
        details={"val_samples": len(X_sub_val)},
    )


def run_feature_selection_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int = 5,
    cv_folds: int = 5,
    random_state: int = 42,
    fast_mode: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, FeatureSelectionResult], Dict[str, Any]]:
    """
    Execute all 5 feature selection methods and compare against Baseline.
    Returns: (summary_dataframe, selection_results_dict, models_dict)
    """
    from src.evaluation import compute_metrics_dict, evaluate_regression_model

    feature_names = list(X_train.columns)
    k = min(k, len(feature_names))

    methods_to_run = [
        ("Baseline (All Features)", None),
        ("SelectKBest (Filter)", lambda: select_k_best_features(X_train, y_train, k=k)),
        ("RFE (Wrapper)", lambda: select_rfe_features(X_train, y_train, n_features_to_select=k)),
        ("SFS Forward (Wrapper)", lambda: select_sfs_features(X_train, y_train, n_features_to_select=k, direction="forward", fast_mode=fast_mode, random_state=random_state)),
        ("Lasso Selection (Embedded)", lambda: select_lasso_features(X_train, y_train, target_k=k, cv=cv_folds, random_state=random_state)),
        ("Permutation Importance", lambda: select_permutation_features(X_train, y_train, k=k, random_state=random_state)),
    ]

    records = []
    results_map = {}
    models_map = {}

    for method_name, runner in methods_to_run:
        start_t = time.perf_counter()
        if runner is None:
            # Baseline uses all features
            selected_feats = feature_names
            fs_res = FeatureSelectionResult(
                method_name=method_name,
                selected_features=selected_feats,
                all_features=feature_names,
                feature_scores=None,
                feature_rankings=None,
                execution_time_sec=0.0,
                details={"feature_count": len(selected_feats)},
            )
        else:
            fs_res = runner()
            selected_feats = fs_res.selected_features

        results_map[method_name] = fs_res

        # Train a LinearRegression on the selected features
        X_tr_sub = X_train[selected_feats]
        X_te_sub = X_test[selected_feats]

        model = LinearRegression()
        model.fit(X_tr_sub, y_train)
        models_map[method_name] = model

        eval_res = evaluate_regression_model(
            model=model,
            X_train=X_tr_sub,
            y_train=y_train,
            X_test=X_te_sub,
            y_test=y_test,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        test_m = eval_res["test"]
        train_m = eval_res["train"]
        cv_m = eval_res["cv"]

        total_time = (time.perf_counter() - start_t) + fs_res.execution_time_sec

        records.append({
            "Method": method_name,
            "Features": len(selected_feats),
            "R2": test_m["R2"],
            "Adj_R2": test_m["Adj_R2"] if test_m["Adj_R2"] is not None else np.nan,
            "RMSE": test_m["RMSE"],
            "MAE": test_m["MAE"],
            "MSE": test_m["MSE"],
            "Train_R2": train_m["R2"],
            "CV_R2_Mean": cv_m.get("CV_R2_Mean"),
            "Time_Sec": round(total_time, 3),
            "Selected_Feature_List": ", ".join(selected_feats),
        })

    summary_df = pd.DataFrame(records).sort_values(by="R2", ascending=False).reset_index(drop=True)
    return summary_df, results_map, models_map
