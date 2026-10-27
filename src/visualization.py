"""
Visualization Module.
Generates beautiful, modern Plotly charts for EDA, Regression Diagnostics, and Prediction Intervals.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

# Modern color palette
COLOR_PRIMARY = "#4F46E5"      # Indigo
COLOR_ACCENT = "#06B6D4"       # Cyan
COLOR_SUCCESS = "#10B981"      # Emerald
COLOR_WARNING = "#F59E0B"      # Amber
COLOR_DANGER = "#EF4444"       # Rose
COLOR_BG_CARD = "#FFFFFF"
COLOR_TEXT_MAIN = "#1E293B"

TEMPLATE = "plotly_white"


def plot_target_distribution(y: pd.Series, target_name: str) -> go.Figure:
    """Histogram with KDE-like distribution of the target variable."""
    fig = px.histogram(
        y,
        x=y.values,
        nbins=40,
        marginal="box",
        title=f"目標變數分佈 (Target Distribution): {target_name}",
        labels={"x": target_name, "count": "次數 (Frequency)"},
        color_discrete_sequence=[COLOR_PRIMARY],
        template=TEMPLATE,
    )
    fig.update_layout(
        font=dict(family="sans-serif", size=13, color=COLOR_TEXT_MAIN),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], max_cols: int = 25) -> go.Figure:
    """Correlation Heatmap for numeric columns."""
    cols_to_use = numeric_cols[:max_cols]
    corr = df[cols_to_use].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"數值特徵相關性矩陣熱圖 (Correlation Matrix Heatmap - 前 {len(cols_to_use)} 個欄位)",
        template=TEMPLATE,
    )
    fig.update_layout(
        font=dict(family="sans-serif", size=11, color=COLOR_TEXT_MAIN),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Missing value counts bar chart."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="本資料集完全無缺失值 (100% Complete)",
            showarrow=False,
            font=dict(size=16, color=COLOR_SUCCESS),
        )
        fig.update_layout(template=TEMPLATE, title="缺失值分佈 (Missing Values Overview)")
        return fig

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "欄位名稱 (Column)", "y": "缺失數量 (Missing Count)"},
        title="各欄位缺失值統計 (Missing Values Count)",
        color=missing.values,
        color_continuous_scale="Reds",
        template=TEMPLATE,
    )
    fig.update_layout(font=dict(family="sans-serif", size=12), margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted",
    target_name: str = "Target",
    max_points: int = 1500,
) -> go.Figure:
    """Actual vs Predicted scatter plot with y=x identity line."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    # Downsample if too large for smooth web rendering
    if len(y_t) > max_points:
        idx = np.random.RandomState(42).choice(len(y_t), size=max_points, replace=False)
        y_t = y_t[idx]
        y_p = y_p[idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_t,
            y=y_p,
            mode="markers",
            marker=dict(color=COLOR_PRIMARY, opacity=0.55, size=6),
            name="預測樣本 (Samples)",
        )
    )

    # Reference y = x line
    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color=COLOR_DANGER, dash="dash", width=2),
            name="理想對角線 (Ideal y = x)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=f"實際值 Actual ({target_name})",
        yaxis_title=f"預測值 Predicted ({target_name})",
        template=TEMPLATE,
        font=dict(family="sans-serif", size=13),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_residuals_vs_fitted(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    title: str = "Residuals vs. Fitted Values (Linearity & Homoscedasticity)",
    max_points: int = 1500,
) -> go.Figure:
    """Residuals vs Fitted plot with zero reference line."""
    yp = np.asarray(y_pred).ravel()
    res = np.asarray(residuals).ravel()

    if len(yp) > max_points:
        idx = np.random.RandomState(42).choice(len(yp), size=max_points, replace=False)
        yp = yp[idx]
        res = res[idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=yp,
            y=res,
            mode="markers",
            marker=dict(color=COLOR_ACCENT, opacity=0.55, size=6),
            name="殘差 (Residuals)",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_DANGER, annotation_text="Zero Residual")

    fig.update_layout(
        title=title,
        xaxis_title="擬合值 / 預測值 (Fitted Values)",
        yaxis_title="殘差 (Residuals = Actual - Predicted)",
        template=TEMPLATE,
        font=dict(family="sans-serif", size=13),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_residuals_qq(residuals: np.ndarray, title: str = "Normal Q-Q Plot") -> go.Figure:
    """Quantile-Quantile (Q-Q) plot to evaluate residual normality."""
    res_clean = np.asarray(residuals).ravel()
    res_clean = res_clean[~np.isnan(res_clean)]

    # Standardize
    std_res = (res_clean - np.mean(res_clean)) / (np.std(res_clean) + 1e-9)
    osm, osr = stats.probplot(std_res, dist="norm", fit=False)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(color=COLOR_PRIMARY, size=5, opacity=0.6),
            name="樣本分位數 (Sample Quantiles)",
        )
    )

    # 45-degree reference line
    q_min, q_max = float(np.min(osm)), float(np.max(osm))
    fig.add_trace(
        go.Scatter(
            x=[q_min, q_max],
            y=[q_min, q_max],
            mode="lines",
            line=dict(color=COLOR_DANGER, dash="dash", width=2),
            name="常態分佈理論線 (Theoretical Line)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="理論常態分位數 (Theoretical Quantiles)",
        yaxis_title="標準化殘差分位數 (Standardized Residuals)",
        template=TEMPLATE,
        font=dict(family="sans-serif", size=13),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_prediction_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    sample_limit: int = 50,
    title: str = "95% Prediction Interval for Observations",
) -> go.Figure:
    """
    Plot predictions with 95% Prediction Interval band against actual observations.
    Sorted by predicted value for clean, readable presentation.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    lb = np.asarray(lower_bound).ravel()
    ub = np.asarray(upper_bound).ravel()

    # Limit to sample_limit points sorted by predicted value
    n = min(len(yt), sample_limit)
    sort_idx = np.argsort(yp)[:n]

    x_axis = np.arange(n)
    yt_s = yt[sort_idx]
    yp_s = yp[sort_idx]
    lb_s = lb[sort_idx]
    ub_s = ub[sort_idx]

    fig = go.Figure()

    # Shaded band for 95% PI
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_axis, x_axis[::-1]]),
            y=np.concatenate([ub_s, lb_s[::-1]]),
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% 預測區間 (95% Prediction Interval)",
        )
    )

    # Upper bound line
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=ub_s,
            mode="lines",
            line=dict(color="rgba(79, 70, 229, 0.4)", dash="dot", width=1.5),
            name="預測上限 (Upper Bound)",
        )
    )

    # Lower bound line
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=lb_s,
            mode="lines",
            line=dict(color="rgba(79, 70, 229, 0.4)", dash="dot", width=1.5),
            name="預測下限 (Lower Bound)",
        )
    )

    # Predicted mean
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=yp_s,
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=2.5),
            marker=dict(size=5),
            name="預測值 (Predicted)",
        )
    )

    # Actual observations
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=yt_s,
            mode="markers",
            marker=dict(color=COLOR_DANGER, size=7, symbol="circle"),
            name="實際觀測值 (Actual)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="樣本排序索引 (Sorted Sample Index)",
        yaxis_title="目標數值 (Target Value)",
        template=TEMPLATE,
        font=dict(family="sans-serif", size=13),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance Ranking",
) -> go.Figure:
    """Horizontal bar chart for feature importances / coefficients."""
    df_feat = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(importances)})
    df_feat = df_feat.sort_values(by="Importance", ascending=True)

    fig = px.bar(
        df_feat,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
        labels={"Importance": "重要度 / 絕對係數 (Abs Weight)", "Feature": "特徵名稱"},
        color="Importance",
        color_continuous_scale="Purples",
        template=TEMPLATE,
    )
    fig.update_layout(font=dict(family="sans-serif", size=12), margin=dict(l=100, r=40, t=60, b=40))
    return fig


def plot_feature_selection_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing metrics across feature selection methods."""
    fig = go.Figure()

    # R2 bar
    if "R2" in comparison_df.columns:
        fig.add_trace(
            go.Bar(
                x=comparison_df["Method"],
                y=comparison_df["R2"],
                name="R² (決定係數)",
                marker_color=COLOR_PRIMARY,
            )
        )

    # Adj R2 bar
    if "Adj_R2" in comparison_df.columns:
        fig.add_trace(
            go.Bar(
                x=comparison_df["Method"],
                y=comparison_df["Adj_R2"],
                name="Adjusted R²",
                marker_color=COLOR_ACCENT,
            )
        )

    fig.update_layout(
        barmode="group",
        title="特徵選擇方法評估指標比較 (Feature Selection Comparison)",
        xaxis_title="特徵選擇方法 (Method)",
        yaxis_title="分數 (Score)",
        template=TEMPLATE,
        font=dict(family="sans-serif", size=13),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
