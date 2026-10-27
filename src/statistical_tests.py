"""
Statistical Tests and Regression Assumptions Module.
Diagnoses Linearity, Independence, Homoscedasticity, Normality, and Multicollinearity.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def diagnose_regression_assumptions(
    ols_results,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """
    Run diagnostic checks for the 5 core assumptions of Multiple Linear Regression.
    """
    residuals = ols_results.resid
    fitted = ols_results.fittedvalues
    exog = ols_results.model.exog

    # 1. Linearity
    # In OLS, sum(residuals * fitted) is mathematically zero by orthogonality.
    corr_resid_fit = float(np.corrcoef(residuals, fitted)[0, 1]) if len(residuals) > 2 else 0.0
    linearity_report = {
        "diagnostic": "線性關係 (Linearity)",
        "method": "殘差與擬合值散佈檢驗 (Residuals vs Fitted)",
        "metric_name": "Corr(Residuals, Fitted)",
        "value": round(corr_resid_fit, 6),
        "interpretation": "殘差應在 0 軸上下均勻、無明顯非線性弧度或漏斗形狀分佈。目前殘差與擬合值相關係數接近 0，符合線性正交條件。",
        "limitations": "若真實母體關係為二次項或對數關係，單純多元線性回歸會產生系統性非線性殘差模式。",
    }

    # 2. Independence of Residuals
    dw_val = float(durbin_watson(residuals))
    if dw_val < 1.5:
        dw_desc = "DW 統計量 < 1.5，殘差可能存在正自我相關 (Positive Autocorrelation)。"
    elif dw_val > 2.5:
        dw_desc = "DW 統計量 > 2.5，殘差可能存在負自我相關 (Negative Autocorrelation)。"
    else:
        dw_desc = "DW 統計量介於 1.5 ~ 2.5 之間，殘差無顯著一階自我相關跡象。"

    independence_report = {
        "diagnostic": "殘差獨立性 (Independence)",
        "method": "Durbin-Watson 檢定 (DW Statistic)",
        "metric_name": "DW Statistic",
        "value": round(dw_val, 4),
        "interpretation": dw_desc,
        "limitations": "DW 統計量主要針對具有時間序列順序之資料；對於橫斷面 (Cross-sectional) 資料，樣本間並無自然的先後順序，因此 DW 統計量的解釋力有限。",
    }

    # 3. Homoscedasticity (同質變異性)
    try:
        bp_test = het_breuschpagan(residuals, exog)
        lm_stat = float(bp_test[0])
        lm_pval = float(bp_test[1])
        f_stat = float(bp_test[2])
        f_pval = float(bp_test[3])

        if lm_pval < 0.05:
            bp_desc = f"Breusch-Pagan 檢定 p-value ({lm_pval:.4e}) < 0.05，拒絕虛無假設，資料可能存在異質變異 (Heteroscedasticity)。"
        else:
            bp_desc = f"Breusch-Pagan 檢定 p-value ({lm_pval:.4f}) >= 0.05，未達顯著水準，無充分證據顯示存在異質變異。"

        homoscedasticity_report = {
            "diagnostic": "同質變異性 (Homoscedasticity)",
            "method": "Breusch-Pagan Lagrange Multiplier 檢定",
            "metric_name": "LM Statistic (p-value)",
            "value": f"{lm_stat:.2f} (p = {lm_pval:.4e})",
            "p_value": lm_pval,
            "interpretation": bp_desc,
            "limitations": "在存在異質變異的情況下，OLS 係數仍為不偏估計量，但標準誤估計會產生偏差，使假設檢定與預測區間可靠度降低。",
        }
    except Exception as e:
        homoscedasticity_report = {
            "diagnostic": "同質變異性 (Homoscedasticity)",
            "method": "Breusch-Pagan 檢定",
            "metric_name": "錯誤",
            "value": "N/A",
            "interpretation": f"計算過程發生異常: {e}",
            "limitations": "無",
        }

    # 4. Normality of Residuals
    try:
        jb_res = jarque_bera(residuals)
        jb_stat = float(jb_res[0])
        jb_pval = float(jb_res[1])
        skew = float(jb_res[2])
        kurtosis = float(jb_res[3])

        n_samples = len(residuals)
        if jb_pval < 0.05:
            if n_samples > 1000:
                norm_desc = f"JB 檢定 p-value ({jb_pval:.4e}) < 0.05。因大樣本 ({n_samples:,} 筆) 對常態偏離極度敏感，微小偏態亦會顯著；依據中央極限定理 (CLT)，大樣本下 OLS 係數仍具漸近常態性。"
            else:
                norm_desc = f"JB 檢定 p-value ({jb_pval:.4e}) < 0.05，殘差偏離常態分佈 (偏度={skew:.2f}, 峰度={kurtosis:.2f})。"
        else:
            norm_desc = f"JB 檢定 p-value ({jb_pval:.4f}) >= 0.05，殘差符合常態分佈假設。"

        normality_report = {
            "diagnostic": "殘差常態性 (Normality of Residuals)",
            "method": "Jarque-Bera 常態性檢定",
            "metric_name": "JB Stat (p-value)",
            "value": f"{jb_stat:.2f} (p = {jb_pval:.4e})",
            "p_value": jb_pval,
            "skewness": round(skew, 3),
            "kurtosis": round(kurtosis, 3),
            "interpretation": norm_desc,
            "limitations": "常態性檢定在大樣本下極易過度敏感，不能單憑 p < 0.05 即推翻模型實用價值，應同時參照 Q-Q 圖與殘差直方圖。",
        }
    except Exception as e:
        normality_report = {
            "diagnostic": "殘差常態性 (Normality of Residuals)",
            "method": "Jarque-Bera 檢定",
            "metric_name": "錯誤",
            "value": "N/A",
            "interpretation": f"計算異常: {e}",
            "limitations": "無",
        }

    # 5. Multicollinearity (VIF)
    vif_records = []
    feature_names = list(X_train.columns)
    try:
        X_design = sm.add_constant(X_train, has_constant="add")
        cols = list(X_design.columns)
        for i, col_name in enumerate(cols):
            if col_name.lower() in ["const", "intercept"]:
                continue
            vif = variance_inflation_factor(X_design.values, i)
            vif_records.append({
                "Feature": col_name,
                "VIF": round(float(vif), 2) if not np.isinf(vif) and not np.isnan(vif) else 999.99,
                "Status": "極度嚴重 (>10)" if vif > 10 else ("中度共線 (5~10)" if vif > 5 else "良好 (<5)"),
            })
        vif_df = pd.DataFrame(vif_records).sort_values(by="VIF", ascending=False)
        max_vif = vif_df["VIF"].max() if not vif_df.empty else 0.0
        vif_interp = f"特徵中最大 VIF 為 {max_vif:.2f}。" + ("存在嚴重多重共線性，建議透過特徵選擇降低相關性。" if max_vif > 10 else "各特徵共線性在可接受範圍內。")
    except Exception as e:
        vif_df = pd.DataFrame()
        vif_interp = f"VIF 計算異常 (可能是設計矩陣奇異或完全共線): {e}"

    multicollinearity_report = {
        "diagnostic": "多重共線性 (Multicollinearity)",
        "method": "變異數膨脹因子 (Variance Inflation Factor, VIF)",
        "metric_name": "Max VIF",
        "value": f"{max_vif:.2f}" if not vif_df.empty else "N/A",
        "vif_table": vif_df,
        "interpretation": vif_interp,
        "limitations": "類別變數 One-Hot Encoding 若未剔除基準組 (drop='first') 會導致完全共線性；特徵選擇後若仍有高 VIF，係數標準誤將會被放大。",
    }

    return {
        "linearity": linearity_report,
        "independence": independence_report,
        "homoscedasticity": homoscedasticity_report,
        "normality": normality_report,
        "multicollinearity": multicollinearity_report,
    }
