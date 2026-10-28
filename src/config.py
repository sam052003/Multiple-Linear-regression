"""
Configuration and constants for Multiple Linear Regression Dashboard.
"""

import os

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
RESULTS_DIR = os.path.join(REPORTS_DIR, "results")

# Problem Identifiers
PROBLEM_50_STARTUPS = "50 Startups — Profit Prediction"
PROBLEM_STUDENT = "Student Performance Factors — Exam Score Prediction"
PROBLEM_FLOOD = "Flood Probability Prediction"

PROBLEM_CHOICES = [
    PROBLEM_50_STARTUPS,
    PROBLEM_STUDENT,
    PROBLEM_FLOOD,
]

# Problem Metadata
PROBLEM_METADATA = {
    PROBLEM_50_STARTUPS: {
        "title": "50 Startups — Profit Prediction",
        "tag": "教學展示案例 (Classic Example)",
        "source": "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv",
        "default_target": "Profit",
        "excluded_cols": [],
        "expected_features": 4,
        "is_formal_study": False,
        "description": "經典多元線性回歸教學案例。分析 50 家新創公司的研發支出 (R&D Spend)、行政管理支出 (Administration)、行銷費用 (Marketing Spend) 以及所在州別 (State) 對公司淨利 (Profit) 的影響。",
        "limitations": "樣本數僅 50 筆，評估指標（尤其是交叉驗證）容易波動，特徵數小於作業規定的 14~39 規格，僅供教學展示與快速驗證。",
    },
    PROBLEM_STUDENT: {
        "title": "Student Performance Factors — Exam Score Prediction",
        "tag": "主要研究資料集 (Main Research)",
        "source": "Kaggle (lainguyn123/student-performance-factors)",
        "local_path": os.path.join(PROCESSED_DATA_DIR, "student_performance.csv"),
        "fallback_url": "https://raw.githubusercontent.com/grugg1233/StudentPerformanceFactors/main/StudentPerformanceFactors.csv",
        "default_target": "Exam_Score",
        "excluded_cols": [],
        "expected_features": 19,
        "is_formal_study": True,
        "description": "正式學術研究主題。探討學習時數、出席率、睡眠時間、過往分數、家庭環境等 19 項原始特徵與學生考試成績 (Exam_Score) 之間的預測關聯，完全符合 14~39 輸入特徵要求。",
        "limitations": "回歸係數反映的是統計預測關聯性，非因果效應（Correlation is not causation），決策時不可逕行將係數視為介入效果。",
    },
    PROBLEM_FLOOD: {
        "title": "Flood Probability Prediction",
        "tag": "第二研究資料集 (Scale & Synthetic Study)",
        "source": "Kaggle Playground Series s4e5",
        "local_path": os.path.join(PROCESSED_DATA_DIR, "flood_sample_30k.csv"),
        "fallback_url": "https://raw.githubusercontent.com/Devaaldo/flood-probability-prediction/main/data/flood/train.csv",
        "default_target": "FloodProbability",
        "excluded_cols": ["id"],
        "expected_features": 20,
        "is_formal_study": True,
        "description": "第二研究大規模資料集。利用 20 個氣象與地理水文數值變數預測發生洪水的機率 (FloodProbability)。以固定隨機種子 (seed=42) 抽樣 30,000 筆資料快取。",
        "limitations": "此資料為 Kaggle 競賽合成資料集；多元線性回歸之數學預測值有可能微幅超出 [0, 1] 邊界；id 欄位已強制排除。",
    },
}

# Machine Learning Defaults
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_CV_FOLDS = 5

# Feature Selection Methods
FS_METHOD_BASELINE = "Baseline (All Features)"
FS_METHOD_SELECT_KBEST = "SelectKBest (Filter - f_regression)"
FS_METHOD_RFE = "Recursive Feature Elimination (RFE - Wrapper)"
FS_METHOD_SFS = "Sequential Feature Selector (SFS - Wrapper)"
FS_METHOD_LASSO = "Lasso Feature Selection (Embedded - LassoCV)"
FS_METHOD_PERMUTATION = "Permutation Importance (Model-based)"

FEATURE_SELECTION_METHODS = [
    FS_METHOD_BASELINE,
    FS_METHOD_SELECT_KBEST,
    FS_METHOD_RFE,
    FS_METHOD_SFS,
    FS_METHOD_LASSO,
    FS_METHOD_PERMUTATION,
]

# Security & Vercel Serverless Limits
MAX_UPLOAD_SIZE_MB = 4.5
MAX_CUSTOM_DATASET_ROWS = 5000
MAX_DOWNLOAD_TIMEOUT_SECONDS = 12
BLOCKED_IP_PREFIXES = ("127.", "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2", "172.3", "192.168.", "0.0.0.0", "localhost", "169.254.", "::1")
