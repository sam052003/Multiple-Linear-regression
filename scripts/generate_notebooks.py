"""
Script: generate_notebooks.py
Description: Generates standard Jupyter Notebooks for Student Performance and Flood Prediction
matching the Dashboard's exact CRISP-DM methodology, seeds, and algorithms.
"""

import os
import json
import nbformat as nbf

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def create_student_performance_notebook():
    nb = nbf.v4.new_notebook()
    nb["cells"] = []

    # Title & Metadata
    nb["cells"].append(nbf.v4.new_markdown_cell("""# Student Performance Factors — Multiple Linear Regression Analysis
## CRISP-DM Framework: Feature Selection, Model Evaluation & 95% Prediction Interval

- **Dataset**: Kaggle Student Performance Factors (`lainguyn123/student-performance-factors`)
- **Target Variable**: `Exam_Score` (Continuous numeric, 0-100)
- **Features**: 19 original features (numerical + categorical)
- **Methodology**: Linear Regression with 5 Feature Selection methods, 95% Prediction Interval via Statsmodels OLS, and 5 Regression Assumption Diagnostics.
"""))

    # Imports
    nb["cells"].append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(".."))

from src.data_loader import load_student_performance
from src.preprocessing import prepare_dataset
from src.modeling import train_baseline_model
from src.evaluation import evaluate_regression_model
from src.feature_selection import run_feature_selection_comparison
from src.prediction_intervals import OLSPredictionModel
from src.statistical_tests import diagnose_regression_assumptions

sns.set_theme(style="whitegrid")
print("Environment and packages loaded successfully!")
"""))

    # Phase 2: Data Understanding
    nb["cells"].append(nbf.v4.new_markdown_cell("""## 1. Data Understanding (EDA)
Load dataset, examine distribution of Exam_Score, missing values, and column schemas.
"""))

    nb["cells"].append(nbf.v4.new_code_cell("""df = load_student_performance()
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()
"""))

    nb["cells"].append(nbf.v4.new_code_cell("""# Target distribution and descriptive stats
print(df["Exam_Score"].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df["Exam_Score"], kde=True, color="#4F46E5")
plt.title("Exam Score Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.show()
"""))

    # Phase 3: Data Preparation
    nb["cells"].append(nbf.v4.new_markdown_cell("""## 2. Data Preparation & Anti-Leakage Pipeline
Train/test split (80/20, seed=42) followed by ColumnTransformer (median imputation, scaling, and one-hot encoding).
"""))

    nb["cells"].append(nbf.v4.new_code_cell("""artifacts = prepare_dataset(
    df=df,
    target_col="Exam_Score",
    test_size=0.20,
    random_state=42,
)
print(f"Train samples: {len(artifacts.X_train_proc)}")
print(f"Test samples: {len(artifacts.X_test_proc)}")
print(f"Encoded feature count: {len(artifacts.feature_names_out)}")
artifacts.X_train_proc.head()
"""))

    # Phase 4: Feature Selection & Modeling
    nb["cells"].append(nbf.v4.new_markdown_cell("""## 3. Modeling & 5 Feature Selection Methods
Evaluate Baseline vs. SelectKBest, RFE, SFS, Lasso, and Permutation Importance.
"""))

    nb["cells"].append(nbf.v4.new_code_cell("""summary_df, results_map, models_map = run_feature_selection_comparison(
    X_train=artifacts.X_train_proc,
    y_train=artifacts.y_train,
    X_test=artifacts.X_test_proc,
    y_test=artifacts.y_test,
    k=5,
    cv_folds=5,
    random_state=42,
    fast_mode=False,
)
summary_df
"""))

    # Phase 5: Evaluation & 95% Prediction Interval
    nb["cells"].append(nbf.v4.new_markdown_cell("""## 4. Evaluation, 95% Prediction Interval & Diagnostics
Statsmodels OLS推論模型與預測區間。
"""))

    nb["cells"].append(nbf.v4.new_code_cell("""ols_model = OLSPredictionModel()
ols_model.fit(artifacts.X_train_proc, artifacts.y_train)
pi_df = ols_model.predict_intervals(artifacts.X_test_proc, alpha=0.05)
pi_df.head()
"""))

    nb["cells"].append(nbf.v4.new_code_cell("""# 95% Prediction Interval Plot
y_test_pred = pi_df["mean"].values
y_test_true = artifacts.y_test.values
lb = pi_df["obs_ci_lower"].values
ub = pi_df["obs_ci_upper"].values

# Sort by predicted value for clean visualization
sort_idx = np.argsort(y_test_pred)[:50]
x_axis = np.arange(50)

plt.figure(figsize=(12, 6))
plt.fill_between(x_axis, lb[sort_idx], ub[sort_idx], color="#4F46E5", alpha=0.2, label="95% Prediction Interval")
plt.plot(x_axis, y_test_pred[sort_idx], color="#4F46E5", lw=2, label="Predicted")
plt.scatter(x_axis, y_test_true[sort_idx], color="#EF4444", s=25, label="Actual Observations")
plt.title("Exam Score: 95% Prediction Interval vs Actual Observations (Test Set Sample)", fontsize=14, fontweight="bold")
plt.xlabel("Sorted Sample Index")
plt.ylabel("Exam Score")
plt.legend()
plt.tight_layout()
plt.show()
"""))

    nb["cells"].append(nbf.v4.new_code_cell("""# Regression Assumptions Diagnostics
diagnostics = diagnose_regression_assumptions(ols_model.ols_results, artifacts.X_train_proc, artifacts.y_train)
for k, v in diagnostics.items():
    print(f"=== {v['diagnostic']} ===")
    print(f"Method: {v['method']}")
    print(f"Value: {v['value']}")
    print(f"Interpretation: {v['interpretation']}")
    print()
"""))

    file_path = os.path.join(NOTEBOOKS_DIR, "student_performance_analysis.ipynb")
    with open(file_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Generated {file_path}")


def create_flood_prediction_notebook():
    nb = nbf.v4.new_notebook()
    nb["cells"] = []

    # Title & Metadata
    nb["cells"].append(nbf.v4.new_markdown_cell("""# Flood Probability Prediction — Multiple Linear Regression Analysis
## Kaggle Playground Series s4e5 Synthetic Tabular Study

- **Dataset**: Kaggle Playground Series s4e5 (`FloodProbability`)
- **Characteristics**: 20 numerical geographic/meteorological features, synthetic competition dataset
- **Sample Strategy**: 30,000 fixed-seed random sample (`random_state=42`)
- **Key Observation**: Synthetic dataset where information is uniformly distributed across all 20 features, contrasting with real-world student data.
"""))

    # Imports
    nb["cells"].append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(".."))

from src.data_loader import load_flood_prediction
from src.preprocessing import prepare_dataset
from src.feature_selection import run_feature_selection_comparison
from src.prediction_intervals import OLSPredictionModel

sns.set_theme(style="whitegrid")
print("Environment and packages loaded successfully!")
"""))

    # Data Loading
    nb["cells"].append(nbf.v4.new_code_cell("""df_flood = load_flood_prediction(sample_size=30000, random_state=42)
print(f"Flood Sample: {df_flood.shape[0]} rows, {df_flood.shape[1]} columns")
df_flood.head()
"""))

    # Preprocessing
    nb["cells"].append(nbf.v4.new_code_cell("""artifacts = prepare_dataset(
    df=df_flood,
    target_col="FloodProbability",
    excluded_cols=["id"],
    test_size=0.20,
    random_state=42,
)
print(f"Train samples: {len(artifacts.X_train_proc)}, Test samples: {len(artifacts.X_test_proc)}")
print(f"Features: {artifacts.feature_names_out}")
"""))

    # Feature Selection Comparison
    nb["cells"].append(nbf.v4.new_markdown_cell("""## Feature Selection Comparison
Compare Baseline (20 features) against K=5 selected features. Notice how R² changes from ~0.84 down to ~0.18 when reducing from 20 to 5 features, demonstrating that synthetic flood data distributes weights evenly across all attributes.
"""))

    nb["cells"].append(nbf.v4.new_code_cell("""summary_df, results_map, _ = run_feature_selection_comparison(
    X_train=artifacts.X_train_proc,
    y_train=artifacts.y_train,
    X_test=artifacts.X_test_proc,
    y_test=artifacts.y_test,
    k=5,
    cv_folds=5,
    random_state=42,
    fast_mode=True,
)
summary_df
"""))

    file_path = os.path.join(NOTEBOOKS_DIR, "flood_prediction_analysis.ipynb")
    with open(file_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Generated {file_path}")


if __name__ == "__main__":
    create_student_performance_notebook()
    create_flood_prediction_notebook()
