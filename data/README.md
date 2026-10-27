# Datasets for Multiple Linear Regression Dashboard

This directory contains processed samples and metadata for the datasets used in this dashboard.

---

## 1. 50 Startups — Profit Prediction (Default / Teaching Example)
- **Source**: [Avik-Jain 100-Days-Of-ML-Code](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)
- **Samples**: 50
- **Target**: `Profit` (USD)
- **Features (4)**:
  - `R&D Spend`: Research & Development spending (Numeric)
  - `Administration`: Administrative spending (Numeric)
  - `Marketing Spend`: Marketing & Advertising spending (Numeric)
  - `State`: Company location (Categorical: New York, California, Florida)
- **Purpose**: Classic textbook MLR dataset demonstrating numerical + categorical features, one-hot encoding, and fast prototyping.
- **Notice**: Contains 4 features and 50 rows; clearly labeled as an introductory teaching example, not the formal 14-39 features assignment study.

---

## 2. Student Performance Factors — Exam Score Prediction (Main Research)
- **Source**: Kaggle ([lainguyn123/student-performance-factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors))
- **File**: `data/processed/student_performance.csv`
- **Samples**: 6,607
- **Target**: `Exam_Score` (Continuous numeric, 0-100)
- **Features (19 original features)**:
  - Numeric (6): `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`
  - Categorical (13): `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`
- **Purpose**: Meets the 14-39 input features requirement. Examines educational, behavioral, and socio-economic factors influencing student outcomes.

---

## 3. Flood Probability Prediction (Second Research / Synthetic Scale)
- **Source**: Kaggle Playground Series Season 4 Episode 5 ([playground-series-s4e5](https://www.kaggle.com/competitions/playground-series-s4e5))
- **File**: `data/processed/flood_sample_30k.csv`
- **Samples**: 30,000 (Randomly sampled with fixed seed `random_state=42` from the >1M row raw training dataset)
- **Target**: `FloodProbability` (Continuous 0.0 - 1.0)
- **Features (20 numeric features)**:
  - `MonsoonIntensity`, `TopographyDrainage`, `RiverManagement`, `Deforestation`, `Urbanization`, `ClimateChange`, `DamsQuality`, `Siltation`, `AgriculturalPractices`, `Encroachments`, `IneffectiveDisasterPreparedness`, `DrainageSystems`, `CoastalVulnerability`, `Landslides`, `Watersheds`, `DeterioratingInfrastructure`, `PopulationScore`, `WetlandLoss`, `InadequatePlanning`, `PoliticalFactors`
- **Special Notes**:
  - `id` column is an identifier only and is strictly excluded from modeling.
  - This is a synthetic dataset generated for the Kaggle Playground competition.
  - Linear regression predictions may mathematically exceed [0, 1]; this model limitation is explicitly disclosed.
