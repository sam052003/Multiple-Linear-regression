/**
 * TypeScript Interfaces strictly matching FastAPI Backend Pydantic Schemas.
 */

export interface DatasetItem {
  id: string;
  title: string;
  tag: string;
  source: string;
  default_target: string;
  excluded_cols: string[];
  expected_features: number;
  is_formal_study: boolean;
  description: string;
  limitations: string;
}

export interface MetricSet {
  MAE: number;
  MSE: number;
  RMSE: number;
  R2: number;
  Adj_R2?: number | null;
  n_samples: number;
  n_features: number;
}

export interface TargetCandidate {
  column: string;
  is_numeric: boolean;
  valid_ratio: number;
  unique_values: number;
  is_suitable_for_regression: boolean;
}

export interface DatasetOverviewResponse {
  id: string;
  title: string;
  n_rows: number;
  n_cols: number;
  columns: string[];
  numeric_columns: string[];
  categorical_columns: string[];
  target_column: string;
  excluded_columns: string[];
  available_features: string[];
  is_compliant_14_39: boolean;
  compliance_message: string;
  preview_head: Array<Record<string, any>>;
  quality_summary: Record<string, any>;
  descriptive_stats: Record<string, Record<string, number>>;
  target_correlation: Record<string, number>;
  target_stats: Record<string, number>;
  custom_data?: Array<Record<string, any>> | null;
  n_valid_target_rows?: number | null;
  n_dropped_target_rows?: number | null;
  candidate_targets?: TargetCandidate[] | null;
  recommended_target?: string | null;
  is_target_valid?: boolean;
  target_validation_message?: string | null;
}

export interface CoefficientItem {
  feature: string;
  coefficient: number;
  abs_coefficient: number;
}

export interface TrainResponse {
  dataset_id: string;
  model_type: string;
  intercept: number;
  coefficients: CoefficientItem[];
  training_time_ms: number;
  n_train_samples: number;
  n_test_samples: number;
  encoded_feature_names: string[];
  train_metrics?: MetricSet | null;
  test_metrics?: MetricSet | null;
  target_column?: string | null;
  n_features?: number | null;
  n_total_samples?: number | null;
  n_valid_target_samples?: number | null;
  n_dropped_target_samples?: number | null;
}

export interface MethodComparisonItem {
  method: string;
  features: number;
  r2: number;
  adj_r2?: number | null;
  rmse: number;
  mae: number;
  mse: number;
  train_r2: number;
  cv_r2_mean?: number | null;
  time_sec: number;
  selected_features: string[];
}

export interface FeatureRankItem {
  feature: string;
  rank: number;
  score: number;
  is_selected: boolean;
}

export interface FeatureSelectionResponse {
  dataset_id: string;
  k: number;
  best_method: string;
  comparison_table: MethodComparisonItem[];
  method_rankings: Record<string, FeatureRankItem[]>;
}

export interface AssumptionDetail {
  diagnostic: string;
  method: string;
  metric_name: string;
  value: any;
  interpretation: string;
  limitations: string;
  extra?: Record<string, any> | null;
}

export interface ActualVsPredictedPoint {
  actual: number;
  predicted: number;
  residual: number;
}

export interface PredictionIntervalSamplePoint {
  actual: number;
  predicted: number;
  mean_ci_lower: number;
  mean_ci_upper: number;
  obs_ci_lower: number;
  obs_ci_upper: number;
}

export interface EvaluationResponse {
  train_metrics: MetricSet;
  test_metrics: MetricSet;
  cv_metrics: Record<string, any>;
  actual_vs_predicted_sample: ActualVsPredictedPoint[];
  prediction_intervals_sample: PredictionIntervalSamplePoint[];
  assumptions: Record<string, AssumptionDetail>;
}

export interface PredictResponse {
  dataset_id: string;
  target_name: string;
  predicted_value: number;
  ci_lower_95: number;
  ci_upper_95: number;
  pi_lower_95: number;
  pi_upper_95: number;
  input_features_summary: Record<string, any>;
}
