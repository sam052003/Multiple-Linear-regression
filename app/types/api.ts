export interface DatasetItem {
  id: string;
  name: string;
  category: string;
  description: string;
  rows: number;
  features: number;
  target: string;
}

export interface MetricSet {
  MAE: number;
  MSE: number;
  RMSE: number;
  R2: number;
  Adj_R2: number;
}

export interface DatasetOverviewResponse {
  dataset_id: string;
  name: string;
  total_rows: number;
  total_features: number;
  target_column: string;
  target_mean: number;
  target_std: number;
  target_min: number;
  target_max: number;
  numeric_features: string[];
  categorical_features: string[];
  missing_values: Record<string, number>;
  correlation_with_target: Record<string, number>;
  sample_head: Array<Record<string, any>>;
  is_feature_count_valid: boolean;
  feature_count_note: string;
}

export interface TrainResponse {
  dataset_id: string;
  target_column: string;
  test_size: number;
  random_state: number;
  train_metrics: MetricSet;
  test_metrics: MetricSet;
  coefficients: Record<string, number>;
  intercept: number;
  n_features: number;
}

export interface FeatureSelectionMethodResult {
  method_name: string;
  selected_features: string[];
  num_selected: number;
  r2: number;
  adj_r2: number;
  rmse: number;
  mae: number;
  cv_r2_mean: number;
  cv_r2_std: number;
  elapsed_sec: number;
}

export interface FeatureSelectionComparisonResponse {
  dataset_id: string;
  baseline_features_count: number;
  best_method: string;
  results: FeatureSelectionMethodResult[];
}

export interface PredictionIntervalsData {
  y_true: number[];
  y_pred: number[];
  ci_lower: number[];
  ci_upper: number[];
  pi_lower: number[];
  pi_upper: number[];
}

export interface AssumptionDiagnostics {
  linearity: {
    status: string;
    correlation_residuals_fitted: number;
    interpretation: string;
  };
  independence: {
    test_name: string;
    statistic: number;
    status: string;
    interpretation: string;
    note: string;
  };
  homoscedasticity: {
    test_name: string;
    statistic: number;
    p_value: number;
    status: string;
    interpretation: string;
  };
  normality: {
    test_name: string;
    statistic: number;
    p_value: number;
    skewness: number;
    kurtosis: number;
    status: string;
    interpretation: string;
  };
  multicollinearity: {
    vif_table: Array<{ feature: string; vif: number }>;
    max_vif: number;
    status: string;
    interpretation: string;
  };
}

export interface EvaluationResponse {
  metrics: {
    train: MetricSet;
    test: MetricSet;
  };
  actual_vs_predicted: {
    actual: number[];
    predicted: number[];
  };
  residuals_analysis: {
    residuals: number[];
    fitted: number[];
  };
  prediction_intervals: PredictionIntervalsData;
  assumption_diagnostics: AssumptionDiagnostics;
  post_selection_warning: string;
}

export interface PredictionResponse {
  point_prediction: number;
  ci_lower: number;
  ci_upper: number;
  pi_lower: number;
  pi_upper: number;
  confidence_level: number;
  warning: string;
}
