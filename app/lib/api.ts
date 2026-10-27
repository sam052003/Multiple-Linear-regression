import {
  DatasetItem,
  DatasetOverviewResponse,
  TrainResponse,
  FeatureSelectionComparisonResponse,
  EvaluationResponse,
  PredictionResponse,
} from '../types/api';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

export async function fetchHealth(): Promise<{ status: string; version: string; modules_loaded: Record<string, boolean> }> {
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed with status ${res.status}`);
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetItem[]> {
  const res = await fetch(`${BASE_URL}/api/datasets`);
  if (!res.ok) throw new Error(`Failed to fetch datasets: ${res.statusText}`);
  return res.json();
}

export async function fetchDatasetOverview(datasetId: string): Promise<DatasetOverviewResponse> {
  const res = await fetch(`${BASE_URL}/api/datasets/${datasetId}/overview`);
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Failed to fetch dataset overview');
  }
  return res.json();
}

export async function submitCustomDatasetUrl(url: string, targetCol?: string): Promise<DatasetOverviewResponse> {
  const res = await fetch(`${BASE_URL}/api/datasets/custom`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, target_column: targetCol }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Custom URL validation failed');
  }
  return res.json();
}

export async function trainBaselineModel(
  datasetId: string,
  targetColumn?: string,
  testSize: number = 0.2,
  randomState: number = 42
): Promise<TrainResponse> {
  const res = await fetch(`${BASE_URL}/api/analysis/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: targetColumn,
      test_size: testSize,
      random_state: randomState,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Failed to train baseline model');
  }
  return res.json();
}

export async function runFeatureSelection(
  datasetId: string,
  targetColumn?: string,
  kFeatures: number = 5,
  fastMode: boolean = true
): Promise<FeatureSelectionComparisonResponse> {
  const res = await fetch(`${BASE_URL}/api/analysis/feature-selection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: targetColumn,
      k_features: kFeatures,
      fast_mode: fastMode,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Feature selection failed');
  }
  return res.json();
}

export async function runEvaluation(
  datasetId: string,
  targetColumn?: string,
  selectedFeatures?: string[],
  testSize: number = 0.2,
  randomState: number = 42
): Promise<EvaluationResponse> {
  const res = await fetch(`${BASE_URL}/api/analysis/evaluation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: targetColumn,
      selected_features: selectedFeatures,
      test_size: testSize,
      random_state: randomState,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Evaluation failed');
  }
  return res.json();
}

export async function predictSingle(
  datasetId: string,
  features: Record<string, any>,
  targetColumn?: string
): Promise<PredictionResponse> {
  const res = await fetch(`${BASE_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      features,
      target_column: targetColumn,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Prediction failed');
  }
  return res.json();
}

export async function fetchMarkdownReport(): Promise<{ report_title: string; markdown: string }> {
  const res = await fetch(`${BASE_URL}/api/reports/markdown`);
  if (!res.ok) throw new Error('Failed to load markdown report');
  return res.json();
}
