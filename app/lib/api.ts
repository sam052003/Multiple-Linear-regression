import {
  DatasetItem,
  DatasetOverviewResponse,
  TrainResponse,
  FeatureSelectionResponse,
  EvaluationResponse,
  PredictResponse,
} from '../types/api';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

export async function fetchHealth(): Promise<{ status: string; version: string; message: string }> {
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed with status ${res.status}`);
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetItem[]> {
  const res = await fetch(`${BASE_URL}/api/datasets`);
  if (!res.ok) throw new Error(`Failed to fetch datasets: ${res.statusText}`);
  return res.json();
}

export async function fetchDatasetOverview(
  datasetId: string,
  targetColumn?: string
): Promise<DatasetOverviewResponse> {
  const query = targetColumn ? `?target_column=${encodeURIComponent(targetColumn)}` : '';
  const res = await fetch(`${BASE_URL}/api/datasets/${datasetId}/overview${query}`);
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Failed to fetch dataset overview');
  }
  return res.json();
}

export async function submitCustomDatasetUrl(
  url: string,
  targetColumn?: string,
  excludedColumns: string[] = []
): Promise<DatasetOverviewResponse> {
  const res = await fetch(`${BASE_URL}/api/datasets/custom-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url,
      target_column: targetColumn || undefined,
      excluded_columns: excludedColumns,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Custom URL validation failed');
  }
  return res.json();
}

export async function uploadCustomCsv(
  file: File,
  targetColumn?: string
): Promise<DatasetOverviewResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (targetColumn) {
    formData.append('target_column', targetColumn);
  }

  const res = await fetch(`${BASE_URL}/api/datasets/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'CSV upload failed');
  }
  return res.json();
}

export async function trainBaselineModel(
  datasetId: string,
  targetColumn?: string,
  testSize: number = 0.2,
  randomState: number = 42,
  customData?: Array<Record<string, any>> | null
): Promise<TrainResponse> {
  const res = await fetch(`${BASE_URL}/api/analysis/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: targetColumn || undefined,
      test_size: testSize,
      random_state: randomState,
      scale_numeric: true,
      custom_data: customData || undefined,
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
  fastMode: boolean = true,
  customData?: Array<Record<string, any>> | null
): Promise<FeatureSelectionResponse> {
  const res = await fetch(`${BASE_URL}/api/analysis/feature-selection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: targetColumn || undefined,
      k_features: kFeatures,
      fast_mode: fastMode,
      custom_data: customData || undefined,
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
  randomState: number = 42,
  customData?: Array<Record<string, any>> | null
): Promise<EvaluationResponse> {
  const res = await fetch(`${BASE_URL}/api/analysis/evaluation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: targetColumn || undefined,
      selected_features: selectedFeatures || undefined,
      test_size: testSize,
      random_state: randomState,
      custom_data: customData || undefined,
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
  targetColumn?: string,
  selectedFeatures?: string[],
  customData?: Array<Record<string, any>> | null
): Promise<PredictResponse> {
  const res = await fetch(`${BASE_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      input_features: features,
      target_column: targetColumn || undefined,
      selected_features: selectedFeatures || undefined,
      custom_data: customData || undefined,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || 'Prediction failed');
  }
  return res.json();
}
