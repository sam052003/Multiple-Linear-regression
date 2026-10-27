'use client';

import React, { useState, useEffect } from 'react';
import {
  Calculator,
  Download,
  TrendingUp,
  Scale,
  Sliders,
} from 'lucide-react';
import { DatasetOverviewResponse, PredictResponse } from '../../types/api';
import { predictSingle } from '../../lib/api';

interface PredictionTabProps {
  overview: DatasetOverviewResponse | null;
}

export default function PredictionTab({ overview }: PredictionTabProps) {
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize form values from preview head
  useEffect(() => {
    if (overview && overview.preview_head && overview.preview_head.length > 0) {
      const firstRow = overview.preview_head[0];
      const initial: Record<string, any> = {};
      [...overview.numeric_columns, ...overview.categorical_columns].forEach((col) => {
        if (col !== overview.target_column && !overview.excluded_columns.includes(col)) {
          initial[col] = firstRow[col] !== undefined ? firstRow[col] : 0;
        }
      });
      setFormValues(initial);
      setPrediction(null);
    }
  }, [overview]);

  if (!overview) return null;

  const handleInputChange = (field: string, val: any) => {
    setFormValues((prev) => ({
      ...prev,
      [field]: val,
    }));
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      const res = await predictSingle(
        overview.id,
        formValues,
        overview.target_column,
        undefined,
        overview.custom_data
      );
      setPrediction(res);
    } catch (err: any) {
      setError(err.message || 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadPredictionCsv = () => {
    if (!prediction) return;
    const header = Object.keys(formValues).concat([
      'Point_Prediction',
      '95_CI_Lower',
      '95_CI_Upper',
      '95_PI_Lower',
      '95_PI_Upper',
    ]);
    const values = Object.values(formValues).concat([
      prediction.predicted_value,
      prediction.ci_lower_95,
      prediction.ci_upper_95,
      prediction.pi_lower_95,
      prediction.pi_upper_95,
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [header.join(','), values.join(',')].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `prediction_${overview.id}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
            CRISP-DM Stage 6: Deployment & Interactive Inference
          </span>
        </div>
        <h3 className="text-xl font-bold text-white">Interactive Prediction & Statistical Inference</h3>
        <p className="text-xs text-slate-400">
          Enter custom feature values to compute real-time point predictions, 95% Confidence Interval for Mean, and 95% Prediction Interval for Observations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form Inputs (2 cols) */}
        <div className="lg:col-span-2 rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-6 shadow-lg">
          <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="h-4 w-4 text-indigo-400" />
              Input Feature Parameters
            </h4>
            <span className="text-xs text-slate-400">
              {overview.numeric_columns.length} Numeric, {overview.categorical_columns.length} Categorical
            </span>
          </div>

          <form onSubmit={handlePredict} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[500px] overflow-y-auto pr-2">
              {/* Numeric Inputs */}
              {overview.numeric_columns
                .filter((col) => col !== overview.target_column && !overview.excluded_columns.includes(col))
                .map((field) => (
                  <div key={field} className="glass-card rounded-xl p-3">
                    <label className="block text-xs font-mono font-medium text-slate-300 mb-1 truncate">
                      {field}
                    </label>
                    <input
                      type="number"
                      step="any"
                      required
                      value={formValues[field] ?? ''}
                      onChange={(e) => handleInputChange(field, parseFloat(e.target.value) || 0)}
                      className="w-full rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-1.5 text-xs font-mono text-slate-100 outline-none focus:border-indigo-500"
                    />
                  </div>
                ))}

              {/* Categorical Inputs */}
              {overview.categorical_columns
                .filter((col) => col !== overview.target_column && !overview.excluded_columns.includes(col))
                .map((field) => (
                  <div key={field} className="glass-card rounded-xl p-3">
                    <label className="block text-xs font-mono font-medium text-purple-300 mb-1 truncate">
                      {field} (Category)
                    </label>
                    <input
                      type="text"
                      required
                      value={formValues[field] ?? ''}
                      onChange={(e) => handleInputChange(field, e.target.value)}
                      className="w-full rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-1.5 text-xs font-mono text-slate-100 outline-none focus:border-purple-500"
                    />
                  </div>
                ))}
            </div>

            {error && (
              <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
                {error}
              </div>
            )}

            <div className="pt-2 flex justify-end">
              <button
                type="submit"
                disabled={isLoading}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 cursor-pointer"
              >
                {isLoading ? (
                  <>
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    <span>Inference via FastAPI...</span>
                  </>
                ) : (
                  <>
                    <Calculator className="h-4 w-4" />
                    <span>Predict & Compute Intervals</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Prediction Results Display (1 col) */}
        <div className="space-y-4">
          <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-6 shadow-lg">
            <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              Inference Outcome
            </h4>

            {prediction ? (
              <div className="space-y-5">
                {/* Point Prediction */}
                <div className="glass-card rounded-2xl p-4 text-center border-indigo-500/30 bg-indigo-950/20">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-400">
                    Predicted Point Value ({prediction.target_name})
                  </span>
                  <p className="text-3xl font-extrabold font-mono text-white mt-1">
                    {prediction.predicted_value.toFixed(3)}
                  </p>
                </div>

                {/* 95% CI vs 95% PI */}
                <div className="space-y-3">
                  <div className="rounded-xl border border-purple-500/30 bg-purple-950/20 p-3.5">
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-semibold text-purple-300">95% Confidence Interval (Mean)</span>
                      <span className="text-[10px] text-purple-400 font-mono">E(y|X)</span>
                    </div>
                    <p className="font-mono text-sm font-bold text-white">
                      [{prediction.ci_lower_95.toFixed(3)}, {prediction.ci_upper_95.toFixed(3)}]
                    </p>
                    <span className="text-[10px] text-slate-400 block mt-1">
                      Width: {(prediction.ci_upper_95 - prediction.ci_lower_95).toFixed(3)}
                    </span>
                  </div>

                  <div className="rounded-xl border border-indigo-500/30 bg-indigo-950/20 p-3.5">
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-semibold text-indigo-300">95% Prediction Interval (Observation)</span>
                      <span className="text-[10px] text-indigo-400 font-mono">y_new</span>
                    </div>
                    <p className="font-mono text-sm font-bold text-white">
                      [{prediction.pi_lower_95.toFixed(3)}, {prediction.pi_upper_95.toFixed(3)}]
                    </p>
                    <span className="text-[10px] text-slate-400 block mt-1">
                      Width: {(prediction.pi_upper_95 - prediction.pi_lower_95).toFixed(3)}
                    </span>
                  </div>
                </div>

                {/* Statistical Note */}
                <div className="flex items-start gap-2 text-slate-400 text-[11px] leading-relaxed border-t border-slate-800/80 pt-3">
                  <Scale className="h-4 w-4 text-indigo-400 shrink-0 mt-0.5" />
                  <span>
                    PI is strictly wider than CI because it accounts for individual variance ($\sigma_\epsilon^2$).
                  </span>
                </div>

                {/* Download Prediction CSV */}
                <button
                  onClick={downloadPredictionCsv}
                  className="w-full flex items-center justify-center gap-2 rounded-xl border border-slate-800 bg-slate-900/80 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-800 hover:text-white transition-all cursor-pointer"
                >
                  <Download className="h-4 w-4" />
                  <span>Download Prediction Record (CSV)</span>
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center text-slate-500">
                <Calculator className="h-10 w-10 stroke-1 mb-2 text-slate-600" />
                <p className="text-xs">Adjust the feature inputs on the left and click Predict to calculate intervals.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
