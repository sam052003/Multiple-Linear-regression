'use client';

import React, { useState } from 'react';
import {
  ShieldCheck,
  Sliders,
  Play,
  CheckCircle2,
  Cpu,
  Layers,
  ArrowRight,
} from 'lucide-react';
import { DatasetOverviewResponse, TrainResponse } from '../../types/api';
import { trainBaselineModel } from '../../lib/api';

interface DataPreparationTabProps {
  overview: DatasetOverviewResponse | null;
  trainResult: TrainResponse | null;
  onTrainComplete: (result: TrainResponse) => void;
  onNavigateToTab: (tab: string) => void;
}

export default function DataPreparationTab({
  overview,
  trainResult,
  onTrainComplete,
  onNavigateToTab,
}: DataPreparationTabProps) {
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!overview) return null;

  const handleTrainBaseline = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await trainBaselineModel(overview.dataset_id, overview.target_column, testSize, randomState);
      onTrainComplete(res);
    } catch (err: any) {
      setError(err.message || 'Baseline training failed');
    } finally {
      setIsLoading(false);
    }
  };

  const trainSamples = Math.round(overview.total_rows * (1 - testSize));
  const testSamples = overview.total_rows - trainSamples;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
            CRISP-DM Stage 3: Data Preparation
          </span>
        </div>
        <h3 className="text-xl font-bold text-white">Data Preprocessing & Leakage Prevention</h3>
        <p className="text-xs text-slate-400">
          Strict Scikit-learn Pipeline architecture ensuring zero information leakage between Training and Test sets.
        </p>
      </div>

      {/* Leakage Prevention Architecture Blueprint */}
      <div className="rounded-2xl border border-emerald-500/20 bg-emerald-950/10 p-5">
        <div className="flex items-center gap-2 text-emerald-400 mb-3">
          <ShieldCheck className="h-5 w-5" />
          <h4 className="text-sm font-semibold">Strict Leakage-Free Pipeline Protocol</h4>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-slate-300">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5">
            <span className="font-semibold text-indigo-400 block mb-1">1. Split Before Transform</span>
            Dataset is partitioned into Train ({((1 - testSize) * 100).toFixed(0)}%) and Test ({(testSize * 100).toFixed(0)}%) prior to any statistical estimation.
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5">
            <span className="font-semibold text-purple-400 block mb-1">2. Pipeline Fit on Train Only</span>
            Numeric Median Imputer and StandardScaler compute parameters exclusively on the Training set. Test data only uses .transform().
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5">
            <span className="font-semibold text-pink-400 block mb-1">3. Dummy Variable Encoding</span>
            Categorical variables use OneHotEncoder with <code>drop=&apos;first&apos;</code> to avoid the Dummy Variable Trap (Multicollinearity).
          </div>
        </div>
      </div>

      {/* Interactive Parameters Card */}
      <div className="glass-card rounded-2xl p-6">
        <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
          <Sliders className="h-4 w-4 text-indigo-400" />
          Split & Model Hyperparameters
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Test Size Slider */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">Test Set Ratio:</span>
              <span className="font-mono font-semibold text-indigo-400">{(testSize * 100).toFixed(0)}% ({testSamples} samples)</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.4"
              step="0.05"
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>10% (More Train)</span>
              <span>20% (Default)</span>
              <span>40% (Conservative)</span>
            </div>
          </div>

          {/* Random Seed */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">Random Seed (Reproducibility):</span>
              <span className="font-mono font-semibold text-purple-400">{randomState}</span>
            </div>
            <input
              type="number"
              value={randomState}
              onChange={(e) => setRandomState(parseInt(e.target.value) || 42)}
              className="w-full rounded-xl border border-slate-800 bg-slate-900/60 px-3.5 py-1.5 text-sm text-slate-200 outline-none focus:border-indigo-500"
            />
            <span className="text-[10px] text-slate-500 mt-1 block">
              Default seed 42 ensures identical Train/Test split across sessions.
            </span>
          </div>
        </div>

        {error && (
          <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            {error}
          </div>
        )}

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-slate-800/80 pt-4">
          <div className="text-xs text-slate-400">
            Training Partition: <span className="font-mono text-white">{trainSamples}</span> rows | Test Partition:{' '}
            <span className="font-mono text-white">{testSamples}</span> rows
          </div>

          <button
            onClick={handleTrainBaseline}
            disabled={isLoading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                <span>Fitting Scikit-learn Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" />
                <span>Fit Baseline Linear Regression</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Train Baseline Results */}
      {trainResult && (
        <div className="rounded-2xl border border-indigo-500/20 bg-indigo-950/10 p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-indigo-300">
              <CheckCircle2 className="h-5 w-5 text-emerald-400" />
              <h4 className="text-sm font-semibold">Baseline Model Fitted Successfully</h4>
            </div>
            <button
              onClick={() => onNavigateToTab('feature_selection')}
              className="flex items-center gap-1.5 text-xs font-semibold text-indigo-400 hover:text-indigo-300"
            >
              <span>Compare 5 Feature Selection Methods</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Metrics comparison cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center mb-5">
            <div className="glass-card rounded-xl p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">Test R² Score</span>
              <p className="text-lg font-bold text-indigo-400 font-mono mt-1">
                {(trainResult.test_metrics.R2 * 100).toFixed(2)}%
              </p>
              <span className="text-[10px] text-slate-500">Train: {(trainResult.train_metrics.R2 * 100).toFixed(2)}%</span>
            </div>

            <div className="glass-card rounded-xl p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">Adjusted R²</span>
              <p className="text-lg font-bold text-purple-400 font-mono mt-1">
                {(trainResult.test_metrics.Adj_R2 * 100).toFixed(2)}%
              </p>
              <span className="text-[10px] text-slate-500">Penalizes extra features</span>
            </div>

            <div className="glass-card rounded-xl p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">Test RMSE</span>
              <p className="text-lg font-bold text-pink-400 font-mono mt-1">
                {trainResult.test_metrics.RMSE.toFixed(3)}
              </p>
              <span className="text-[10px] text-slate-500">Target scale error</span>
            </div>

            <div className="glass-card rounded-xl p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">Encoded Features</span>
              <p className="text-lg font-bold text-slate-200 font-mono mt-1">
                {trainResult.n_features}
              </p>
              <span className="text-[10px] text-slate-500">After One-Hot Encoding</span>
            </div>
          </div>

          {/* Top Coefficients Table */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h5 className="text-xs font-semibold text-slate-300 mb-2">Model Intercept: {trainResult.intercept.toFixed(4)}</h5>
            <div className="max-h-40 overflow-y-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="text-[10px] text-slate-500 uppercase bg-slate-950/40">
                  <tr>
                    <th className="px-3 py-1.5">Transformed Feature</th>
                    <th className="px-3 py-1.5">Regression Coefficient (Beta)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/40">
                  {Object.entries(trainResult.coefficients)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .slice(0, 10)
                    .map(([feat, coef]) => (
                      <tr key={feat} className="hover:bg-slate-800/30">
                        <td className="px-3 py-1 font-mono text-[11px] text-slate-300">{feat}</td>
                        <td className="px-3 py-1 font-mono text-[11px]">
                          <span className={coef >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                            {coef >= 0 ? `+${coef.toFixed(4)}` : coef.toFixed(4)}
                          </span>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
