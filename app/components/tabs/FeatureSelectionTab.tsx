'use client';

import React, { useState } from 'react';
import {
  Sparkles,
  Trophy,
  Zap,
  CheckCircle2,
  Clock,
  ArrowRight,
  Filter,
} from 'lucide-react';
import {
  DatasetOverviewResponse,
  FeatureSelectionResponse,
  MethodComparisonItem,
} from '../../types/api';
import { runFeatureSelection } from '../../lib/api';
import PlotlyChart from '../PlotlyChart';

interface FeatureSelectionTabProps {
  overview: DatasetOverviewResponse | null;
  fsResult: FeatureSelectionResponse | null;
  onFsComplete: (result: FeatureSelectionResponse) => void;
  onSelectMethodForEval: (method: MethodComparisonItem) => void;
  onNavigateToTab: (tab: string) => void;
}

export default function FeatureSelectionTab({
  overview,
  fsResult,
  onFsComplete,
  onSelectMethodForEval,
  onNavigateToTab,
}: FeatureSelectionTabProps) {
  const [kFeatures, setKFeatures] = useState(5);
  const [fastMode, setFastMode] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeMethodDetail, setActiveMethodDetail] = useState<string | null>(null);

  if (!overview) return null;

  const handleRunComparison = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await runFeatureSelection(
        overview.id,
        overview.target_column,
        kFeatures,
        fastMode,
        overview.custom_data
      );
      onFsComplete(res);
      if (res.comparison_table.length > 0) {
        setActiveMethodDetail(res.comparison_table[0].method);
      }
    } catch (err: any) {
      setError(err.message || 'Feature selection computation failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Prepare plot data for comparing R2 and RMSE across methods
  const methods = fsResult?.comparison_table.map((r) => r.method) || [];
  const r2Scores = fsResult?.comparison_table.map((r) => r.r2) || [];
  const cvR2Scores = fsResult?.comparison_table.map((r) => r.cv_r2_mean || 0) || [];

  const comparisonPlotData = [
    {
      name: 'Test R²',
      x: methods,
      y: r2Scores,
      type: 'bar',
      marker: { color: 'rgba(99, 102, 241, 0.85)' },
    },
    {
      name: '5-Fold CV Mean R²',
      x: methods,
      y: cvR2Scores,
      type: 'bar',
      marker: { color: 'rgba(236, 72, 153, 0.85)' },
    },
  ];

  const comparisonPlotLayout = {
    title: { text: 'Method Performance Comparison (R² & 5-Fold CV)', font: { size: 14, color: '#f1f5f9' } },
    barmode: 'group',
    xaxis: { tickangle: -20, automargin: true },
    yaxis: { title: 'R² Score', range: [0, 1] },
    height: 360,
    legend: { orientation: 'h', y: 1.15, x: 0.1 },
  };

  const selectedResult =
    fsResult?.comparison_table.find((r) => r.method === activeMethodDetail) ||
    fsResult?.comparison_table[0];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
            CRISP-DM Stage 4: Modeling & Feature Selection
          </span>
        </div>
        <h3 className="text-xl font-bold text-white">5-Method Feature Selection & Benchmark</h3>
        <p className="text-xs text-slate-400">
          Parallel comparative analysis of Filter, Wrapper (RFE & SFS), Embedded (Lasso), and Model-based (Permutation) methods against Baseline.
        </p>
      </div>

      {/* Control Card */}
      <div className="glass-card rounded-2xl p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-6">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Target Subset Size (k features):</label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="2"
                  max={Math.min(10, Math.max(3, overview.available_features.length - 1))}
                  value={kFeatures}
                  onChange={(e) => setKFeatures(parseInt(e.target.value))}
                  className="accent-indigo-500 cursor-pointer"
                />
                <span className="font-mono text-sm font-bold text-indigo-400">{kFeatures}</span>
              </div>
            </div>

            <div className="flex items-center gap-2 pt-4 sm:pt-0">
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={fastMode}
                  onChange={(e) => setFastMode(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
                <span className="ml-2 text-xs font-medium text-slate-300 flex items-center gap-1">
                  <Zap className="h-3.5 w-3.5 text-amber-400" />
                  Fast Mode (Vercel Guard)
                </span>
              </label>
            </div>
          </div>

          <button
            onClick={handleRunComparison}
            disabled={isLoading}
            className="flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                <span>Running 5 Algorithms & CV...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span>Run Feature Selection Comparison</span>
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            {error}
          </div>
        )}
      </div>

      {/* Comparison Results */}
      {fsResult && (
        <div className="space-y-6">
          {/* Best Method Trophy Card */}
          <div className="relative overflow-hidden rounded-2xl border border-amber-500/30 bg-gradient-to-r from-amber-950/20 via-slate-900 to-indigo-950/20 p-5 shadow-xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-500/20 border border-amber-500/40 text-amber-400 shadow-lg shadow-amber-500/20">
                  <Trophy className="h-6 w-6" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-amber-400">Top Performing Model</span>
                    <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 text-[10px] font-semibold text-emerald-400">
                      Highest CV R²
                    </span>
                  </div>
                  <h4 className="text-lg font-bold text-white mt-0.5">{fsResult.best_method}</h4>
                </div>
              </div>

              {selectedResult && (
                <button
                  onClick={() => {
                    onSelectMethodForEval(selectedResult);
                    onNavigateToTab('evaluation');
                  }}
                  className="flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-5 py-2.5 text-xs font-semibold text-slate-950 shadow-lg shadow-amber-500/30 hover:bg-amber-400 transition-all cursor-pointer"
                >
                  <span>Evaluate {selectedResult.method}</span>
                  <ArrowRight className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>

          {/* Performance Comparison Chart */}
          <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
            <PlotlyChart data={comparisonPlotData} layout={comparisonPlotLayout} />
          </div>

          {/* Full Comparison Table */}
          <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
            <div className="mb-4 flex items-center justify-between">
              <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                <Filter className="h-4 w-4 text-indigo-400" />
                Complete Algorithms Benchmark Table
              </h4>
              <span className="text-xs text-slate-400">Click a row to inspect selected features</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="border-b border-slate-800 bg-slate-900/60 text-slate-400 uppercase text-[10px]">
                  <tr>
                    <th className="px-4 py-3">Method Name</th>
                    <th className="px-4 py-3">Features Selected</th>
                    <th className="px-4 py-3">Test R²</th>
                    <th className="px-4 py-3">Adjusted R²</th>
                    <th className="px-4 py-3">RMSE</th>
                    <th className="px-4 py-3">MAE</th>
                    <th className="px-4 py-3">5-Fold CV R²</th>
                    <th className="px-4 py-3">Duration</th>
                    <th className="px-4 py-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {fsResult.comparison_table.map((res) => {
                    const isBest = res.method === fsResult.best_method;
                    const isSelected = res.method === activeMethodDetail;

                    return (
                      <tr
                        key={res.method}
                        onClick={() => setActiveMethodDetail(res.method)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-indigo-950/40 border-l-2 border-indigo-500'
                            : 'hover:bg-slate-800/30'
                        }`}
                      >
                        <td className="px-4 py-3 font-semibold text-white flex items-center gap-2">
                          {isBest && <Trophy className="h-3.5 w-3.5 text-amber-400 shrink-0" />}
                          <span>{res.method}</span>
                        </td>
                        <td className="px-4 py-3 font-mono text-[11px] text-slate-300">
                          {res.features} features
                        </td>
                        <td className="px-4 py-3 font-mono font-semibold text-indigo-400">
                          {(res.r2 * 100).toFixed(2)}%
                        </td>
                        <td className="px-4 py-3 font-mono text-purple-400">
                          {res.adj_r2 !== null && res.adj_r2 !== undefined
                            ? `${(res.adj_r2 * 100).toFixed(2)}%`
                            : 'N/A'}
                        </td>
                        <td className="px-4 py-3 font-mono text-pink-400">
                          {res.rmse.toFixed(3)}
                        </td>
                        <td className="px-4 py-3 font-mono text-slate-300">
                          {res.mae.toFixed(3)}
                        </td>
                        <td className="px-4 py-3 font-mono text-emerald-400">
                          {res.cv_r2_mean !== null && res.cv_r2_mean !== undefined
                            ? `${(res.cv_r2_mean * 100).toFixed(2)}%`
                            : 'N/A'}
                        </td>
                        <td className="px-4 py-3 font-mono text-slate-400 flex items-center gap-1">
                          <Clock className="h-3 w-3 text-slate-500" />
                          <span>{res.time_sec.toFixed(3)}s</span>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectMethodForEval(res);
                              onNavigateToTab('evaluation');
                            }}
                            className="rounded-lg bg-indigo-600/20 border border-indigo-500/40 px-2.5 py-1 text-[11px] font-semibold text-indigo-300 hover:bg-indigo-600 hover:text-white transition-all cursor-pointer"
                          >
                            Select & Diagnose
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Selected Method Features Breakdown */}
          {selectedResult && (
            <div className="glass-card rounded-2xl p-5">
              <h4 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                Selected Features in {selectedResult.method} ({selectedResult.features})
              </h4>
              <div className="flex flex-wrap gap-2 mt-3">
                {selectedResult.selected_features.map((feat) => (
                  <span
                    key={feat}
                    className="rounded-xl border border-indigo-500/30 bg-indigo-950/40 px-3 py-1.5 font-mono text-xs text-indigo-200 shadow-sm"
                  >
                    {feat}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
