'use client';

import React from 'react';
import { BarChart3, Database, ShieldAlert, Sparkles, AlertCircle } from 'lucide-react';
import { DatasetOverviewResponse } from '../../types/api';
import PlotlyChart from '../PlotlyChart';

interface DataUnderstandingTabProps {
  overview: DatasetOverviewResponse | null;
}

export default function DataUnderstandingTab({ overview }: DataUnderstandingTabProps) {
  if (!overview) return null;

  // Prepare data for Correlation with Target bar chart
  const corrEntries = Object.entries(overview.target_correlation || {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  const corrFeatures = corrEntries.map(([feat]) => feat);
  const corrValues = corrEntries.map(([, val]) => val);
  const barColors = corrValues.map((v) =>
    v >= 0 ? 'rgba(99, 102, 241, 0.85)' : 'rgba(236, 72, 153, 0.85)'
  );

  const correlationPlotData = [
    {
      x: corrFeatures,
      y: corrValues,
      type: 'bar',
      marker: { color: barColors },
      text: corrValues.map((v) => v.toFixed(3)),
      textposition: 'auto',
      hoverinfo: 'x+y',
    },
  ];

  const correlationPlotLayout = {
    title: {
      text: `Pearson Correlation with Target (${overview.target_column})`,
      font: { size: 14, color: '#f1f5f9' },
    },
    xaxis: { tickangle: -35, automargin: true },
    yaxis: { title: 'Correlation Coefficient (r)', range: [-1, 1] },
    height: 380,
  };

  const missingSummary = overview.quality_summary?.missing_summary || {};
  const hasMissing = Object.keys(missingSummary).length > 0;

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
            CRISP-DM Stage 2: Data Understanding
          </span>
        </div>
        <h3 className="text-xl font-bold text-white">Exploratory Data Analysis (EDA)</h3>
        <p className="text-xs text-slate-400">
          Understanding feature distributions, Pearson correlation with target variable, and missing data completeness.
        </p>
      </div>

      {/* Target Distribution Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card rounded-2xl p-5 md:col-span-1">
          <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-pink-400" />
            Target Variable Profile
          </h4>
          <div className="space-y-3 text-xs">
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <span className="text-slate-400">Target Name:</span>
              <span className="font-mono font-semibold text-pink-400">{overview.target_column}</span>
            </div>
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <span className="text-slate-400">Mean:</span>
              <span className="font-mono text-slate-200">
                {overview.target_stats?.mean !== undefined ? overview.target_stats.mean.toFixed(2) : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <span className="text-slate-400">Std Deviation:</span>
              <span className="font-mono text-slate-200">
                {overview.target_stats?.std !== undefined ? overview.target_stats.std.toFixed(2) : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <span className="text-slate-400">Min Value:</span>
              <span className="font-mono text-slate-200">
                {overview.target_stats?.min !== undefined ? overview.target_stats.min.toFixed(2) : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Max Value:</span>
              <span className="font-mono text-slate-200">
                {overview.target_stats?.max !== undefined ? overview.target_stats.max.toFixed(2) : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        {/* Missing Values Status */}
        <div className="glass-card rounded-2xl p-5 md:col-span-2">
          <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Database className="h-4 w-4 text-indigo-400" />
            Data Quality & Completeness
          </h4>
          {hasMissing ? (
            <div className="space-y-2">
              <p className="text-xs text-amber-300 flex items-center gap-1.5">
                <AlertCircle className="h-4 w-4" />
                Missing values detected. The automated Pipeline will apply median imputation for numeric features and mode imputation for categorical features.
              </p>
              <div className="max-h-36 overflow-y-auto">
                <table className="w-full text-xs text-left text-slate-300">
                  <thead className="text-[11px] text-slate-400 bg-slate-900/40">
                    <tr>
                      <th className="px-3 py-1.5">Feature</th>
                      <th className="px-3 py-1.5">Missing Count</th>
                      <th className="px-3 py-1.5">Ratio</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(missingSummary).map(([feat, info]: [string, any]) => (
                      <tr key={feat} className="border-t border-slate-800/60">
                        <td className="px-3 py-1 text-slate-300 font-mono text-[11px]">{feat}</td>
                        <td className="px-3 py-1 font-mono text-amber-400">{info.count}</td>
                        <td className="px-3 py-1 font-mono text-slate-400">{(info.ratio * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-6 text-center">
              <span className="rounded-full bg-emerald-500/10 p-3 text-emerald-400 mb-2">
                <ShieldAlert className="h-6 w-6" />
              </span>
              <p className="text-sm font-semibold text-emerald-400">Zero Missing Values Detected</p>
              <p className="text-xs text-slate-400 mt-1">
                All {overview.n_rows.toLocaleString()} records across {overview.n_cols} columns are complete.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Correlation Chart */}
      {corrFeatures.length > 0 && (
        <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="h-4 w-4 text-indigo-400" />
            <h4 className="text-sm font-semibold text-white">Target Correlation Analysis</h4>
          </div>
          <PlotlyChart data={correlationPlotData} layout={correlationPlotLayout} />
        </div>
      )}

      {/* Feature Breakdown Lists */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card rounded-2xl p-5">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-indigo-400 mb-3">
            Numeric Features ({overview.numeric_columns.length})
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {overview.numeric_columns.map((feat) => (
              <span
                key={feat}
                className="rounded-lg border border-slate-700 bg-slate-800/60 px-2.5 py-1 text-[11px] font-mono text-slate-300"
              >
                {feat}
              </span>
            ))}
          </div>
        </div>

        <div className="glass-card rounded-2xl p-5">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-purple-400 mb-3">
            Categorical Features ({overview.categorical_columns.length})
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {overview.categorical_columns.length > 0 ? (
              overview.categorical_columns.map((feat) => (
                <span
                  key={feat}
                  className="rounded-lg border border-purple-900/40 bg-purple-950/20 px-2.5 py-1 text-[11px] font-mono text-purple-300"
                >
                  {feat}
                </span>
              ))
            ) : (
              <span className="text-xs text-slate-500 italic">None (All features are numeric)</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
