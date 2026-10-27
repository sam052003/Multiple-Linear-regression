'use client';

import React from 'react';
import {
  BookOpen,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Hash,
  Activity,
  ArrowRight,
  ShieldAlert,
} from 'lucide-react';
import { DatasetOverviewResponse } from '../../types/api';

interface OverviewTabProps {
  overview: DatasetOverviewResponse | null;
  onNavigateToTab: (tab: string) => void;
}

export default function OverviewTab({ overview, onNavigateToTab }: OverviewTabProps) {
  if (!overview) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <div className="h-7 w-7 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
          <span className="text-sm text-slate-400">Loading dataset overview...</span>
        </div>
      </div>
    );
  }

  const is50Startups = overview.dataset_id === '50_startups';
  const isStudent = overview.dataset_id === 'student_performance';
  const isFlood = overview.dataset_id === 'flood_prediction';

  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-indigo-950/20 to-slate-900 p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
                CRISP-DM Stage 1: Business Understanding
              </span>
              {isStudent && (
                <span className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-400">
                  Primary Research Dataset
                </span>
              )}
              {is50Startups && (
                <span className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-xs font-semibold text-amber-400">
                  Default Teaching Problem
                </span>
              )}
            </div>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-white">{overview.name}</h2>
            <p className="mt-1 text-sm text-slate-400 max-w-2xl">
              {is50Startups &&
                'Classic 50 Startups case study analyzing R&D, Administration, Marketing Spend and State location to predict corporate Profit.'}
              {isStudent &&
                'Comprehensive study evaluating the predictive relationship between study habits, socio-environmental factors, and academic Exam Scores.'}
              {isFlood &&
                'High-dimensional environmental dataset predicting flood risk probability using sub-sampled verified observations.'}
              {!is50Startups && !isStudent && !isFlood && 'Custom loaded tabular dataset with automated type inference.'}
            </p>
          </div>

          <button
            onClick={() => onNavigateToTab('understanding')}
            className="flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 transition-all hover:bg-indigo-500 shrink-0"
          >
            <span>Proceed to Data Understanding</span>
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Total Observations</span>
            <div className="rounded-lg bg-indigo-500/10 p-2 text-indigo-400">
              <Hash className="h-4 w-4" />
            </div>
          </div>
          <p className="mt-2 text-2xl font-bold text-white">{overview.total_rows.toLocaleString()}</p>
          <span className="text-[11px] text-slate-500">Rows in dataset</span>
        </div>

        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Features Count</span>
            <div className="rounded-lg bg-purple-500/10 p-2 text-purple-400">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <p className="mt-2 text-2xl font-bold text-white">{overview.total_features}</p>
          <span className="text-[11px] text-slate-500">
            {overview.numeric_features.length} Numeric, {overview.categorical_features.length} Categorical
          </span>
        </div>

        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Target Variable</span>
            <div className="rounded-lg bg-pink-500/10 p-2 text-pink-400">
              <Activity className="h-4 w-4" />
            </div>
          </div>
          <p className="mt-2 text-xl font-bold text-pink-400 truncate">{overview.target_column}</p>
          <span className="text-[11px] text-slate-500">
            Mean: {overview.target_mean?.toFixed(2)} (Std: {overview.target_std?.toFixed(2)})
          </span>
        </div>

        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Specification Check</span>
            <div
              className={`rounded-lg p-2 ${
                overview.is_feature_count_valid
                  ? 'bg-emerald-500/10 text-emerald-400'
                  : 'bg-amber-500/10 text-amber-400'
              }`}
            >
              {overview.is_feature_count_valid ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <AlertTriangle className="h-4 w-4" />
              )}
            </div>
          </div>
          <p
            className={`mt-2 text-sm font-bold ${
              overview.is_feature_count_valid ? 'text-emerald-400' : 'text-amber-400'
            }`}
          >
            {overview.is_feature_count_valid ? '14~39 Specs Met' : 'Teaching Case Note'}
          </p>
          <span className="text-[10px] text-slate-500 block leading-tight">{overview.feature_count_note}</span>
        </div>
      </div>

      {/* Notice Alert if 50 Startups */}
      {is50Startups && (
        <div className="flex items-start gap-3 rounded-2xl border border-amber-500/20 bg-amber-500/10 p-4 text-xs text-amber-300">
          <AlertTriangle className="h-5 w-5 shrink-0 text-amber-400" />
          <div>
            <span className="font-semibold text-amber-200">教學展示案例提示：</span>
            本資料集（50 Startups）為經典教學範例，僅有 4 個原始特徵與 50 筆資料，方便快速觀察數值與類別回歸行為。正式研究報告與 14~39 特徵規格請切換至
            <button
              onClick={() => onNavigateToTab('student_performance_trigger')}
              className="ml-1 underline font-semibold text-indigo-300 hover:text-white"
            >
              Student Performance Factors
            </button>
            。
          </div>
        </div>
      )}

      {/* Dataset Preview Table */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">First 5 Sample Records</h3>
          </div>
          <span className="text-xs text-slate-400">Previewing raw data</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="border-b border-slate-800 bg-slate-900/60 text-slate-400 uppercase text-[10px] tracking-wider">
              <tr>
                {overview.sample_head.length > 0 &&
                  Object.keys(overview.sample_head[0]).map((col) => (
                    <th
                      key={col}
                      className={`px-4 py-3 font-semibold ${
                        col === overview.target_column ? 'text-pink-400 bg-pink-500/10' : ''
                      }`}
                    >
                      {col} {col === overview.target_column && '(Target)'}
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {overview.sample_head.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                  {Object.entries(row).map(([col, val]) => (
                    <td
                      key={col}
                      className={`px-4 py-2.5 font-mono text-[11px] ${
                        col === overview.target_column ? 'font-semibold text-pink-300 bg-pink-500/5' : ''
                      }`}
                    >
                      {typeof val === 'number' ? val.toFixed(3).replace(/\.?0+$/, '') : String(val)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
