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
  Filter,
} from 'lucide-react';
import { DatasetOverviewResponse } from '../../types/api';

interface OverviewTabProps {
  overview: DatasetOverviewResponse | null;
  onNavigateToTab: (tab: string) => void;
  onTargetChange?: (newTarget: string) => void;
}

export default function OverviewTab({
  overview,
  onNavigateToTab,
  onTargetChange,
}: OverviewTabProps) {
  if (!overview) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <div className="h-7 w-7 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
          <span className="text-sm text-slate-400">正在載入問題資料集概況...</span>
        </div>
      </div>
    );
  }

  const is50Startups = overview.id === '50_startups';
  const isStudent = overview.id === 'student_performance';
  const isFlood = overview.id === 'flood_probability' || overview.id === 'flood_prediction';

  const validTargetRows =
    overview.n_valid_target_rows !== undefined && overview.n_valid_target_rows !== null
      ? overview.n_valid_target_rows
      : overview.n_rows;
  const droppedTargetRows = overview.n_dropped_target_rows ?? (overview.n_rows - validTargetRows);

  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-indigo-950/20 to-slate-900 p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
                CRISP-DM 第一階段：問題理解（Business Understanding）
              </span>
              {isStudent && (
                <span className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-400">
                  主要研究資料集
                </span>
              )}
              {is50Startups && (
                <span className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-xs font-semibold text-amber-400">
                  預設教學展示案例
                </span>
              )}
            </div>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-white">{overview.title}</h2>
            <p className="mt-1 text-sm text-slate-400 max-w-2xl">
              {is50Startups &&
                '經典 50 Startups 商業案例，分析研發費用（R&D）、管理費用（Administration）、行銷支出（Marketing Spend）及營運州別，建立多元線性回歸模型預測企業獲利（Profit）。'}
              {isStudent &&
                '深入研究學習時數、出勤率、同儕影響及社會環境因素對學業考試成績（Exam Score）之線性關聯與回歸效能。'}
              {isFlood &&
                '高維度環境觀測資料集，運用已驗證之抽樣資料探討各項氣候水文特徵對洪水機率（Flood Probability）之預測效果。'}
              {!is50Startups && !isStudent && !isFlood && '使用者自訂上傳之表格資料集，系統已完成自動欄位型別推論與前處理規格檢查。'}
            </p>
          </div>

          <button
            onClick={() => onNavigateToTab('understanding')}
            className="flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 transition-all hover:bg-indigo-500 shrink-0"
          >
            <span>前往第二階段：資料理解</span>
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">總資料筆數（Rows）</span>
            <div className="rounded-lg bg-indigo-500/10 p-2 text-indigo-400">
              <Hash className="h-4 w-4" />
            </div>
          </div>
          <p className="mt-2 text-2xl font-bold text-white">{overview.n_rows.toLocaleString()}</p>
          <span className="text-[11px] text-slate-500">樣本總數量</span>
        </div>

        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">原始特徵欄位數（Features）</span>
            <div className="rounded-lg bg-purple-500/10 p-2 text-purple-400">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <p className="mt-2 text-2xl font-bold text-white">{overview.available_features.length}</p>
          <span className="text-[11px] text-slate-500">
            {overview.numeric_columns.length} 個數值型，{overview.categorical_columns.length} 個類別型
          </span>
        </div>

        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">目標預測變數（Target）</span>
            <div className="rounded-lg bg-pink-500/10 p-2 text-pink-400">
              <Activity className="h-4 w-4" />
            </div>
          </div>
          <p className="mt-2 text-xl font-bold text-pink-400 truncate">{overview.target_column}</p>
          <span className="text-[11px] text-slate-500">
            有效樣本：{validTargetRows} / {overview.n_rows} 筆
          </span>
        </div>

        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">規格檢驗（14~39 規格）</span>
            <div
              className={`rounded-lg p-2 ${
                overview.is_compliant_14_39
                  ? 'bg-emerald-500/10 text-emerald-400'
                  : 'bg-amber-500/10 text-amber-400'
              }`}
            >
              {overview.is_compliant_14_39 ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <AlertTriangle className="h-4 w-4" />
              )}
            </div>
          </div>
          <p
            className={`mt-2 text-sm font-bold ${
              overview.is_compliant_14_39 ? 'text-emerald-400' : 'text-amber-400'
            }`}
          >
            {overview.is_compliant_14_39 ? '符合 14~39 特徵規格' : '教學範例提示'}
          </p>
          <span className="text-[10px] text-slate-500 block leading-tight">{overview.compliance_message}</span>
        </div>
      </div>

      {/* Target Column Selector & Sample Accounting Card */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-pink-400" />
              <h3 className="text-sm font-semibold text-white">
                目標預測變數（Target Variable）指定與確認
              </h3>
            </div>
            <p className="mt-1 text-xs text-slate-400">
              設定多元線性回歸模型之預測目標（Y）。系統自動標記推薦的連續數值欄位，避免誤選文字類別。
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400 shrink-0 font-medium">切換目標變數：</span>
            <select
              value={overview.target_column}
              onChange={(e) => onTargetChange && onTargetChange(e.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-900 px-3.5 py-2 text-sm font-semibold text-pink-300 outline-none focus:border-pink-500 cursor-pointer shadow-inner"
            >
              {overview.columns.map((col) => {
                const cand = overview.candidate_targets?.find((c) => c.column === col);
                const isNumeric = cand ? cand.is_numeric : overview.numeric_columns.includes(col);
                const isSuitable = cand ? cand.is_suitable_for_regression : isNumeric;
                const tag = isSuitable
                  ? '（推薦數值回歸）'
                  : '（類別或非數值）';
                return (
                  <option key={col} value={col} className="bg-slate-900 text-slate-200">
                    {col} {tag}
                  </option>
                );
              })}
            </select>
          </div>
        </div>

        {/* Target Validation Alert if invalid */}
        {overview.is_target_valid === false && (
          <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
            <div>
              <span className="font-semibold block mb-0.5">目標欄位規格不符警示：</span>
              <span>{overview.target_validation_message || '目前選取的目標欄位無法轉換為連續數值，不適合進行多元線性回歸。請從上方選單切換至推薦數值欄位。'}</span>
            </div>
          </div>
        )}

        {/* Sample Accounting Breakdown */}
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3 border-t border-slate-800/80 pt-4 text-xs">
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
            <span className="text-slate-400 block text-[11px]">原始資料筆數</span>
            <span className="text-sm font-mono font-bold text-white">{overview.n_rows.toLocaleString()} 筆</span>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
            <span className="text-slate-400 block text-[11px]">有效 Target 數值筆數</span>
            <span className="text-sm font-mono font-bold text-emerald-400">
              {validTargetRows.toLocaleString()} 筆
            </span>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
            <span className="text-slate-400 block text-[11px]">排除無效/缺失 Target 筆數</span>
            <span className="text-sm font-mono font-bold text-amber-400">
              {droppedTargetRows.toLocaleString()} 筆
            </span>
          </div>
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
              className="ml-1 underline font-semibold text-indigo-300 hover:text-white cursor-pointer"
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
            <h3 className="text-sm font-semibold text-white">前 10 筆原始資料預覽（Raw Data Preview）</h3>
          </div>
          <span className="text-xs text-slate-400">展示資料前 10 筆內容</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="border-b border-slate-800 bg-slate-900/60 text-slate-400 uppercase text-[10px] tracking-wider">
              <tr>
                {overview.preview_head.length > 0 &&
                  Object.keys(overview.preview_head[0]).map((col) => (
                    <th
                      key={col}
                      className={`px-4 py-3 font-semibold ${
                        col === overview.target_column ? 'text-pink-400 bg-pink-500/10' : ''
                      }`}
                    >
                      {col} {col === overview.target_column && '（目標變數）'}
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {overview.preview_head.map((row, idx) => (
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
