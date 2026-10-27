'use client';

import React, { useState } from 'react';
import {
  ShieldCheck,
  Sliders,
  Play,
  CheckCircle2,
  Layers,
  ArrowRight,
  Database,
  AlertTriangle,
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

  const validTargetRows =
    overview.n_valid_target_rows !== undefined && overview.n_valid_target_rows !== null
      ? overview.n_valid_target_rows
      : overview.n_rows;
  const droppedTargetRows = overview.n_dropped_target_rows ?? (overview.n_rows - validTargetRows);

  const estimatedTrainSamples = Math.round(validTargetRows * (1 - testSize));
  const estimatedTestSamples = validTargetRows - estimatedTrainSamples;

  const handleTrainBaseline = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await trainBaselineModel(
        overview.id,
        overview.target_column,
        testSize,
        randomState,
        overview.custom_data
      );
      onTrainComplete(res);
    } catch (err: any) {
      setError(err.message || '基準多元線性回歸模型訓練失敗');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
            CRISP-DM 第三階段：資料準備（Data Preparation）
          </span>
        </div>
        <h3 className="text-xl font-bold text-white">特徵前處理與防止資料洩漏管線</h3>
        <p className="text-xs text-slate-400">
          採用嚴謹之 Scikit-learn 機器學習 Pipeline 架構，確保訓練集與測試集之間零資訊洩漏。
        </p>
      </div>

      {/* Sample Accounting & Verification Grid */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-3">
          <Database className="h-4 w-4 text-indigo-400" />
          <h4 className="text-sm font-semibold text-white">樣本數計數與真實訓練資料一致性檢驗</h4>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 text-xs">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
            <span className="text-slate-400 block text-[11px] mb-1">1. 原始資料筆數</span>
            <span className="text-sm font-mono font-bold text-white">{overview.n_rows.toLocaleString()}</span>
            <span className="text-[10px] text-slate-500 block mt-0.5">來源總筆數</span>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
            <span className="text-slate-400 block text-[11px] mb-1">2. 有效 Target 筆數</span>
            <span className="text-sm font-mono font-bold text-emerald-400">{validTargetRows.toLocaleString()}</span>
            <span className="text-[10px] text-slate-500 block mt-0.5">可回歸數值樣本</span>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
            <span className="text-slate-400 block text-[11px] mb-1">3. 排除無效 Target</span>
            <span className="text-sm font-mono font-bold text-amber-400">{droppedTargetRows.toLocaleString()}</span>
            <span className="text-[10px] text-slate-500 block mt-0.5">缺失或非數值列</span>
          </div>

          <div className={`rounded-xl border p-3 ${trainResult ? 'border-indigo-500/30 bg-indigo-500/10' : 'border-slate-800 bg-slate-900/40'}`}>
            <span className="text-slate-400 block text-[11px] mb-1">4. 實際訓練集筆數</span>
            <span className="text-sm font-mono font-bold text-indigo-300">
              {trainResult ? `${trainResult.n_train_samples.toLocaleString()} 筆` : `約 ${estimatedTrainSamples.toLocaleString()} 筆（預估）`}
            </span>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              {trainResult ? '後端真實訓練集' : '未執行模型訓練'}
            </span>
          </div>

          <div className={`rounded-xl border p-3 ${trainResult ? 'border-purple-500/30 bg-purple-500/10' : 'border-slate-800 bg-slate-900/40'}`}>
            <span className="text-slate-400 block text-[11px] mb-1">5. 實際測試集筆數</span>
            <span className="text-sm font-mono font-bold text-purple-300">
              {trainResult ? `${trainResult.n_test_samples.toLocaleString()} 筆` : `約 ${estimatedTestSamples.toLocaleString()} 筆（預估）`}
            </span>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              {trainResult ? '後端真實測試集' : '未執行模型訓練'}
            </span>
          </div>
        </div>
      </div>

      {/* Leakage Prevention Architecture Blueprint */}
      <div className="rounded-2xl border border-emerald-500/20 bg-emerald-950/10 p-5">
        <div className="flex items-center gap-2 text-emerald-400 mb-3">
          <ShieldCheck className="h-5 w-5" />
          <h4 className="text-sm font-semibold">嚴格防範資料洩漏（Leakage-Free）之管線規範</h4>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-slate-300">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5">
            <span className="font-semibold text-indigo-400 block mb-1">1. 切分優先於特徵轉換</span>
            在任何統計參數計算前，即將有效樣本嚴格切分為訓練集 ({((1 - testSize) * 100).toFixed(0)}%) 與測試集 ({(testSize * 100).toFixed(0)}%)。
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5">
            <span className="font-semibold text-purple-400 block mb-1">2. Pipeline 僅在訓練集 Fit</span>
            數值中位數補值、標準化縮放（StandardScaler）等轉換器，僅以訓練集計算均值與標準差，測試集僅執行 .transform()。
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5">
            <span className="font-semibold text-pink-400 block mb-1">3. 虛擬變數編碼防範多重共線性</span>
            類別變數使用 OneHotEncoder 設定 <code>drop=&apos;first&apos;</code>，主動排除基準類別，避免虛擬變數陷阱（Dummy Variable Trap）造成共線性。
          </div>
        </div>
      </div>

      {/* Interactive Parameters Card */}
      <div className="glass-card rounded-2xl p-6">
        <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
          <Sliders className="h-4 w-4 text-indigo-400" />
          資料切分與基準模型超參數
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Test Size Slider */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">測試集切分比例（Test Ratio）：</span>
              <span className="font-mono font-semibold text-indigo-400">
                {(testSize * 100).toFixed(0)}%（預估約 {estimatedTestSamples} 筆）
              </span>
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
              <span>10%（訓練樣本多）</span>
              <span>20%（標準推薦值）</span>
              <span>40%（保守驗證）</span>
            </div>
          </div>

          {/* Random Seed */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">隨機種子（可重複性 Reproducibility）：</span>
              <span className="font-mono font-semibold text-purple-400">{randomState}</span>
            </div>
            <input
              type="number"
              value={randomState}
              onChange={(e) => setRandomState(parseInt(e.target.value) || 42)}
              className="w-full rounded-xl border border-slate-800 bg-slate-900/60 px-3.5 py-1.5 text-sm text-slate-200 outline-none focus:border-indigo-500"
            />
            <span className="text-[10px] text-slate-500 mt-1 block">
              預設種子 42 確保跨連線與跨實驗之切分結果一致。
            </span>
          </div>
        </div>

        {error && (
          <div className="mt-4 flex items-start gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
            <div>
              <span className="font-semibold block mb-0.5">模型切分與訓練失敗：</span>
              <span>{error}</span>
            </div>
          </div>
        )}

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-slate-800/80 pt-4">
          <div className="text-xs text-slate-400">
            {trainResult ? (
              <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
                <CheckCircle2 className="h-4 w-4" />
                真實訓練集：<span className="font-mono text-white">{trainResult.n_train_samples}</span> 筆 | 真實測試集：{' '}
                <span className="font-mono text-white">{trainResult.n_test_samples}</span> 筆（切分完成）
              </span>
            ) : (
              <span>
                預估訓練集：<span className="font-mono text-slate-300">{estimatedTrainSamples}</span> 筆 | 預估測試集：{' '}
                <span className="font-mono text-slate-300">{estimatedTestSamples}</span> 筆（尚未切分）
              </span>
            )}
          </div>

          <button
            onClick={handleTrainBaseline}
            disabled={isLoading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                <span>正在訓練 Scikit-learn 管線...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" />
                <span>訓練基準多元線性回歸模型（Baseline）</span>
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
              <h4 className="text-sm font-semibold">
                基準多元線性回歸模型訓練完成（訓練集：{trainResult.n_train_samples} 筆 / 測試集：{trainResult.n_test_samples} 筆）
              </h4>
            </div>
            <button
              onClick={() => onNavigateToTab('feature_selection')}
              className="flex items-center gap-1.5 text-xs font-semibold text-indigo-400 hover:text-indigo-300"
            >
              <span>前往比較五種特徵選擇方法</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Metrics comparison cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center mb-5">
            <div className="glass-card rounded-xl p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">測試集決定係數（R²）</span>
              <p className="text-lg font-bold text-indigo-400 font-mono mt-1">
                {trainResult.test_metrics?.R2 !== undefined
                  ? `${(trainResult.test_metrics.R2 * 100).toFixed(2)}%`
                  : 'N/A'}
              </p>
              <span className="text-[10px] text-slate-500">
                訓練集：{' '}
                {trainResult.train_metrics?.R2 !== undefined
                  ? `${(trainResult.train_metrics.R2 * 100).toFixed(2)}%`
                  : 'N/A'}
              </span>
            </div>

            <div className="glass-card rounded-xl p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">調整後決定係數（Adj. R²）</span>
              <p className="text-lg font-bold text-purple-400 font-mono mt-1">
                {trainResult.test_metrics?.Adj_R2 !== undefined && trainResult.test_metrics.Adj_R2 !== null
                  ? `${(trainResult.test_metrics.Adj_R2 * 100).toFixed(2)}%`
                  : 'N/A'}
              </p>
              <span className="text-[10px] text-slate-500">懲罰不顯著之額外特徵</span>
            </div>

            <div className="glass-card rounded-xl p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">均方根誤差（RMSE）</span>
              <p className="text-lg font-bold text-pink-400 font-mono mt-1">
                {trainResult.test_metrics?.RMSE !== undefined
                  ? trainResult.test_metrics.RMSE.toFixed(3)
                  : 'N/A'}
              </p>
              <span className="text-[10px] text-slate-500">與目標變數同單位之預測誤差</span>
            </div>

            <div className="glass-card rounded-xl p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">轉換後總特徵數</span>
              <p className="text-lg font-bold text-slate-200 font-mono mt-1">
                {trainResult.encoded_feature_names?.length || trainResult.n_features || 0}
              </p>
              <span className="text-[10px] text-slate-500">經 One-Hot 編碼處理後</span>
            </div>
          </div>

          {/* Top Coefficients Table */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h5 className="text-xs font-semibold text-slate-300 mb-2">
              模型截距項（Intercept）：{trainResult.intercept?.toFixed(4)}
            </h5>
            <div className="max-h-48 overflow-y-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="text-[10px] text-slate-500 uppercase bg-slate-950/40 sticky top-0">
                  <tr>
                    <th className="px-3 py-1.5">轉換後特徵名稱</th>
                    <th className="px-3 py-1.5">回歸係數（Regression Beta）</th>
                    <th className="px-3 py-1.5">絕對影響度（|Beta|）</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/40">
                  {trainResult.coefficients.slice(0, 15).map((item) => (
                    <tr key={item.feature} className="hover:bg-slate-800/30">
                      <td className="px-3 py-1 font-mono text-[11px] text-slate-300">{item.feature}</td>
                      <td className="px-3 py-1 font-mono text-[11px]">
                        <span className={item.coefficient >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {item.coefficient >= 0 ? `+${item.coefficient.toFixed(4)}` : item.coefficient.toFixed(4)}
                        </span>
                      </td>
                      <td className="px-3 py-1 font-mono text-[11px] text-slate-400">
                        {item.abs_coefficient.toFixed(4)}
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
