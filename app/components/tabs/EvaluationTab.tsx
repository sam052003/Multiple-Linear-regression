'use client';

import React, { useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Play,
  Info,
  Scale,
} from 'lucide-react';
import {
  DatasetOverviewResponse,
  EvaluationResponse,
  MethodComparisonItem,
} from '../../types/api';
import { runEvaluation } from '../../lib/api';
import PlotlyChart from '../PlotlyChart';

interface EvaluationTabProps {
  overview: DatasetOverviewResponse | null;
  evalResult: EvaluationResponse | null;
  selectedMethod: MethodComparisonItem | null;
  onEvalComplete: (result: EvaluationResponse) => void;
}

export default function EvaluationTab({
  overview,
  evalResult,
  selectedMethod,
  onEvalComplete,
}: EvaluationTabProps) {
  const [activeDiagTab, setActiveDiagTab] = useState<'linearity' | 'independence' | 'homoscedasticity' | 'normality' | 'vif'>('linearity');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!overview) return null;

  const handleRunEvaluation = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await runEvaluation(
        overview.id,
        overview.target_column,
        selectedMethod?.selected_features,
        0.2,
        42,
        overview.custom_data
      );
      onEvalComplete(res);
    } catch (err: any) {
      setError(err.message || '模型評估運算失敗');
    } finally {
      setIsLoading(false);
    }
  };

  // 1. Actual vs Predicted Plot
  const actuals = evalResult?.actual_vs_predicted_sample.map((d) => d.actual) || [];
  const preds = evalResult?.actual_vs_predicted_sample.map((d) => d.predicted) || [];
  const residuals = evalResult?.actual_vs_predicted_sample.map((d) => d.residual) || [];

  const minVal = actuals.length > 0 ? Math.min(...actuals) : 0;
  const maxVal = actuals.length > 0 ? Math.max(...actuals) : 100;

  const actualVsPredPlotData = evalResult
    ? [
        {
          x: actuals,
          y: preds,
          mode: 'markers',
          type: 'scatter',
          name: '觀測樣本點',
          marker: { color: 'rgba(99, 102, 241, 0.7)', size: 6 },
        },
        {
          x: [minVal, maxVal],
          y: [minVal, maxVal],
          mode: 'lines',
          type: 'scatter',
          name: '理想擬合線 (y = x)',
          line: { color: '#ec4899', dash: 'dash', width: 2 },
        },
      ]
    : [];

  const actualVsPredLayout = {
    title: { text: '實際值與預測值散布圖（Actual vs. Predicted）', font: { size: 14, color: '#f1f5f9' } },
    xaxis: { title: '實際目標變數值（Actual Target）' },
    yaxis: { title: '模型預測值（Predicted Value）' },
    height: 380,
  };

  // 2. 95% Confidence & Prediction Intervals Plot
  const piSamples = evalResult?.prediction_intervals_sample || [];
  const piPlotData = (() => {
    if (piSamples.length === 0) return [];

    const x = piSamples.map((_, i) => i + 1);
    const yPred = piSamples.map((d) => d.predicted);
    const yTrue = piSamples.map((d) => d.actual);
    const piLow = piSamples.map((d) => d.obs_ci_lower);
    const piUp = piSamples.map((d) => d.obs_ci_upper);
    const ciLow = piSamples.map((d) => d.mean_ci_lower);
    const ciUp = piSamples.map((d) => d.mean_ci_upper);

    return [
      // 95% PI Upper & Lower Band
      {
        x: [...x, ...[...x].reverse()],
        y: [...piUp, ...[...piLow].reverse()],
        fill: 'toself',
        fillcolor: 'rgba(99, 102, 241, 0.12)',
        line: { color: 'transparent' },
        name: '95% 預測區間（單一觀測值 Prediction Interval）',
        showlegend: true,
        type: 'scatter',
      },
      // 95% CI Upper & Lower Band
      {
        x: [...x, ...[...x].reverse()],
        y: [...ciUp, ...[...ciLow].reverse()],
        fill: 'toself',
        fillcolor: 'rgba(168, 85, 247, 0.25)',
        line: { color: 'transparent' },
        name: '95% 信賴區間（母體平均值 Confidence Interval）',
        showlegend: true,
        type: 'scatter',
      },
      // Predicted line
      {
        x: x,
        y: yPred,
        mode: 'lines',
        name: '擬合期望值 E(y|X)',
        line: { color: '#818cf8', width: 2.5 },
        type: 'scatter',
      },
      // Actual points
      {
        x: x,
        y: yTrue,
        mode: 'markers',
        name: '實際觀測值 (y)',
        marker: { color: '#38bdf8', size: 5 },
        type: 'scatter',
      },
    ];
  })();

  const piPlotLayout = {
    title: {
      text: '95% 母體平均值信賴區間 vs. 95% 單一觀測值預測區間',
      font: { size: 14, color: '#f1f5f9' },
    },
    xaxis: { title: '測試集樣本（依預測值由低至高排序）' },
    yaxis: { title: overview.target_column },
    height: 400,
    legend: { orientation: 'h', y: 1.15, x: 0 },
  };

  // 3. Residuals vs Fitted Plot
  const resPlotData = evalResult
    ? [
        {
          x: preds,
          y: residuals,
          mode: 'markers',
          type: 'scatter',
          name: '殘差分佈點',
          marker: { color: 'rgba(236, 72, 153, 0.7)', size: 5 },
        },
        {
          x: preds.length > 0 ? [Math.min(...preds), Math.max(...preds)] : [0, 1],
          y: [0, 0],
          mode: 'lines',
          name: '零殘差基準線',
          line: { color: '#ffffff', width: 1.5, dash: 'dot' },
        },
      ]
    : [];

  const resPlotLayout = {
    title: { text: '殘差與擬合值散布圖（Residuals vs. Fitted Values）', font: { size: 14, color: '#f1f5f9' } },
    xaxis: { title: '模型擬合值（Fitted Values）' },
    yaxis: { title: '殘差（Residuals: e = y - ŷ）' },
    height: 380,
  };

  const assumptions = evalResult?.assumptions;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
            CRISP-DM 第五階段：模型評估與診斷（Evaluation & Regression Diagnostics）
          </span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-xl font-bold text-white">模型評估與統計假設檢定</h3>
            <p className="text-xs text-slate-400">
              評估預測精準度、95% 預測區間涵蓋率，並針對高斯-馬可夫（Gauss-Markov）五大經典回歸統計假設進行全面檢定。
            </p>
          </div>

          <button
            onClick={handleRunEvaluation}
            disabled={isLoading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 shrink-0 cursor-pointer"
          >
            {isLoading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                <span>正在計算 OLS 與統計假設檢定...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" />
                <span>{evalResult ? '重新執行模型診斷' : '執行模型評估與統計診斷'}</span>
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

      {evalResult && (
        <div className="space-y-6">
          {/* Post Selection Inference Warning */}
          <div className="flex items-start gap-3 rounded-2xl border border-indigo-500/30 bg-indigo-950/20 p-4 text-xs text-indigo-200">
            <Info className="h-5 w-5 shrink-0 text-indigo-400 mt-0.5" />
            <div>
              <span className="font-semibold text-white">特徵選擇後推論（Post-Selection Inference）方法論提示：</span>
              <p className="mt-0.5 text-slate-300">
                當使用特徵選擇演算法篩選出的變數進行模型評估時，標準 OLS 的信賴區間與預測區間是條件於所選特徵子集之下的計算結果。真實情境下，因包含模型選擇的不確定性，未經校正的區間可能呈現名義過窄現象，應結合測試集實際涵蓋率審慎評估。
              </p>
            </div>
          </div>

          {/* Metric KPI Cards (Train vs Test) */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="glass-card rounded-2xl p-4 text-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">測試集 R²</span>
              <p className="text-2xl font-bold font-mono text-indigo-400 mt-1">
                {(evalResult.test_metrics.R2 * 100).toFixed(2)}%
              </p>
              <span className="text-[10px] text-slate-500">訓練集：{(evalResult.train_metrics.R2 * 100).toFixed(2)}%</span>
            </div>

            <div className="glass-card rounded-2xl p-4 text-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">調整後 R²</span>
              <p className="text-2xl font-bold font-mono text-purple-400 mt-1">
                {evalResult.test_metrics.Adj_R2 !== null && evalResult.test_metrics.Adj_R2 !== undefined
                  ? `${(evalResult.test_metrics.Adj_R2 * 100).toFixed(2)}%`
                  : 'N/A'}
              </p>
              <span className="text-[10px] text-slate-500">
                訓練集：{' '}
                {evalResult.train_metrics.Adj_R2 !== null && evalResult.train_metrics.Adj_R2 !== undefined
                  ? `${(evalResult.train_metrics.Adj_R2 * 100).toFixed(2)}%`
                  : 'N/A'}
              </span>
            </div>

            <div className="glass-card rounded-2xl p-4 text-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">均方根誤差（RMSE）</span>
              <p className="text-2xl font-bold font-mono text-pink-400 mt-1">
                {evalResult.test_metrics.RMSE.toFixed(3)}
              </p>
              <span className="text-[10px] text-slate-500">訓練集：{evalResult.train_metrics.RMSE.toFixed(3)}</span>
            </div>

            <div className="glass-card rounded-2xl p-4 text-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">平均絕對誤差（MAE）</span>
              <p className="text-2xl font-bold font-mono text-slate-200 mt-1">
                {evalResult.test_metrics.MAE.toFixed(3)}
              </p>
              <span className="text-[10px] text-slate-500">訓練集：{evalResult.train_metrics.MAE.toFixed(3)}</span>
            </div>

            <div className="glass-card rounded-2xl p-4 text-center col-span-2 sm:col-span-1">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">均方誤差（MSE）</span>
              <p className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                {evalResult.test_metrics.MSE.toFixed(3)}
              </p>
              <span className="text-[10px] text-slate-500">平方誤差懲罰</span>
            </div>
          </div>

          {/* 95% Confidence vs Prediction Interval Chart */}
          <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
            <PlotlyChart data={piPlotData} layout={piPlotLayout} />
            <div className="mt-3 flex items-center gap-2 text-xs text-slate-400 border-t border-slate-800/60 pt-3">
              <Scale className="h-4 w-4 text-indigo-400 shrink-0" />
              <span>
                <strong>統計推論定理：</strong>請注意 95% 預測區間（單一觀測值）必然顯著寬於 95% 信賴區間（母體平均值），原因在於預測單一未來的觀測值時，除了包含參數估計的標準誤外，還必須納入個別隨機誤差項變異 $\sigma^2_\epsilon$。
              </span>
            </div>
          </div>

          {/* Actual vs Pred & Residuals Plots Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
              <PlotlyChart data={actualVsPredPlotData} layout={actualVsPredLayout} />
            </div>

            <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
              <PlotlyChart data={resPlotData} layout={resPlotLayout} />
            </div>
          </div>

          {/* 5 Classical Assumptions Diagnostics Panel */}
          {assumptions && (
            <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-6 shadow-lg">
              <div className="flex items-center gap-2 mb-4">
                <ShieldAlert className="h-5 w-5 text-indigo-400" />
                <h4 className="text-base font-bold text-white">五大經典線性回歸統計假設檢定（Gauss-Markov 定理）</h4>
              </div>

              {/* Assumption Tabs Switch */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 rounded-xl border border-slate-800 bg-slate-900/80 p-1 mb-5">
                {[
                  { key: 'linearity', label: '1. 線性性 (Linearity)' },
                  { key: 'independence', label: '2. 殘差獨立性 (Independence)' },
                  { key: 'homoscedasticity', label: '3. 同質變異性 (Homoscedasticity)' },
                  { key: 'normality', label: '4. 殘差常態性 (Normality)' },
                  { key: 'vif', label: '5. 多重共線性 (VIF)' },
                ].map((item) => (
                  <button
                    key={item.key}
                    onClick={() => setActiveDiagTab(item.key as any)}
                    className={`rounded-lg py-2 text-xs font-semibold transition-all cursor-pointer ${
                      activeDiagTab === item.key
                        ? 'bg-indigo-600 text-white shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              {/* Tab 1: Linearity */}
              {activeDiagTab === 'linearity' && assumptions.linearity && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">{assumptions.linearity.diagnostic}</span>
                    <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-semibold text-emerald-400">
                      檢定方法：{assumptions.linearity.method}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">
                    {assumptions.linearity.metric_name}: <strong className="font-mono text-indigo-400">{String(assumptions.linearity.value)}</strong>
                  </p>
                  <p className="text-xs text-slate-400">{assumptions.linearity.interpretation}</p>
                </div>
              )}

              {/* Tab 2: Independence */}
              {activeDiagTab === 'independence' && assumptions.independence && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">{assumptions.independence.diagnostic}</span>
                    <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-semibold text-emerald-400">
                      檢定方法：{assumptions.independence.method}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">
                    {assumptions.independence.metric_name}: <strong className="font-mono text-indigo-400">{String(assumptions.independence.value)}</strong>（常規獨立無自相關範圍：1.5 ~ 2.5）
                  </p>
                  <p className="text-xs text-slate-400">{assumptions.independence.interpretation}</p>
                  <div className="rounded-xl border border-indigo-500/20 bg-indigo-950/20 p-3 text-xs text-indigo-300">
                    <strong>檢定限制說明：</strong> {assumptions.independence.limitations}
                  </div>
                </div>
              )}

              {/* Tab 3: Homoscedasticity */}
              {activeDiagTab === 'homoscedasticity' && assumptions.homoscedasticity && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">{assumptions.homoscedasticity.diagnostic}</span>
                    <span className="rounded-full bg-indigo-500/10 border border-indigo-500/30 px-3 py-1 text-xs font-semibold text-indigo-400">
                      檢定方法：{assumptions.homoscedasticity.method}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">
                    LM 統計量: <strong className="font-mono text-slate-200">{String(assumptions.homoscedasticity.value)}</strong> | p-value: <strong className="font-mono text-purple-400">{assumptions.homoscedasticity.extra?.p_value !== undefined ? assumptions.homoscedasticity.extra.p_value.toExponential(3) : 'N/A'}</strong>
                  </p>
                  <p className="text-xs text-slate-400">{assumptions.homoscedasticity.interpretation}</p>
                  <p className="text-[11px] text-slate-500 italic">{assumptions.homoscedasticity.limitations}</p>
                </div>
              )}

              {/* Tab 4: Normality */}
              {activeDiagTab === 'normality' && assumptions.normality && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">{assumptions.normality.diagnostic}</span>
                    <span className="rounded-full bg-indigo-500/10 border border-indigo-500/30 px-3 py-1 text-xs font-semibold text-indigo-400">
                      檢定方法：{assumptions.normality.method}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">
                    統計量: <strong className="font-mono text-slate-200">{String(assumptions.normality.value)}</strong> | p-value: <strong className="font-mono text-purple-400">{assumptions.normality.extra?.p_value !== undefined ? assumptions.normality.extra.p_value.toExponential(3) : 'N/A'}</strong>
                  </p>
                  <p className="text-xs text-slate-400">{assumptions.normality.interpretation}</p>
                  <p className="text-[11px] text-slate-500 italic">{assumptions.normality.limitations}</p>
                </div>
              )}

              {/* Tab 5: Multicollinearity (VIF) */}
              {activeDiagTab === 'vif' && assumptions.multicollinearity && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">{assumptions.multicollinearity.diagnostic}</span>
                    <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-semibold text-emerald-400">
                      {assumptions.multicollinearity.metric_name}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">{assumptions.multicollinearity.interpretation}</p>
                  <div className="max-h-56 overflow-y-auto rounded-xl border border-slate-800">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-900/80 text-[10px] uppercase text-slate-400 sticky top-0">
                        <tr>
                          <th className="px-4 py-2">特徵名稱</th>
                          <th className="px-4 py-2">VIF 分數</th>
                          <th className="px-4 py-2">共線性程度評估</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {assumptions.multicollinearity.extra?.vif_table?.map((row: any) => {
                          const feat = row.Feature || row.feature;
                          const vifVal = parseFloat(row.VIF || row.vif || 0);
                          return (
                            <tr key={feat} className="hover:bg-slate-800/30">
                              <td className="px-4 py-2 font-mono text-[11px] text-slate-200">{feat}</td>
                              <td className="px-4 py-2 font-mono font-semibold text-indigo-400">{vifVal.toFixed(2)}</td>
                              <td className="px-4 py-2">
                                {vifVal < 5 ? (
                                  <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">低共線性（&lt; 5）</span>
                                ) : vifVal < 10 ? (
                                  <span className="text-[10px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded">中度共線性（&lt; 10）</span>
                                ) : (
                                  <span className="text-[10px] text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded">嚴重共線性（&gt; 10）</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
