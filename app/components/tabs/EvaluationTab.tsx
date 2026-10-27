'use client';

import React, { useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Activity,
  Layers,
  ShieldAlert,
  Play,
  Info,
  Scale,
} from 'lucide-react';
import {
  DatasetOverviewResponse,
  EvaluationResponse,
  FeatureSelectionMethodResult,
} from '../../types/api';
import { runEvaluation } from '../../lib/api';
import PlotlyChart from '../PlotlyChart';

interface EvaluationTabProps {
  overview: DatasetOverviewResponse | null;
  evalResult: EvaluationResponse | null;
  selectedMethod: FeatureSelectionMethodResult | null;
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
        overview.dataset_id,
        overview.target_column,
        selectedMethod?.selected_features
      );
      onEvalComplete(res);
    } catch (err: any) {
      setError(err.message || 'Evaluation computation failed');
    } finally {
      setIsLoading(false);
    }
  };

  // 1. Actual vs Predicted Plot
  const actualVsPredPlotData = evalResult
    ? [
        {
          x: evalResult.actual_vs_predicted.actual,
          y: evalResult.actual_vs_predicted.predicted,
          mode: 'markers',
          type: 'scatter',
          name: 'Observations',
          marker: { color: 'rgba(99, 102, 241, 0.7)', size: 6 },
        },
        {
          x: [
            Math.min(...evalResult.actual_vs_predicted.actual),
            Math.max(...evalResult.actual_vs_predicted.actual),
          ],
          y: [
            Math.min(...evalResult.actual_vs_predicted.actual),
            Math.max(...evalResult.actual_vs_predicted.actual),
          ],
          mode: 'lines',
          type: 'scatter',
          name: 'Ideal (y = x)',
          line: { color: '#ec4899', dash: 'dash', width: 2 },
        },
      ]
    : [];

  const actualVsPredLayout = {
    title: { text: 'Actual vs. Predicted Values', font: { size: 14, color: '#f1f5f9' } },
    xaxis: { title: 'Actual Target Values' },
    yaxis: { title: 'Predicted Values' },
    height: 380,
  };

  // 2. 95% Confidence & Prediction Intervals Plot
  // Sort by predicted values to create a smooth band plot
  const piData = evalResult?.prediction_intervals;
  const piPlotData = (() => {
    if (!piData || piData.y_pred.length === 0) return [];

    // Take top 60 sorted samples for clear visualization
    const indices = Array.from(Array(piData.y_pred.length).keys())
      .sort((a, b) => piData.y_pred[a] - piData.y_pred[b])
      .slice(0, 60);

    const x = indices.map((_, i) => i + 1);
    const yPred = indices.map((i) => piData.y_pred[i]);
    const yTrue = indices.map((i) => piData.y_true[i]);
    const piLow = indices.map((i) => piData.pi_lower[i]);
    const piUp = indices.map((i) => piData.pi_upper[i]);
    const ciLow = indices.map((i) => piData.ci_lower[i]);
    const ciUp = indices.map((i) => piData.ci_upper[i]);

    return [
      // 95% PI Upper & Lower Band
      {
        x: [...x, ...[...x].reverse()],
        y: [...piUp, ...[...piLow].reverse()],
        fill: 'toself',
        fillcolor: 'rgba(99, 102, 241, 0.12)',
        line: { color: 'transparent' },
        name: '95% Prediction Interval (Individual)',
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
        name: '95% Confidence Interval (Mean)',
        showlegend: true,
        type: 'scatter',
      },
      // Predicted line
      {
        x: x,
        y: yPred,
        mode: 'lines',
        name: 'Fitted Mean E(y|X)',
        line: { color: '#818cf8', width: 2.5 },
        type: 'scatter',
      },
      // Actual points
      {
        x: x,
        y: yTrue,
        mode: 'markers',
        name: 'Actual Observation (y)',
        marker: { color: '#38bdf8', size: 5 },
        type: 'scatter',
      },
    ];
  })();

  const piPlotLayout = {
    title: {
      text: '95% Confidence Interval (Mean) vs. 95% Prediction Interval (Observation)',
      font: { size: 14, color: '#f1f5f9' },
    },
    xaxis: { title: 'Test Observations (Ranked by Predicted Value)' },
    yaxis: { title: overview.target_column },
    height: 400,
    legend: { orientation: 'h', y: 1.15, x: 0 },
  };

  // 3. Residuals vs Fitted Plot
  const resPlotData = evalResult
    ? [
        {
          x: evalResult.residuals_analysis.fitted,
          y: evalResult.residuals_analysis.residuals,
          mode: 'markers',
          type: 'scatter',
          name: 'Residuals',
          marker: { color: 'rgba(236, 72, 153, 0.7)', size: 5 },
        },
        {
          x: [
            Math.min(...evalResult.residuals_analysis.fitted),
            Math.max(...evalResult.residuals_analysis.fitted),
          ],
          y: [0, 0],
          mode: 'lines',
          name: 'Zero Residual',
          line: { color: '#ffffff', width: 1.5, dash: 'dot' },
        },
      ]
    : [];

  const resPlotLayout = {
    title: { text: 'Residuals vs. Fitted Values', font: { size: 14, color: '#f1f5f9' } },
    xaxis: { title: 'Fitted (Predicted) Values' },
    yaxis: { title: 'Residuals (e = y - y_hat)' },
    height: 380,
  };

  const diag = evalResult?.assumption_diagnostics;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
            CRISP-DM Stage 5: Evaluation & Statistical Diagnostics
          </span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-xl font-bold text-white">Model Evaluation & Assumption Diagnostics</h3>
            <p className="text-xs text-slate-400">
              Assessing predictive accuracy, 95% Prediction Interval coverage, and validating 5 Gauss-Markov assumptions.
            </p>
          </div>

          <button
            onClick={handleRunEvaluation}
            disabled={isLoading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 shrink-0"
          >
            {isLoading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                <span>Computing OLS & Assumptions...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" />
                <span>{evalResult ? 'Re-run Evaluation' : 'Run Evaluation Diagnostics'}</span>
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
          {evalResult.post_selection_warning && (
            <div className="flex items-start gap-3 rounded-2xl border border-indigo-500/30 bg-indigo-950/20 p-4 text-xs text-indigo-200">
              <Info className="h-5 w-5 shrink-0 text-indigo-400 mt-0.5" />
              <div>
                <span className="font-semibold text-white">Post-Selection Inference Methodology Notice:</span>
                <p className="mt-0.5 text-slate-300">{evalResult.post_selection_warning}</p>
              </div>
            </div>
          )}

          {/* Metric KPI Cards (Train vs Test) */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="glass-card rounded-2xl p-4 text-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Test R²</span>
              <p className="text-2xl font-bold font-mono text-indigo-400 mt-1">
                {(evalResult.metrics.test.R2 * 100).toFixed(2)}%
              </p>
              <span className="text-[10px] text-slate-500">Train: {(evalResult.metrics.train.R2 * 100).toFixed(2)}%</span>
            </div>

            <div className="glass-card rounded-2xl p-4 text-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Adjusted R²</span>
              <p className="text-2xl font-bold font-mono text-purple-400 mt-1">
                {(evalResult.metrics.test.Adj_R2 * 100).toFixed(2)}%
              </p>
              <span className="text-[10px] text-slate-500">Train: {(evalResult.metrics.train.Adj_R2 * 100).toFixed(2)}%</span>
            </div>

            <div className="glass-card rounded-2xl p-4 text-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Test RMSE</span>
              <p className="text-2xl font-bold font-mono text-pink-400 mt-1">
                {evalResult.metrics.test.RMSE.toFixed(3)}
              </p>
              <span className="text-[10px] text-slate-500">Train: {evalResult.metrics.train.RMSE.toFixed(3)}</span>
            </div>

            <div className="glass-card rounded-2xl p-4 text-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Test MAE</span>
              <p className="text-2xl font-bold font-mono text-slate-200 mt-1">
                {evalResult.metrics.test.MAE.toFixed(3)}
              </p>
              <span className="text-[10px] text-slate-500">Train: {evalResult.metrics.train.MAE.toFixed(3)}</span>
            </div>

            <div className="glass-card rounded-2xl p-4 text-center col-span-2 sm:col-span-1">
              <span className="text-[10px] uppercase tracking-wider text-slate-400">Test MSE</span>
              <p className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                {evalResult.metrics.test.MSE.toFixed(3)}
              </p>
              <span className="text-[10px] text-slate-500">Squared error</span>
            </div>
          </div>

          {/* 95% Confidence vs Prediction Interval Chart */}
          <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
            <PlotlyChart data={piPlotData} layout={piPlotLayout} />
            <div className="mt-3 flex items-center gap-2 text-xs text-slate-400 border-t border-slate-800/60 pt-3">
              <Scale className="h-4 w-4 text-indigo-400 shrink-0" />
              <span>
                <strong>Statistical Theorem:</strong> Notice that the 95% Prediction Interval (Observation) is strictly wider than the 95% Confidence Interval (Mean) due to the individual random shock term $\sigma^2_\epsilon$.
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
          {diag && (
            <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-6 shadow-lg">
              <div className="flex items-center gap-2 mb-4">
                <ShieldAlert className="h-5 w-5 text-indigo-400" />
                <h4 className="text-base font-bold text-white">5 Classical Linear Regression Assumptions (Gauss-Markov)</h4>
              </div>

              {/* Assumption Tabs Switch */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 rounded-xl border border-slate-800 bg-slate-900/80 p-1 mb-5">
                {[
                  { key: 'linearity', label: '1. Linearity' },
                  { key: 'independence', label: '2. Independence' },
                  { key: 'homoscedasticity', label: '3. Homoscedasticity' },
                  { key: 'normality', label: '4. Normality' },
                  { key: 'vif', label: '5. Multicollinearity' },
                ].map((item) => (
                  <button
                    key={item.key}
                    onClick={() => setActiveDiagTab(item.key as any)}
                    className={`rounded-lg py-2 text-xs font-semibold transition-all ${
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
              {activeDiagTab === 'linearity' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">Residuals vs. Fitted Orthogonality Check</span>
                    <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-semibold text-emerald-400">
                      {diag.linearity.status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">
                    Correlation between Residuals and Fitted Values: <strong className="font-mono text-indigo-400">{diag.linearity.correlation_residuals_fitted?.toFixed(4)}</strong>
                  </p>
                  <p className="text-xs text-slate-400">{diag.linearity.interpretation}</p>
                </div>
              )}

              {/* Tab 2: Independence */}
              {activeDiagTab === 'independence' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">Durbin-Watson Test for Autocorrelation</span>
                    <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-semibold text-emerald-400">
                      {diag.independence.status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">
                    Durbin-Watson Statistic (DW): <strong className="font-mono text-indigo-400">{diag.independence.statistic?.toFixed(3)}</strong> (Expected range: 1.5 ~ 2.5 for independence)
                  </p>
                  <p className="text-xs text-slate-400">{diag.independence.interpretation}</p>
                  <div className="rounded-xl border border-indigo-500/20 bg-indigo-950/20 p-3 text-xs text-indigo-300">
                    <strong>Note:</strong> {diag.independence.note}
                  </div>
                </div>
              )}

              {/* Tab 3: Homoscedasticity */}
              {activeDiagTab === 'homoscedasticity' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">Breusch-Pagan Test for Constant Variance</span>
                    <span className="rounded-full bg-indigo-500/10 border border-indigo-500/30 px-3 py-1 text-xs font-semibold text-indigo-400">
                      {diag.homoscedasticity.status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">
                    LM Statistic: <strong className="font-mono text-slate-200">{diag.homoscedasticity.statistic?.toFixed(3)}</strong> | p-value: <strong className="font-mono text-purple-400">{diag.homoscedasticity.p_value?.toExponential(3)}</strong>
                  </p>
                  <p className="text-xs text-slate-400">{diag.homoscedasticity.interpretation}</p>
                </div>
              )}

              {/* Tab 4: Normality */}
              {activeDiagTab === 'normality' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">Jarque-Bera Residual Normality Test</span>
                    <span className="rounded-full bg-indigo-500/10 border border-indigo-500/30 px-3 py-1 text-xs font-semibold text-indigo-400">
                      {diag.normality.status}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-slate-300">
                    <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">JB Statistic</span>
                      <strong className="font-mono">{diag.normality.statistic?.toFixed(2)}</strong>
                    </div>
                    <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">p-value</span>
                      <strong className="font-mono">{diag.normality.p_value?.toExponential(2)}</strong>
                    </div>
                    <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Skewness</span>
                      <strong className="font-mono">{diag.normality.skewness?.toFixed(3)}</strong>
                    </div>
                    <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Kurtosis</span>
                      <strong className="font-mono">{diag.normality.kurtosis?.toFixed(3)}</strong>
                    </div>
                  </div>
                  <p className="text-xs text-slate-400">{diag.normality.interpretation}</p>
                </div>
              )}

              {/* Tab 5: Multicollinearity (VIF) */}
              {activeDiagTab === 'vif' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-semibold text-white">Variance Inflation Factor (VIF)</span>
                    <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-semibold text-emerald-400">
                      Max VIF: {diag.multicollinearity.max_vif?.toFixed(2)}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">{diag.multicollinearity.interpretation}</p>
                  <div className="max-h-56 overflow-y-auto rounded-xl border border-slate-800">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-900/80 text-[10px] uppercase text-slate-400 sticky top-0">
                        <tr>
                          <th className="px-4 py-2">Feature</th>
                          <th className="px-4 py-2">VIF Score</th>
                          <th className="px-4 py-2">Assessment</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {diag.multicollinearity.vif_table.map((row) => (
                          <tr key={row.feature} className="hover:bg-slate-800/30">
                            <td className="px-4 py-2 font-mono text-[11px] text-slate-200">{row.feature}</td>
                            <td className="px-4 py-2 font-mono font-semibold text-indigo-400">{row.vif.toFixed(2)}</td>
                            <td className="px-4 py-2">
                              {row.vif < 5 ? (
                                <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">Low (&lt; 5)</span>
                              ) : row.vif < 10 ? (
                                <span className="text-[10px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded">Moderate (&lt; 10)</span>
                              ) : (
                                <span className="text-[10px] text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded">Severe (&gt; 10)</span>
                              )}
                            </td>
                          </tr>
                        ))}
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
