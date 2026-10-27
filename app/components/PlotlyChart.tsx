'use client';

import React from 'react';
import dynamic from 'next/dynamic';

// Dynamically import Plot with ssr: false to prevent Next.js server evaluation
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => (
    <div className="flex h-64 w-full items-center justify-center rounded-xl border border-slate-800 bg-slate-900/50">
      <div className="flex flex-col items-center gap-2">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
        <span className="text-xs text-slate-400">Loading interactive chart...</span>
      </div>
    </div>
  ),
});

interface PlotlyChartProps {
  data: any[];
  layout?: any;
  config?: any;
  style?: React.CSSProperties;
  className?: string;
}

export default function PlotlyChart({
  data,
  layout,
  config,
  style = { width: '100%', height: '100%' },
  className = '',
}: PlotlyChartProps) {
  // Default dark theme layout for sleek aesthetic
  const defaultLayout = {
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15, 23, 42, 0.4)',
    font: {
      color: '#94a3b8',
      family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      size: 12,
    },
    margin: { l: 50, r: 30, t: 40, b: 50 },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.06)',
      zerolinecolor: 'rgba(255, 255, 255, 0.1)',
    },
    yaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.06)',
      zerolinecolor: 'rgba(255, 255, 255, 0.1)',
    },
    ...layout,
  };

  const defaultConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: 'mlr_chart',
      height: 600,
      width: 900,
      scale: 2,
    },
    ...config,
  };

  return (
    <div className={`relative w-full overflow-hidden rounded-xl ${className}`}>
      <Plot data={data} layout={defaultLayout} config={defaultConfig} style={style} useResizeHandler />
    </div>
  );
}
