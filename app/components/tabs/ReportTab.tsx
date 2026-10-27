'use client';

import React, { useState, useEffect } from 'react';
import {
  FileText,
  Download,
  FileDown,
  Sparkles,
  ExternalLink,
  BookOpen,
  CheckCircle2,
} from 'lucide-react';
import { fetchMarkdownReport } from '../../lib/api';

export default function ReportTab() {
  const [reportMarkdown, setReportMarkdown] = useState<string>('');
  const [reportTitle, setReportTitle] = useState<string>('Academic Research Report');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchMarkdownReport()
      .then((markdownText) => {
        setReportMarkdown(markdownText);
        setReportTitle('Multiple Linear Regression Analysis Final Report');
      })
      .catch((err) => {
        setReportMarkdown('Failed to load markdown report. Ensure backend is running.');
      })
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <span className="rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-semibold text-indigo-400">
            CRISP-DM Stage 6: Documentation & Deliverables
          </span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-xl font-bold text-white">Comprehensive Analysis Report & Downloads</h3>
            <p className="text-xs text-slate-400">
              Download academic report in PDF, read Markdown synthesis, or export experimental results in CSV format.
            </p>
          </div>

          {/* Download Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <a
              href="/api/reports/pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:from-indigo-500 hover:to-purple-500 transition-all"
            >
              <FileDown className="h-4 w-4" />
              <span>Download Report (PDF)</span>
            </a>

            <a
              href="/api/reports/download-results?type=metrics"
              download
              className="flex items-center gap-1.5 rounded-xl border border-slate-800 bg-slate-900/80 px-3.5 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-800 hover:text-white transition-all"
            >
              <Download className="h-4 w-4" />
              <span>Metrics CSV</span>
            </a>

            <a
              href="/api/reports/download-results?type=features"
              download
              className="flex items-center gap-1.5 rounded-xl border border-slate-800 bg-slate-900/80 px-3.5 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-800 hover:text-white transition-all"
            >
              <Download className="h-4 w-4" />
              <span>Rankings CSV</span>
            </a>
          </div>
        </div>
      </div>

      {/* Report Viewer */}
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-indigo-400" />
            <h4 className="text-sm font-semibold text-white">{reportTitle}</h4>
          </div>
          <span className="text-xs text-slate-500 font-mono">reports/final_report.md</span>
        </div>

        {isLoading ? (
          <div className="flex h-64 items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
              <span className="text-xs text-slate-400">Loading analysis report...</span>
            </div>
          </div>
        ) : (
          <div className="max-h-[600px] overflow-y-auto rounded-xl border border-slate-800/80 bg-slate-950/60 p-5 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre-wrap selection:bg-indigo-500 selection:text-white">
            {reportMarkdown}
          </div>
        )}
      </div>
    </div>
  );
}
