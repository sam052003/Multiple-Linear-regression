'use client';

import React from 'react';
import { Database, Github, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react';
import { DatasetItem } from '../types/api';

interface NavbarProps {
  datasets: DatasetItem[];
  selectedProblem: string;
  onSelectProblem: (id: string) => void;
  isBackendHealthy: boolean;
  onOpenCustomModal: () => void;
  customDatasetInfo?: { id: string; name: string } | null;
}

export default function Navbar({
  datasets,
  selectedProblem,
  onSelectProblem,
  isBackendHealthy,
  onOpenCustomModal,
  customDatasetInfo,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-[#090d16]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-purple-600 to-pink-500 shadow-lg shadow-indigo-500/25">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white sm:text-xl">
                MLR <span className="gradient-text">Studio</span>
              </h1>
              <span className="hidden rounded-full border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-medium text-indigo-300 sm:inline-flex">
                CRISP-DM v2.0
              </span>
            </div>
            <p className="text-xs text-slate-400">多元線性回歸與特徵選擇分析平台</p>
          </div>
        </div>

        {/* Center: Problem Selector */}
        <div className="flex items-center gap-2">
          <div className="flex items-center rounded-xl border border-slate-800 bg-slate-900/80 p-1 shadow-inner">
            <Database className="ml-2 h-4 w-4 text-indigo-400" />
            <select
              value={selectedProblem}
              onChange={(e) => {
                const val = e.target.value;
                if (val === '__custom_modal__') {
                  onOpenCustomModal();
                } else {
                  onSelectProblem(val);
                }
              }}
              className="bg-transparent px-3 py-1 text-sm font-medium text-slate-200 outline-none hover:text-white cursor-pointer"
            >
              <optgroup label="內建資料集問題">
                <option value="50_startups" className="bg-slate-900 text-slate-200">
                  50 Startups — 新創公司利潤預測（預設）
                </option>
                <option value="student_performance" className="bg-slate-900 text-slate-200">
                  Student Performance — 學生成績預測（主要研究）
                </option>
                <option value="flood_probability" className="bg-slate-900 text-slate-200">
                  Flood Probability — 洪水機率預測（比較研究）
                </option>
              </optgroup>
              <optgroup label="自訂資料集">
                {customDatasetInfo && (
                  <option value={customDatasetInfo.id} className="bg-slate-900 text-indigo-300 font-medium">
                    自訂資料集 — {customDatasetInfo.name}
                  </option>
                )}
                <option value="__custom_modal__" className="bg-slate-900 text-indigo-400">
                  ＋ 自訂資料集（CSV 網址／上傳）
                </option>
              </optgroup>
            </select>
          </div>
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-3">
          {/* Health Pill */}
          <div
            className={`hidden items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium md:flex ${
              isBackendHealthy
                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                : 'border-amber-500/30 bg-amber-500/10 text-amber-400'
            }`}
          >
            {isBackendHealthy ? (
              <>
                <CheckCircle2 className="h-3.5 w-3.5" />
                <span>分析服務已連線</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-3.5 w-3.5" />
                <span>正在連接分析服務</span>
              </>
            )}
          </div>

          {/* GitHub Repo */}
          <a
            href="https://github.com/sam052003/Multiple-Linear-regression"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs font-medium text-slate-300 transition-all hover:border-slate-700 hover:bg-slate-800 hover:text-white"
          >
            <Github className="h-4 w-4" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
}
