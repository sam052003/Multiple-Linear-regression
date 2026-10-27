'use client';

import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import CustomDatasetModal from './components/CustomDatasetModal';
import OverviewTab from './components/tabs/OverviewTab';
import DataUnderstandingTab from './components/tabs/DataUnderstandingTab';
import DataPreparationTab from './components/tabs/DataPreparationTab';
import FeatureSelectionTab from './components/tabs/FeatureSelectionTab';
import EvaluationTab from './components/tabs/EvaluationTab';
import PredictionTab from './components/tabs/PredictionTab';
import ReportTab from './components/tabs/ReportTab';

import {
  DatasetItem,
  DatasetOverviewResponse,
  TrainResponse,
  FeatureSelectionResponse,
  MethodComparisonItem,
  EvaluationResponse,
} from './types/api';
import { fetchDatasets, fetchDatasetOverview, fetchHealth } from './lib/api';
import {
  BookOpen,
  PieChart,
  Sliders,
  Sparkles,
  Activity,
  Calculator,
  FileText,
} from 'lucide-react';

export default function DashboardPage() {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [selectedProblem, setSelectedProblem] = useState<string>('50_startups'); // Default as specified
  const [activeTab, setActiveTab] = useState<string>('overview');

  const [overview, setOverview] = useState<DatasetOverviewResponse | null>(null);
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [fsResult, setFsResult] = useState<FeatureSelectionResponse | null>(null);
  const [evalResult, setEvalResult] = useState<EvaluationResponse | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<MethodComparisonItem | null>(null);

  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean>(true);
  const [isCustomModalOpen, setIsCustomModalOpen] = useState<boolean>(false);
  const [isLoadingOverview, setIsLoadingOverview] = useState<boolean>(false);

  // Check health and load datasets list
  useEffect(() => {
    fetchHealth()
      .then(() => setIsBackendHealthy(true))
      .catch(() => setIsBackendHealthy(false));

    fetchDatasets()
      .then((data) => setDatasets(data))
      .catch((err) => console.error('Failed to load datasets:', err));
  }, []);

  // Fetch overview whenever selectedProblem changes
  useEffect(() => {
    if (!selectedProblem || selectedProblem.startsWith('custom_')) return;
    setIsLoadingOverview(true);
    // Clear dependent states to prevent state leakage between problems
    setTrainResult(null);
    setFsResult(null);
    setEvalResult(null);
    setSelectedMethod(null);

    fetchDatasetOverview(selectedProblem)
      .then((data) => setOverview(data))
      .catch((err) => console.error('Failed to load overview:', err))
      .finally(() => setIsLoadingOverview(false));
  }, [selectedProblem]);

  const handleCustomLoaded = (customOverview: DatasetOverviewResponse) => {
    setOverview(customOverview);
    setSelectedProblem(customOverview.id);
    setTrainResult(null);
    setFsResult(null);
    setEvalResult(null);
    setSelectedMethod(null);
    setActiveTab('overview');
  };

  const tabs = [
    { id: 'overview', label: '1. Business Understanding', icon: BookOpen },
    { id: 'understanding', label: '2. Data Understanding', icon: PieChart },
    { id: 'preparation', label: '3. Data Preparation', icon: Sliders },
    { id: 'feature_selection', label: '4. Feature Selection', icon: Sparkles },
    { id: 'evaluation', label: '5. Evaluation & Diagnostics', icon: Activity },
    { id: 'prediction', label: '6. Interactive Prediction', icon: Calculator },
    { id: 'report', label: 'Reports & Downloads', icon: FileText },
  ];

  return (
    <div className="flex min-h-screen flex-col bg-[#090d16]">
      {/* Top Navbar */}
      <Navbar
        datasets={datasets}
        selectedProblem={selectedProblem}
        onSelectProblem={(id) => setSelectedProblem(id)}
        isBackendHealthy={isBackendHealthy}
        onOpenCustomModal={() => setIsCustomModalOpen(true)}
      />

      {/* Main Container */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        {/* CRISP-DM Navigation Bar */}
        <div className="mb-6 overflow-x-auto pb-2">
          <nav className="flex min-w-max space-x-2 rounded-2xl border border-slate-800 bg-[#0f172a]/70 p-1.5 shadow-lg backdrop-blur-md">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-xs font-semibold transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-md shadow-indigo-600/25'
                      : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                  }`}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content Rendering */}
        <div className="min-h-[500px]">
          {isLoadingOverview ? (
            <div className="flex h-72 items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
                <span className="text-sm font-medium text-slate-400">Loading Problem Dataset...</span>
              </div>
            </div>
          ) : (
            <>
              {activeTab === 'overview' && (
                <OverviewTab
                  overview={overview}
                  onNavigateToTab={(target) => {
                    if (target === 'student_performance_trigger') {
                      setSelectedProblem('student_performance');
                    } else {
                      setActiveTab(target);
                    }
                  }}
                />
              )}

              {activeTab === 'understanding' && <DataUnderstandingTab overview={overview} />}

              {activeTab === 'preparation' && (
                <DataPreparationTab
                  overview={overview}
                  trainResult={trainResult}
                  onTrainComplete={(res) => setTrainResult(res)}
                  onNavigateToTab={(target) => setActiveTab(target)}
                />
              )}

              {activeTab === 'feature_selection' && (
                <FeatureSelectionTab
                  overview={overview}
                  fsResult={fsResult}
                  onFsComplete={(res) => setFsResult(res)}
                  onSelectMethodForEval={(method) => setSelectedMethod(method)}
                  onNavigateToTab={(target) => setActiveTab(target)}
                />
              )}

              {activeTab === 'evaluation' && (
                <EvaluationTab
                  overview={overview}
                  evalResult={evalResult}
                  selectedMethod={selectedMethod}
                  onEvalComplete={(res) => setEvalResult(res)}
                />
              )}

              {activeTab === 'prediction' && <PredictionTab overview={overview} />}

              {activeTab === 'report' && <ReportTab />}
            </>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-800/80 bg-[#090d16] py-6 text-center text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <p>
            Multiple Linear Regression Dashboard — CRISP-DM Analysis Engine.
          </p>
          <p className="mt-1 text-[11px] text-slate-600">
            Powered by Next.js 14, TypeScript, Tailwind CSS, Plotly.js & FastAPI (Python Machine Learning Core)
          </p>
        </div>
      </footer>

      {/* Custom Dataset Modal */}
      <CustomDatasetModal
        isOpen={isCustomModalOpen}
        onClose={() => setIsCustomModalOpen(false)}
        onCustomLoaded={handleCustomLoaded}
      />
    </div>
  );
}
