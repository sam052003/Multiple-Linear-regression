'use client';

import React, { useState, useRef } from 'react';
import { X, Globe, Upload, ShieldCheck, AlertTriangle, FileText, CheckCircle2 } from 'lucide-react';
import { submitCustomDatasetUrl, uploadCustomCsv } from '../lib/api';
import { DatasetOverviewResponse } from '../types/api';

interface CustomDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCustomLoaded: (overview: DatasetOverviewResponse, displayName: string) => void;
}

export default function CustomDatasetModal({
  isOpen,
  onClose,
  onCustomLoaded,
}: CustomDatasetModalProps) {
  const [mode, setMode] = useState<'url' | 'upload'>('upload');
  const [url, setUrl] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Upload file state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [detectedColumns, setDetectedColumns] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const extractUrlDisplayName = (rawUrl: string): string => {
    try {
      const u = new URL(rawUrl);
      const parts = u.pathname.split('/').filter(Boolean);
      const last = parts.length > 0 ? parts[parts.length - 1] : '';
      if (last && (last.endsWith('.csv') || last.length > 2)) {
        return decodeURIComponent(last);
      }
    } catch {
      // fallback
    }
    return 'CSV 網址';
  };

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!url.startsWith('https://')) {
      setError('僅支援安全的公開 HTTPS 網址（防範 SSRF 攻擊）。');
      return;
    }
    setIsLoading(true);
    try {
      const overview = await submitCustomDatasetUrl(url, targetColumn || undefined);
      const displayName = extractUrlDisplayName(url);
      onCustomLoaded(overview, displayName);
      onClose();
    } catch (err: any) {
      setError(err.message || '載入與驗證資料集失敗。');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      setError('僅支援 .csv 檔案格式。');
      return;
    }

    if (file.size > 4.5 * 1024 * 1024) {
      setError('檔案大小超過 Vercel Serverless Function 限制（上限 4.5 MB）。');
      return;
    }

    setError(null);
    setSelectedFile(file);

    // Read first 8KB to extract column headers
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const text = (evt.target?.result as string) || '';
        const firstLine = text.split(/\r?\n/)[0];
        if (firstLine) {
          // Parse CSV header simply respecting basic commas
          const cols = firstLine
            .split(',')
            .map((c) => c.trim().replace(/^["']|["']$/g, ''))
            .filter((c) => c.length > 0);
          setDetectedColumns(cols);
        }
      } catch {
        setDetectedColumns([]);
      }
    };
    reader.readAsText(file.slice(0, 8192));
  };

  const handleFileUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('請先選取本機 CSV 檔案。');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const overview = await uploadCustomCsv(selectedFile, targetColumn || undefined);
      onCustomLoaded(overview, selectedFile.name);
      onClose();
    } catch (err: any) {
      setError(err.message || '上傳與處理資料集時發生錯誤。');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md p-4">
      <div className="relative w-full max-w-lg rounded-2xl border border-slate-800 bg-[#0f172a] p-6 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="mb-5 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <Globe className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">載入自訂資料集 (Custom Dataset)</h3>
            <p className="text-xs text-slate-400">支援本機 CSV 上傳與公開 HTTPS 網址，嚴格驗證目標變數</p>
          </div>
        </div>

        {/* Tab switch */}
        <div className="mb-4 grid grid-cols-2 rounded-xl border border-slate-800 bg-slate-900/80 p-1">
          <button
            onClick={() => setMode('upload')}
            className={`rounded-lg py-2 text-xs font-semibold transition-all ${
              mode === 'upload' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            本機 CSV 上傳（推薦）
          </button>
          <button
            onClick={() => setMode('url')}
            className={`rounded-lg py-2 text-xs font-semibold transition-all ${
              mode === 'url' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            公開 HTTPS 網址
          </button>
        </div>

        {error && (
          <div className="mb-4 flex items-start gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {mode === 'upload' ? (
          <form onSubmit={handleFileUploadSubmit} className="space-y-4">
            {!selectedFile ? (
              <label className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/40 p-6 text-center cursor-pointer hover:border-indigo-500 hover:bg-slate-900/80 transition-all">
                <Upload className="mb-2 h-8 w-8 text-indigo-400" />
                <span className="text-sm font-medium text-slate-200">點擊選取 CSV 檔案</span>
                <span className="mt-1 text-xs text-slate-400 font-mono">上限：4.5 MB / 5,000 筆資料</span>
                <span className="mt-0.5 text-[11px] text-slate-500">完整資料集直接載入分析，不進行默默截斷</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  disabled={isLoading}
                  className="hidden"
                />
              </label>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between rounded-xl border border-indigo-500/30 bg-indigo-500/10 p-3 text-xs">
                  <div className="flex items-center gap-2 text-indigo-200 font-medium">
                    <FileText className="h-4 w-4 text-indigo-400" />
                    <span>{selectedFile.name}</span>
                    <span className="text-slate-400 font-mono">({(selectedFile.size / 1024).toFixed(1)} KB)</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedFile(null);
                      setDetectedColumns([]);
                      setTargetColumn('');
                      if (fileInputRef.current) fileInputRef.current.value = '';
                    }}
                    className="text-[11px] text-slate-400 hover:text-rose-400 underline"
                  >
                    重新選取
                  </button>
                </div>

                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-300">
                    預測目標變數 (Target Column)
                  </label>
                  {detectedColumns.length > 0 ? (
                    <select
                      value={targetColumn}
                      onChange={(e) => setTargetColumn(e.target.value)}
                      className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3.5 py-2 text-sm text-slate-100 outline-none focus:border-indigo-500 cursor-pointer"
                    >
                      <option value="">自動推薦最佳數值回歸目標（預設推薦）</option>
                      {detectedColumns.map((col) => (
                        <option key={col} value={col} className="bg-slate-900">
                          {col}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      placeholder="例如：Price, Score, Target（選填，留空自動推薦）"
                      value={targetColumn}
                      onChange={(e) => setTargetColumn(e.target.value)}
                      className="w-full rounded-xl border border-slate-800 bg-slate-900/60 px-3.5 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-indigo-500"
                    />
                  )}
                  <p className="mt-1 text-[11px] text-slate-500">
                    系統將優先推薦適合多元線性回歸之連續數值欄位，避免誤選文字分類標籤。
                  </p>
                </div>

                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={onClose}
                    className="rounded-xl border border-slate-800 px-4 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800"
                  >
                    取消
                  </button>
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="flex items-center gap-1.5 rounded-xl bg-indigo-600 px-5 py-2 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:bg-indigo-500 disabled:opacity-50"
                  >
                    {isLoading ? '正在上傳與驗證...' : '確認上傳並分析'}
                  </button>
                </div>
              </div>
            )}
          </form>
        ) : (
          <form onSubmit={handleUrlSubmit} className="space-y-4">
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-300">CSV 公開網址 (URL)</label>
              <input
                type="url"
                required
                placeholder="https://raw.githubusercontent.com/.../dataset.csv"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="w-full rounded-xl border border-slate-800 bg-slate-900/60 px-3.5 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-indigo-500"
              />
              <div className="mt-1 flex items-center gap-1.5 text-[11px] text-emerald-400/90">
                <ShieldCheck className="h-3.5 w-3.5" />
                <span>SSRF 安全防護已啟動：禁止私有與內部 IP（127.0.0.1、10.x、192.168.x）。</span>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-slate-300">
                目標變數欄位名稱 (Target Column) <span className="text-slate-500">（選填，留空自動推薦）</span>
              </label>
              <input
                type="text"
                placeholder="例如：Price, Score, Target"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full rounded-xl border border-slate-800 bg-slate-900/60 px-3.5 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-indigo-500"
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="rounded-xl border border-slate-800 px-4 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800"
              >
                取消
              </button>
              <button
                type="submit"
                disabled={isLoading}
                className="rounded-xl bg-indigo-600 px-5 py-2 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:bg-indigo-500 disabled:opacity-50"
              >
                {isLoading ? '正在驗證並載入...' : '載入資料集'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
