'use client';

import React, { useState } from 'react';
import { X, Globe, Upload, ShieldCheck, AlertTriangle } from 'lucide-react';
import { submitCustomDatasetUrl } from '../lib/api';
import { DatasetOverviewResponse } from '../types/api';

interface CustomDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCustomLoaded: (overview: DatasetOverviewResponse) => void;
}

export default function CustomDatasetModal({
  isOpen,
  onClose,
  onCustomLoaded,
}: CustomDatasetModalProps) {
  const [mode, setMode] = useState<'url' | 'upload'>('url');
  const [url, setUrl] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!url.startsWith('https://')) {
      setError('Only secure HTTPS public URLs are allowed (SSRF protection).');
      return;
    }
    setIsLoading(true);
    try {
      const overview = await submitCustomDatasetUrl(url, targetColumn || undefined);
      onCustomLoaded(overview);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to validate and load dataset.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      setError('Only .csv files are supported.');
      return;
    }

    if (file.size > 4.5 * 1024 * 1024) {
      setError('File size exceeds Vercel Serverless Function limit (4.5 MB).');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    if (targetColumn) {
      formData.append('target_column', targetColumn);
    }

    try {
      const res = await fetch('/api/datasets/custom', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || 'Upload failed');
      }
      const overview = await res.json();
      onCustomLoaded(overview);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Upload processing error.');
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
            <h3 className="text-lg font-bold text-white">Load Custom Dataset</h3>
            <p className="text-xs text-slate-400">Bring your own data with SSRF protection</p>
          </div>
        </div>

        {/* Tab switch */}
        <div className="mb-4 grid grid-cols-2 rounded-xl border border-slate-800 bg-slate-900/80 p-1">
          <button
            onClick={() => setMode('url')}
            className={`rounded-lg py-2 text-xs font-semibold transition-all ${
              mode === 'url' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Public HTTPS URL
          </button>
          <button
            onClick={() => setMode('upload')}
            className={`rounded-lg py-2 text-xs font-semibold transition-all ${
              mode === 'upload' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Local CSV Upload
          </button>
        </div>

        {error && (
          <div className="mb-4 flex items-start gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {mode === 'url' ? (
          <form onSubmit={handleUrlSubmit} className="space-y-4">
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-300">CSV Public URL</label>
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
                <span>SSRF Guard active: Private IPs (127.0.0.1, 10.x, 192.168.x) blocked.</span>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-slate-300">
                Target Column Name <span className="text-slate-500">(Optional, defaults to last column)</span>
              </label>
              <input
                type="text"
                placeholder="e.g. Price, Score, Target"
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
                Cancel
              </button>
              <button
                type="submit"
                disabled={isLoading}
                className="rounded-xl bg-indigo-600 px-5 py-2 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 hover:bg-indigo-500 disabled:opacity-50"
              >
                {isLoading ? 'Verifying & Fetching...' : 'Load Dataset'}
              </button>
            </div>
          </form>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-300">Target Column (Optional)</label>
              <input
                type="text"
                placeholder="e.g. Target column name"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full rounded-xl border border-slate-800 bg-slate-900/60 px-3.5 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-indigo-500"
              />
            </div>

            <label className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/40 p-6 text-center cursor-pointer hover:border-indigo-500 hover:bg-slate-900/80 transition-all">
              <Upload className="mb-2 h-8 w-8 text-indigo-400" />
              <span className="text-sm font-medium text-slate-200">
                {isLoading ? 'Uploading & Processing...' : 'Click to select CSV file'}
              </span>
              <span className="mt-1 text-xs text-slate-500">Max size: 4.5 MB (Vercel Limit)</span>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                disabled={isLoading}
                className="hidden"
              />
            </label>
          </div>
        )}
      </div>
    </div>
  );
}
