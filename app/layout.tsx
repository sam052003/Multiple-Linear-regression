import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'MLR Studio — 多元線性回歸與特徵選擇分析平台',
  description:
    '依循 CRISP-DM 流程之多元線性回歸（Multiple Linear Regression）互動分析平台，提供特徵選擇比較、統計假設診斷、95% 預測區間與即時預測應用。',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-Hant-TW" className="dark">
      <body className="min-h-screen bg-[#090d16] text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
