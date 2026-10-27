import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Multiple Linear Regression Dashboard | CRISP-DM & Feature Selection',
  description:
    'Interactive Regression Analysis with CRISP-DM, 5 Feature Selection Methods, Statsmodels 95% Prediction Interval, and Assumption Diagnostics.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#090d16] text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
