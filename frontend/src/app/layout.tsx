import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeltaRL Trader — RL-Powered Crypto Trading Bot",
  description:
    "Autonomous reinforcement-learning trading bot for Delta Exchange India perpetual futures. Demo/Live mode with real-time equity curves, training controls, and risk management.",
  keywords: ["crypto", "trading bot", "reinforcement learning", "Delta Exchange", "perpetual futures"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
      </head>
      <body className="bg-bg-base text-text-primary antialiased">
        {children}
      </body>
    </html>
  );
}
