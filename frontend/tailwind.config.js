/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark trading terminal palette
        bg: {
          base:    "#0A0B0E",
          surface: "#10131A",
          card:    "#141820",
          border:  "#1E2435",
          hover:   "#1A2030",
        },
        brand: {
          primary:   "#00D4FF",
          secondary: "#6366F1",
          accent:    "#10B981",
        },
        signal: {
          long:    "#10B981",  // green
          short:   "#EF4444",  // red
          flat:    "#6B7280",  // gray
          warning: "#F59E0B",
          danger:  "#EF4444",
        },
        text: {
          primary:   "#E2E8F0",
          secondary: "#94A3B8",
          muted:     "#475569",
          accent:    "#00D4FF",
        },
      },
      fontFamily: {
        sans:   ["Inter", "system-ui", "sans-serif"],
        mono:   ["JetBrains Mono", "Fira Code", "monospace"],
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "grid-pattern":    "linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)",
      },
      animation: {
        "pulse-slow":   "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "glow":         "glow 2s ease-in-out infinite alternate",
        "slide-in":     "slideIn 0.3s ease-out",
        "fade-in":      "fadeIn 0.4s ease-out",
        "number-tick":  "numberTick 0.3s ease-out",
      },
      keyframes: {
        glow: {
          "0%":   { boxShadow: "0 0 5px rgba(0,212,255,0.2)" },
          "100%": { boxShadow: "0 0 20px rgba(0,212,255,0.5)" },
        },
        slideIn: {
          "0%":   { transform: "translateX(-10px)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" },
        },
        fadeIn: {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
        numberTick: {
          "0%":   { transform: "translateY(-4px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
      },
      boxShadow: {
        "card":     "0 4px 24px rgba(0,0,0,0.4)",
        "card-hover": "0 8px 32px rgba(0,212,255,0.1)",
        "glow-sm":  "0 0 10px rgba(0,212,255,0.3)",
        "glow-md":  "0 0 20px rgba(0,212,255,0.4)",
        "green-glow": "0 0 15px rgba(16,185,129,0.4)",
        "red-glow":   "0 0 15px rgba(239,68,68,0.4)",
      },
    },
  },
  plugins: [],
};
