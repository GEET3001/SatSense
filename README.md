# SatSense (Mempool Intelligence)

A full-stack Bitcoin analytics platform that correlates real-time market sentiment with mempool congestion patterns using a state-of-the-art 2026-regime ML model.

## Overview

SatSense is designed to predict Bitcoin fee trends by analyzing the interplay between social catalysts (news, social media) and on-chain transaction pressure. It features a high-performance FastAPI backend for ML processing and a sleek Next.js dashboard for real-time visualization.

### Key Features
- **2026 Fee Market Modeling:** Calibrated for the post-L2-dominance era where institutional settlement and retail Lightning usage define the fee floor.
- **Sentiment-Congestion Correlation:** Uses FinBERT to score real-time news feeds, combined with the **Fear & Greed Index** and **GitHub developer momentum**, to correlate market mood with mempool depth.
- **Stateful Markov Simulation:** A robust synthetic fallback engine that maintains realistic dashboard activity even when upstream APIs (like mempool.space) are unreachable.
- **Predictive Analytics:** Random Forest models trained on historical 2026 data to project fees for 1, 3, and 6 block confirmation targets.

## Architecture

- **Frontend:** Next.js 14, TailwindCSS, Recharts, Framer Motion.
- **Backend:** FastAPI (Python), FinBERT (Transformers), Scikit-Learn.
- **Database:** Supabase (PostgreSQL) for persistence and time-series snapshots.
- **Automation:** Internal background scheduler for 5-minute periodic snapshots and model retraining.

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- Supabase Account

### 2. Backend Setup (ML)
```bash
cd ml
pip install -r requirements.txt
# Configure .env with SUPABASE_URL and SUPABASE_SERVICE_KEY
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
# Configure .env with NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY
npm run dev -- -p 3001
```

##  ML Logic & 2026 Regime

The system operates on a **Three-Layer Fee Model** calibrated for April 2026:
1. **Filler (1-3 sat/vB):** Baseline demand from Ordinals and low-priority consolidations.
2. **Utility (4-15 sat/vB):** Daily retail activity, exchange withdrawals, and L2 channel funding.
3. **Priority (15-50 sat/vB):** Institutional settlement batches and L2 rollup commits.


### Real-Time Adaptivity
If external APIs block requests, the engine automatically switches to a **Stateful Simulation** that models block mining events every 10 minutes (V-shape mempool patterns) and maintains momentum-based fee transitions to ensure the dashboard remains realistic and alive.

##  Database Schema
- `mempool_snapshots`: Raw on-chain metrics (tx count, size, median fee).
- `sentiment_snapshots`: Scored catalysts and article volume.
- `features`: Integrated feature sets used for ML training.
- `predictions`: Model outputs for future block targets.
- `actuals`: Ground truth data used to measure model accuracy.

---
*Built for the 2026 Bitcoin Economy.*
