# SatSense (Mempool Intelligence)

A full-stack Bitcoin analytics platform that correlates real-time market sentiment with mempool congestion patterns using a state-of-the-art 2026-regime ML model.

## Overview

SatSense is designed to predict Bitcoin fee trends by analyzing the interplay between social catalysts (news, social media) and on-chain transaction pressure. It features a high-performance FastAPI backend for ML processing and a sleek Next.js dashboard for real-time visualization.

### Key Features
- **2026 Fee Market Modeling:** Calibrated for the post-L2-dominance era where institutional settlement and retail Lightning usage define the fee floor.
- **Price-Driven Market Sentiment:** Live BTC 24h price momentum is the dominant signal, blended with the **Fear & Greed Index**, **FinBERT-scored news feeds**, and **GitHub developer momentum** — so the sentiment gauge moves the same direction as the market (Bullish when BTC is up, Bearish when it's down).
- **Live On-Chain Data:** Pulls real-time mempool depth and fee dispersion from `mempool.space`, with fee percentiles computed over the *confirmation window* (the transactions actually competing for the next ~6 blocks).
- **Resilient Fallbacks:** Multi-source price feeds (CoinGecko → Coinbase → Kraken → Binance) and a stateful synthetic engine keep the dashboard live even when upstream APIs are blocked or rate-limited.
- **Predictive Analytics:** Random Forest models trained on historical 2026 data to project fees for 1, 3, and 6 block confirmation targets.

## Architecture

- **Frontend:** Next.js 16, TailwindCSS, Recharts, Framer Motion.
- **Backend:** FastAPI (Python), FinBERT via Hugging Face Inference API, Scikit-Learn.
- **Database:** Supabase (PostgreSQL) for persistence and time-series snapshots.
- **Automation:** Internal background scheduler for 5-minute periodic snapshots and model retraining.
- **Data Sources:** `mempool.space` (on-chain), CoinGecko/Coinbase/Kraken/Binance (BTC price), alternative.me (Fear & Greed), RSS + Reddit (news), GitHub (dev momentum).

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 20.9+ (required by Next.js 16)
- Supabase Account

### 2. Backend Setup (ML)
```bash
cd ml
pip install -r requirements.txt
# Configure .env with SUPABASE_URL and SUPABASE_SERVICE_KEY
# Optional: HUGGINGFACE_TOKEN for FinBERT news scoring (falls back to keyword scoring if unset)
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

> **First run:** hit `GET /bootstrap` once to seed Supabase with ~14 days of history and train the initial models. The scheduler then writes a fresh snapshot every 5 minutes.

### 3. Frontend Setup
```bash
cd frontend
npm install
# Configure .env with:
#   NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY  (dashboard reads snapshots)
#   ML_URL                                                   (backend base URL for the /news proxy)
npm run dev -- -p 3001
```

##  ML Logic & 2026 Regime

The system operates on a **Three-Layer Fee Model** calibrated for April 2026:
1. **Filler (1-3 sat/vB):** Baseline demand from Ordinals and low-priority consolidations.
2. **Utility (4-15 sat/vB):** Daily retail activity, exchange withdrawals, and L2 channel funding.
3. **Priority (15-50 sat/vB):** Institutional settlement batches and L2 rollup commits.


### Market Sentiment Engine
Sentiment is a weighted blend of independent signals, normalised to a `-1.0` (max bearish) → `+1.0` (max bullish) scale:

| Signal | Weight | Notes |
|---|---|---|
| BTC 24h price momentum | **5.0** | Dominant. ±4% maps to ±1.0. Multi-source with automatic failover. |
| News (FinBERT / keyword) | 1.0 | Averaged across RSS + Reddit so article *volume* can't dilute direction. |
| Fear & Greed Index | 1.0 | Slow-moving macro confirmation. |
| GitHub dev momentum | 0.5 | Mild long-term tailwind. |

Treating price as the primary driver (rather than averaging hundreds of near-neutral headlines) is what keeps the gauge aligned with the live market.

### Real-Time Adaptivity
On-chain metrics are pulled live from `mempool.space`. If any upstream API blocks or rate-limits requests, the engine degrades gracefully — price feeds fail over across four exchanges, and mempool data falls back to a **Stateful Simulation** that models block mining events every ~10 minutes (V-shape mempool patterns) with momentum-based fee transitions, so the dashboard stays realistic and alive.

##  Backend API
- `GET /health` — service status, loaded models, and active scoring engine.
- `GET /bootstrap` — seed ~14 days of synthetic history, then train models (run once).
- `POST /snapshot` — capture one live mempool + sentiment snapshot (also runs every 5 min via the scheduler).
- `POST /train` — retrain Random Forest models on accumulated data.
- `GET /news` — latest scored catalysts (proxied by the dashboard's `/api/news`).

##  Database Schema
- `mempool_snapshots`: Raw on-chain metrics (tx count, size, median fee).
- `sentiment_snapshots`: Scored catalysts and article volume.
- `features`: Integrated feature sets used for ML training.
- `predictions`: Model outputs for future block targets.
- `actuals`: Ground truth data used to measure model accuracy.

---
*Built for the 2026 Bitcoin Economy.*
