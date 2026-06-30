# SatSense (Mempool Intelligence)

A full-stack Bitcoin analytics platform that correlates real-time market sentiment with mempool congestion patterns using a 2026-regime ML model.

## Overview

SatSense predicts Bitcoin fee trends by analysing the interplay between social catalysts (news, social media) and on-chain transaction pressure. It features a high-performance FastAPI backend for ML processing and a Next.js dashboard for real-time visualisation, with two independent sentiment gauges and an impact-ranked news feed.

### Key Features
- **Price-Driven Market Mood:** Live BTC 24h price momentum is the dominant signal, blended with the Fear & Greed Index, FinBERT-scored news, and GitHub developer momentum — so the Market mood gauge moves in the same direction as the live market (Bullish when BTC is up, Bearish when it's down).
- **Dual Sentiment Gauges:** *Market mood* (price + macro blend) and *News tone* (average headline sentiment) are displayed separately so you can see when headlines diverge from price action.
- **Hybrid News Scoring:** FinBERT (ProsusAI/finbert via Hugging Face Inference API) scores every headline, then a **crypto-domain lexicon** corrects for cases FinBERT misreads — e.g. "roundtrips 900 days" or "lower high". Single tokens and multi-word phrases (give back, sell off, death cross, all time high, etc.) are matched. A keyword-only engine serves as fallback when no HF token is set.
- **Impact-Ranked News Feed:** Each headline gets an `impact` score (tone magnitude × source credibility). The feed surfaces the most market-moving items first. Reddit is slightly up-weighted (1.3×) versus RSS (1.0×).
- **Clean RSS Text:** Publisher suffixes appended by feeds (e.g. "Bitcoin Magazine" at the end of titles) are stripped automatically. Summaries that merely echo the headline are dropped so card text is never repetitive.
- **Live On-Chain Data:** Pulls real-time mempool depth and fee dispersion from `mempool.space`, with percentiles computed over the *confirmation window* (the transactions actually competing for the next ~6 blocks, not the full dust-floor backlog).
- **Resilient Fallbacks:** Multi-source price feeds (CoinGecko → Coinbase → Kraken → Binance) and a stateful synthetic mempool engine keep the dashboard live even when upstream APIs are blocked or rate-limited.
- **Predictive Analytics:** Random Forest models trained on historical 2026 data to project fees for 1, 3, and 6 block confirmation targets.

## Architecture

- **Frontend:** Next.js 16, TailwindCSS, Recharts, Framer Motion, Jotai.
- **Backend:** FastAPI (Python), FinBERT via Hugging Face Inference API, Scikit-Learn.
- **Database:** Supabase (PostgreSQL) for persistence and time-series snapshots.
- **Automation:** Internal background scheduler for 5-minute periodic snapshots and model retraining.
- **Data Sources:** `mempool.space` (on-chain), CoinGecko/Coinbase/Kraken/Binance (BTC price), alternative.me (Fear & Greed), RSS + Reddit (news), GitHub (dev momentum).

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 20.9+ (required by Next.js 16)
- Supabase account

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
# Configure .env.local with:
#   NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY  (dashboard reads snapshots)
#   ML_URL                                                   (backend base URL for the /news proxy)
npm run dev -- -p 3001
```

## ML Logic & 2026 Regime

The system operates on a **Three-Layer Fee Model** calibrated for 2026:
1. **Filler (1–4 sat/vB):** Baseline demand from Ordinals/Runes and low-priority consolidations.
2. **Utility (4–16 sat/vB):** Daily retail activity, exchange withdrawals, and L2 channel funding.
3. **Priority (16–50 sat/vB):** Institutional settlement batches and L2 rollup commits.

### Market Sentiment Engine

Sentiment is a weighted blend of independent signals, normalised to `−1.0` (max bearish) → `+1.0` (max bullish):

| Signal | Weight | Notes |
|---|---|---|
| BTC 24h price momentum | **5.0** | Dominant. ±4 % day maps to ±1.0. Fails over across 4 exchanges. |
| News (FinBERT + lexicon) | 1.0 | Averaged across RSS + Reddit (source-credibility weighted). |
| Fear & Greed Index | 1.0 | Slow-moving macro confirmation from alternative.me. |
| GitHub dev momentum | 0.5 | Mild long-term tailwind from Bitcoin Core commit rate. |

The blend uses a **weighted component model** — all signals contribute through their weight, not via averaging raw counts. This prevents a flood of near-neutral headlines from drowning the price signal.

### News Scoring Pipeline

```
Headline text
    │
    ├─ FinBERT (HF Inference API)  → P(positive) − P(negative)
    │
    └─ Crypto lexicon              → weighted token + phrase match
              │                         (tanh saturation)
              │
              └─ _blend_finbert_keyword()
                    │ Lexicon confident (|score| ≥ 0.4) AND disagrees with FinBERT?
                    │   → 70 % lexicon + 30 % FinBERT
                    │ Otherwise:
                    └─ → 60 % FinBERT + 40 % lexicon
```

When `HUGGINGFACE_TOKEN` is unset, the crypto lexicon runs alone.

### Fee Cluster Assignment

Consistent thresholds used across live data, bootstrap, and the KMeans fallback:

| Cluster | Range | Label |
|---|---|---|
| 0 | < 4 sat/vB | Filler floor |
| 1 | 4–9 sat/vB | Economy / utility |
| 2 | 9–16 sat/vB | Normal |
| 3 | 16–30 sat/vB | Priority |
| 4 | > 30 sat/vB | Urgent / L2 spike |

### Real-Time Adaptivity
On-chain metrics are pulled live from `mempool.space`. If any upstream API blocks or rate-limits requests, the engine degrades gracefully — price feeds fail over across four exchanges, and mempool data falls back to a **Stateful Simulation** that models block mining events every ~10 minutes (V-shape mempool patterns) with momentum-based fee transitions, so the dashboard stays realistic and live.

## Backend API
- `GET /health` — service status, loaded models, and active scoring engine.
- `GET /bootstrap` — seed ~14 days of synthetic history, then train models (run once).
- `POST /snapshot` — capture one live mempool + sentiment snapshot (also runs every 5 min via the scheduler).
- `POST /train` — retrain Random Forest models on accumulated data.
- `GET /news` — latest impact-ranked scored catalysts (proxied by the dashboard's `/api/news`).

## Database Schema
- `mempool_snapshots`: Raw on-chain metrics (tx count, size, median fee, p10/p90 fee rates).
- `sentiment_snapshots`: Scored catalysts and article volume.
- `features`: Integrated feature sets used for ML training (linked to both snapshot tables).
- `predictions`: Model outputs for 1, 3, and 6 block confirmation targets.
- `actuals`: Ground truth data used to measure model accuracy and trigger retraining.

---
*Built for the 2026 Bitcoin Economy.*
