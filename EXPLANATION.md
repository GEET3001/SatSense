# SatSense — System Architecture & Technical Explanation

This document is a deep-dive into how SatSense works end-to-end: how data flows from raw sources into the ML models, how sentiment is scored, and how the dashboard reflects the live market.

---

## 1. High-Level System Overview

The system has three layers: **Data Ingestion**, **ML Processing**, and **Dashboard Visualisation**.

```
┌─────────────────────────────────────────────────────────┐
│                   External Data Sources                  │
│  RSS Feeds · Reddit · Fear & Greed · GitHub · BTC Price │
│                    mempool.space                         │
└────────────────────────┬────────────────────────────────┘
                         │  every 5 minutes
                         ▼
┌─────────────────────────────────────────────────────────┐
│                FastAPI ML Backend (Render)               │
│                                                         │
│  fetch_sentiment_data()   fetch_mempool_data()          │
│          │                        │                     │
│          └──────────┬─────────────┘                     │
│                     ▼                                   │
│             Feature Vector (15 features)                │
│                     │                                   │
│            Random Forest Regressors                     │
│         (rf_1block · rf_3block · rf_6block)             │
│                     │                                   │
│            snapshot_endpoint() → Supabase               │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               Supabase (PostgreSQL)                      │
│  mempool_snapshots · sentiment_snapshots                 │
│  features · predictions · actuals                        │
└────────────────────────┬────────────────────────────────┘
                         │  REST + Realtime WebSocket
                         ▼
┌─────────────────────────────────────────────────────────┐
│            Next.js 16 Dashboard (Vercel)                 │
│  /api/latest → Supabase read                            │
│  /api/news   → ML backend /news proxy                   │
│  Jotai atoms · Recharts · Framer Motion                 │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Data Ingestion Pipeline

Every 5 minutes an `asyncio` background scheduler fires `snapshot_endpoint()`, which in parallel runs:

### 2a. Mempool Data (`fetch_mempool_data`)

Tries the live `mempool.space` API first:
- Fetches `/api/mempool` — returns tx count, total vsize, and a fee histogram `[[feerate, vsize], ...]`
- Computes percentiles over the **confirmation window** (top 6,000,000 vB of the backlog — the transactions actually competing for the next ~6 blocks). Using the whole backlog collapses to the 1 sat/vB dust floor and carries no signal.
- Derives: `median_fee_rate`, `p10`, `p90`, `fee_iqr = p90 − p10`, `avg_tx_size_bytes`, `high_fee_pct`

If `mempool.space` is unreachable, a **Stateful Simulation** fallback runs:
- Session-aware fee layer sampling: quiet (04–08 UTC), US open (12–16), peak (18–22), Asian (00–04)
- Three fee layers per session with calibrated probability weights: Filler (1–4 sat/vB), Utility (4–16 sat/vB), Priority (16–50 sat/vB)
- Block mining events every ~10 minutes drain ~2,500 transactions (Poisson-distributed)
- EMA-style state transitions: `cur_fee += (target − cur_fee) × inertia` — produces realistic V-shape mempool patterns between blocks
- Weekend shifts: probability weight moves toward Filler (lower institutional activity)

Fee cluster assignment is consistent across live, synthetic, and bootstrap data via a single `_assign_fee_cluster()` helper:

| Cluster | Range | Label |
|---|---|---|
| 0 | < 4 sat/vB | Filler floor |
| 1 | 4–9 sat/vB | Economy / utility |
| 2 | 9–16 sat/vB | Normal |
| 3 | 16–30 sat/vB | Priority |
| 4 | > 30 sat/vB | Urgent / L2 spike |

A KMeans model (5 clusters) is used when loaded; `_assign_fee_cluster()` is the rule-based fallback.

### 2b. Sentiment Data (`fetch_sentiment_data`)

Runs four sub-fetches concurrently inside a shared `httpx.AsyncClient`:

**RSS News (7 feeds, sync via executor)**
- Feeds: CoinDesk, CoinTelegraph, Bitcoin Magazine, Decrypt, Bitcoinist, NewsBTC, CryptoSlate
- `_clean_rss_text(title, summary)` strips publisher name suffixes from titles (e.g. `"… Bitcoin Magazine"` appended by their RSS feed), detects when the summary just echoes the headline (common in Bitcoin Magazine's feed), and either returns the title alone or title + a short summary excerpt
- Each entry returns a `title` (for display), `text` (title + excerpt, used for scoring), and `url` (article link)

**Reddit (r/Bitcoin hot posts)**
- Fetches top 10 posts via the public JSON API
- Captures `title` and `permalink` (used to construct `https://reddit.com/r/...` link)

**Fear & Greed Index**
- `alternative.me/fng` — returns 0–100 value
- Normalised to `−1.0 … +1.0` via `(val − 50) / 50`

**BTC 24h Price Momentum** ← dominant signal
- Tries 4 sources in order: CoinGecko → Coinbase → Kraken → Binance
- `btc_price_score = clamp(change_24h / 4.0, −1, +1)` — a ±4% day maps to fully bullish/bearish
- This is why CoinGecko blocking cloud IPs (Render) doesn't kill the feature

**GitHub Dev Momentum**
- Bitcoin Core commits in the last 24h, capped at `min(0.2, commits × 0.04)` — a mild positive tailwind signal

---

## 3. Sentiment Scoring Pipeline

Each headline goes through a two-stage scorer.

### Stage 1 — FinBERT (when `HUGGINGFACE_TOKEN` is set)

ProsusAI/finbert via the Hugging Face Inference API. For a batch of headlines it returns `[[{label, score}, …], …]` — one list of three label-dicts per input. The signed score is:

```
signed = P(positive) − P(negative)
```

This captures both direction and certainty — a 90% confident positive scores +0.90; a split 40/40/20 scores near 0.

**Known FinBERT limitation:** it reads surface financial tone but misses crypto-specific bearish framing. "Institutions cut Bitcoin ETF exposure" scores mildly positive because "keep buying" is present. "Altcoin market cap roundtrips 900 days" scores positive because no crash word appears.

### Stage 2 — Crypto-Domain Lexicon (always runs)

A weighted token + phrase matcher built specifically for crypto market language.

**Token matching** (exact word equality, not substring, to avoid "up" in "support"):
- Positive: `surge/soar/breakout/ath/bullish/rally/inflows/approved` (0.6–1.0), `gain/recover/climb` (0.6–0.7), `support/institutional` (0.3–0.4)
- Negative: `crash/collapse/bearish/selloff/hack/fraud` (0.9–1.0), `liquidation/ban/exploit` (0.8), `drop/fall/loss/decline` (0.6), `cut/sell/selling/exit/reduce` (0.4–0.5), `warning/weak/risk` (0.4–0.5)

**Phrase matching** (substring of normalised text — catches meaning tokens lose):
- Positive: `all time high`, `golden cross`, `rate cut`, `break out`, `higher high`
- Negative: `cut exposure`, `cut bitcoin`, `sell off`, `round trip`, `death cross`, `bear market`, `give back`, `lower high`, `wipe out`

Score = `tanh((pos_sum − neg_sum) / 1.5)` — tanh saturation means two strong words approach ±1 without clipping weaker noise.

### Blending FinBERT + Lexicon (`_blend_finbert_keyword`)

```python
if abs(keyword) >= 0.4 and sign(keyword) != sign(finbert):
    # Lexicon is confident and disagrees — domain knowledge overrides
    blended = 0.7 × keyword + 0.3 × finbert
else:
    # FinBERT leads, lexicon nudges
    blended = 0.6 × finbert + 0.4 × keyword
```

Example of the override in action: "Institutions cut Bitcoin ETF exposure but keep buying XRP" — FinBERT ≈ +0.30 (reads "keep buying"), lexicon = −0.70 (`cut` token + `cut bitcoin` phrase). Lexicon is confident (0.70 ≥ 0.4) and disagrees → blended = **−0.40 (Bearish)**.

### Sentiment Component Blend

Rather than averaging all article scores (which dilutes direction with near-zero noise), individual signals are blended as **weighted components**:

| Signal | Weight | Notes |
|---|---|---|
| BTC 24h price momentum | **5.0** | Dominant — live market direction |
| News (FinBERT + lexicon avg) | 1.0 | Averaged across RSS + Reddit |
| Fear & Greed Index | 1.0 | Slow-moving macro confirmation |
| GitHub dev momentum | 0.5 | Mild long-term tailwind |

```python
current_avg = sum(score × weight for each component) / sum(weights)
```

This is what makes the "Market mood" gauge track the live market: a BTC price with 5× weight can only be overridden by the other signals collectively at a combined weight of 2.5.

### Sentiment Velocity

`velocity = current_score − previous_score` (stored in `PREV_SENTIMENT` global).

A rapid drop matters more than a stable low reading — this is passed as a feature to the Random Forest models.

### Impact Ranking

Each news item gets an `impact` score:
```
impact = |tone_score| × source_credibility
```
Reddit posts are weighted 1.3× vs RSS 1.0× (Reddit signals are more reactive to live events). Items are sorted descending by impact so the most market-moving headlines surface first.

---

## 4. Feature Vector & ML Prediction

### The 15-Feature Vector

```python
[
    tx_count,          # pending transactions
    median_fee_rate,   # sat/vB centre of confirmation window
    total_size_mb,     # mempool backlog in MB
    fee_cluster,       # 0–4 regime label (KMeans or rule-based)
    sentiment_score,   # blended market sentiment −1…+1
    sentiment_velocity,# rate of change of sentiment
    article_volume,    # number of articles processed
    hour_of_day,       # 0–23 UTC
    day_of_week,       # 0=Monday … 6=Sunday
    is_peak_hour,      # 1 if EU/US overlap (13–19 UTC)
    is_asian_session,  # 1 if 00–04 UTC
    is_floor_period,   # 1 if daily quiet window (07–11 UTC)
    is_weekend,        # 1 if Saturday or Sunday
    fee_iqr,           # p90 − p10 (fee dispersion signal)
    vsize_per_tx,      # avg transaction size in vbytes
]
```

Features are scaled via `StandardScaler` before inference (fitted on training data, saved to `scaler.pkl`).

### Three Separate Random Forest Regressors

| Model | Target | Horizon |
|---|---|---|
| `rf_1block` | Next block fee (sat/vB) | ≈ 10 min |
| `rf_3block` | 3-block fee (sat/vB) | ≈ 30 min |
| `rf_6block` | 6-block fee (sat/vB) | ≈ 60 min |

300 trees each, `max_depth=14`, `min_samples_leaf=3`. Separate models because each horizon has different dynamics: 1-block fees are highly sensitive to current congestion; 6-block fees mean-revert faster.

### Confidence Score

```python
confidence = mean over models of:
    1.0 − (std(tree_predictions) / mean(tree_predictions))
```

Low variance between 300 trees = high confidence. Displayed as a progress bar under each fee card (green > 70%, amber > 40%, red otherwise).

### Fallback Prediction

When models aren't loaded, a simple rule fires:
```
fee_1block = median × 1.12
fee_3block = median × 0.95
fee_6block = median × 0.75
confidence = 0.0
model_version = "fallback-rules"
```
Shown with a "Rule-based estimate" badge on the dashboard.

---

## 5. Data Persistence (Supabase)

### Schema

```
mempool_snapshots  ──┐
                     ├──► features ──► predictions ──► actuals
sentiment_snapshots ─┘
```

- **`mempool_snapshots`**: raw on-chain metrics per 5-min tick (tx_count, median_fee_rate, p10/p90, high_fee_pct, avg_tx_size_bytes)
- **`sentiment_snapshots`**: scored sentiment per tick (score, velocity, article_volume, dominant_topic)
- **`features`**: joined ML input row linked to both snapshot tables — this is what the model trains on
- **`predictions`**: model outputs (fee_1block/3block/6block, confidence, model_version) linked to a feature row
- **`actuals`**: ground truth fees paid, used to close the feedback loop for retraining

### Real-Time Updates

The frontend subscribes to `INSERT` events on the `features` table via a Supabase Realtime WebSocket channel. When a new snapshot lands (every 5 minutes), the channel fires and `fetchLatest()` re-runs — pulling the full joined payload including the new prediction. This is why the dashboard updates live without polling.

---

## 6. Frontend Architecture

### State Management (Jotai)

Four atoms cover all dashboard state:

```typescript
latestPredictionAtom   // current fee_1/3/6block + confidence
featuresHistoryAtom    // last 288 rows (24h of 5-min snapshots)
sentimentHistoryAtom   // last 24 sentiment scores (2h sparkline)
connectionStatusAtom   // 'loading' | 'live' | 'error'
```

### Dual Sentiment Gauges

The dashboard separates sentiment into two independent signals:
- **Market mood** — `sentiment_score` from Supabase (BTC price + macro blend). Updates every 5 minutes via Realtime.
- **News tone** — average of current `/api/news` item scores. Updates every 60 seconds independently via a `setInterval` in `page.tsx`.

Thresholds: `> +0.15` = Bullish, `< −0.15` = Bearish, otherwise Neutral. The SentimentSparkline colour uses the same thresholds for visual consistency.

### News Feed

`/api/news` (Next.js route) proxies to the ML backend's `/GET news`. Each item has:
- `title` — clean headline (publisher suffix stripped, no summary echo)
- `url` — direct link to the original article or Reddit post
- `score` — blended FinBERT + lexicon tone (−1 … +1)
- `impact` — `|score| × source_credibility` used for ranking

Cards show the headline as a clickable link (opens in new tab). Tone badge: green > +0.2, red < −0.2, grey otherwise.

### API Routes

| Route | Source | Purpose |
|---|---|---|
| `/api/latest` | Supabase `features` + `predictions` | Fee predictions + mempool history for charts |
| `/api/news` | ML backend `/news` | Latest impact-ranked headlines |

`/api/latest` creates its own Supabase client with the service key (falls back to anon key) to bypass RLS. Both routes are `force-dynamic` — no edge caching.

---

## 7. Known Limitations

**Label leakage in 3-block and 6-block training:**
`y_3block` and `y_6block` are trained on the model's own previous predictions, not real ground truth. The `actuals` table exists to fix this — the correct approach is to record actual confirmed fees 30 and 60 minutes after prediction and retrain on those. The 1-block model uses real `actual_fee_paid` when available.

**Render free tier sleeps after 15 min idle:**
The first request after a sleep period wakes the backend (~30–60s). This delays the scheduler restart. A paid tier or a keep-alive ping service fixes this.

**FinBERT Inference API cold starts:**
Hugging Face's free Inference API can take 20–30s on model cold start. The backend falls back to the keyword lexicon for that cycle and retries on the next snapshot.

**Bootstrap data is synthetic:**
The initial `/bootstrap` seeds 14 days of realistic but computer-generated mempool history. Model accuracy improves as real on-chain data accumulates over weeks.
