import os
import json
import math
import asyncio
import calendar as _calendar
import datetime
from datetime import timezone
import logging
import random
import re
import html

from fastapi import FastAPI, BackgroundTasks
import uvicorn
import httpx
import feedparser
import numpy as np
import pandas as pd
import joblib
from supabase import create_client
from contextlib import asynccontextmanager

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HF_TOKEN:
    logger.warning("HUGGINGFACE_TOKEN not set. Falling back to keyword scoring.")

MODELS = {}
PREV_SENTIMENT = 0.0
LATEST_NEWS = []

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/")

os.makedirs(MODEL_PATH, exist_ok=True)

if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.warning("Supabase credentials not fully set. Proceeding carefully.")
    supabase_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application")
    files_to_load = [
        "rf_1block.pkl",
        "rf_3block.pkl",
        "rf_6block.pkl",
        "kmeans.pkl",
        "scaler.pkl",
    ]
    for file_name in files_to_load:
        file_path = os.path.join(MODEL_PATH, file_name)
        if os.path.exists(file_path):
            try:
                mod = joblib.load(file_path)
                key_name = file_name.replace(".pkl", "")
                MODELS[key_name] = mod
                logger.info(f"Successfully loaded {file_name}")
            except Exception as e:
                logger.error(f"Failed to load {file_name}: {e}")
        else:
            logger.info(f"Model file missing: {file_name}")

    #  Auto-snapshot scheduler 
    interval_secs = int(os.getenv("SNAPSHOT_INTERVAL_MINUTES", "5")) * 60

    async def _scheduler():
        await asyncio.sleep(15)  # brief startup grace period
        while True:
            try:
                logger.info("Scheduler: running auto-snapshot...")
                await snapshot_endpoint()
                logger.info("Scheduler: snapshot complete.")
            except Exception as e:
                logger.error(f"Scheduler: snapshot error: {e}")
            await asyncio.sleep(interval_secs)

    task = asyncio.create_task(_scheduler())
    logger.info(f"Scheduler started (every {interval_secs // 60}m).")

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down FastAPI application")


app = FastAPI(lifespan=lifespan)



# Weighted crypto-news lexicons (single lowercase tokens). Strong movers ~1.0,
# moderate ~0.5-0.7, mild ~0.3-0.4. Used as a crypto-domain correction on top of
# FinBERT (which reads surface tone but misses crypto-specific bearish framing).
_POS_LEXICON = {
    "surge": 1.0, "surges": 1.0, "soar": 1.0, "soars": 1.0, "skyrocket": 1.0,
    "rally": 0.9, "rallies": 0.9, "breakout": 0.9, "ath": 1.0, "bullish": 1.0,
    "bull": 0.7, "moon": 0.7, "rebound": 0.7, "rebounds": 0.7, "jumps": 0.7, "jump": 0.7,
    "inflows": 0.7, "inflow": 0.7, "adoption": 0.7, "adopt": 0.6, "approve": 0.8,
    "approval": 0.8, "approved": 0.8, "gain": 0.6, "gains": 0.6, "rise": 0.6, "rises": 0.6,
    "recover": 0.6, "recovers": 0.6, "upgrade": 0.6, "accumulate": 0.6, "record": 0.4,
    "partnership": 0.5, "buy": 0.4, "institutional": 0.4, "support": 0.3, "high": 0.3,
    "climbs": 0.7, "climb": 0.7, "surpass": 0.7, "surpasses": 0.7, "outperform": 0.6,
    "outperforms": 0.6, "upside": 0.6, "milestone": 0.4,
}
_NEG_LEXICON = {
    "crash": 1.0, "crashes": 1.0, "plunge": 1.0, "plunges": 1.0, "plummet": 1.0,
    "collapse": 1.0, "bearish": 1.0, "selloff": 0.9, "tumble": 0.9, "tumbles": 0.9,
    "dump": 0.9, "dumps": 0.9, "hack": 0.9, "hacked": 0.9, "scam": 0.9, "fraud": 0.9,
    "exploit": 0.8, "liquidation": 0.8, "liquidated": 0.8, "slump": 0.8, "bear": 0.7,
    "ban": 0.8, "banned": 0.8, "outflows": 0.7, "outflow": 0.7, "lawsuit": 0.6, "sue": 0.6,
    "reject": 0.6, "rejected": 0.6, "fear": 0.6, "fud": 0.6, "drop": 0.6, "drops": 0.6,
    "fall": 0.6, "falls": 0.6, "decline": 0.6, "loss": 0.6, "losses": 0.6, "pressure": 0.5,
    "warning": 0.5, "weak": 0.5, "risk": 0.4, "low": 0.3,
    # crypto-specific bearish framing FinBERT misreads as neutral/positive
    "roundtrip": 1.0, "roundtrips": 1.0, "capitulation": 0.9, "capitulate": 0.9,
    "downturn": 0.7, "downtrend": 0.7, "retrace": 0.5, "retraces": 0.5, "retreat": 0.6,
    "underwater": 0.7, "bleed": 0.7, "bleeds": 0.7, "erase": 0.7, "erases": 0.7,
    "erased": 0.7, "wipe": 0.8, "wipes": 0.8, "wiped": 0.8, "sink": 0.6, "sinks": 0.6,
    "slip": 0.5, "slips": 0.5, "slide": 0.6, "slides": 0.6, "fails": 0.5, "fail": 0.5,
    "struggle": 0.5, "struggles": 0.5, "stall": 0.5, "stalls": 0.5, "halt": 0.5,
    "probe": 0.5, "investigation": 0.5, "delay": 0.5, "delays": 0.5, "crackdown": 0.7,
    "correction": 0.5,
}

# Multi-word phrases (checked as substrings of the normalised text). These carry
# meaning that single tokens lose, e.g. "give back" / "round trip" / "lower high".
_POS_PHRASES = {
    "all time high": 1.0, "record high": 0.8, "new high": 0.7, "new highs": 0.7,
    "break out": 0.8, "breaks out": 0.8, "breaking out": 0.8, "higher high": 0.7,
    "golden cross": 0.9, "buy the dip": 0.4,
}
_NEG_PHRASES = {
    "give back": 0.8, "gives back": 0.8, "gave back": 0.8, "giving back": 0.8,
    "sell off": 0.9, "sells off": 0.9, "round trip": 1.0, "lower high": 0.7,
    "lower low": 0.7, "death cross": 0.9, "bear market": 0.9, "wipe out": 0.9,
    "wiped out": 0.9, "break down": 0.6, "breaks down": 0.6, "breaking down": 0.6,
    "all time low": 1.0, "record low": 0.9, "loses support": 0.8, "lost support": 0.8,
    "below support": 0.7, "sell pressure": 0.7, "selling pressure": 0.7,
    "profit taking": 0.5,
}


# Source credibility multiplier for impact ranking (Reddit slightly up-weighted).
_SRC_WEIGHT = {"News": 1.0, "Reddit": 1.3}

# Many RSS feeds append their own site name to the title field, sometimes with
# a separator (dash/pipe) and sometimes with just a space. Match at end of string.
_RSS_PUBLISHER_SUFFIX = re.compile(
    r'[\s\-–|]*\b(Bitcoin Magazine|CoinDesk|Cointelegraph|Decrypt|Bitcoinist'
    r'|NewsBTC|CryptoSlate|The Block|BeInCrypto)\s*$',
    re.IGNORECASE,
)


def _clean_rss_text(title: str, summary: str) -> str:
    """Return a clean, non-repetitive text for an RSS entry.

    Some feeds (e.g. Bitcoin Magazine) append the site name to every title AND
    start their summary with the full headline again, producing:
      "Headline Site Name\\n\\nHeadline Site Name body..."
    We strip the publisher suffix and skip the summary if it just echoes the title.
    """
    title = html.unescape(title).strip()
    title = _RSS_PUBLISHER_SUFFIX.sub('', title).strip()

    summary = html.unescape(re.sub(r'<[^>]+>', '', summary)).strip()
    if not summary:
        return title

    # Detect repeated header: normalise both and compare the first 50 chars.
    nt = re.sub(r'\s+', ' ', title.lower())
    ns = re.sub(r'\s+', ' ', summary.lower())
    overlap = min(50, len(nt))
    if overlap >= 20 and ns.startswith(nt[:overlap]):
        return title  # summary is just the headline again — use title only

    # Append a short excerpt of the summary for richer context.
    return (title + ". " + summary[:200]).strip()


def _mk_item(source: str, title: str, score: float, url: str = "") -> dict:
    return {
        "source": source,
        "title": title,
        "url": url,
        "score": round(float(score), 3),
        "impact": round(abs(float(score)) * _SRC_WEIGHT.get(source, 1.0), 3),
    }


def _assign_fee_cluster(median_fee_rate: float) -> int:
    """Unified 2026-regime fee cluster assignment used by both live data and bootstrap.
    Cluster 0 = Filler floor  (< 4 sat/vB)
    Cluster 1 = Economy/utility (4–9 sat/vB)
    Cluster 2 = Normal          (9–16 sat/vB)
    Cluster 3 = Priority        (16–30 sat/vB)
    Cluster 4 = Urgent/L2 spike (> 30 sat/vB)
    """
    if median_fee_rate < 4:
        return 0
    elif median_fee_rate < 9:
        return 1
    elif median_fee_rate < 16:
        return 2
    elif median_fee_rate < 30:
        return 3
    return 4


def _keyword_score_one(text: str) -> float:
    """Crypto-domain sentiment in [-1, 1] from weighted tokens + phrases."""
    norm = re.sub(r"\s+", " ", str(text).lower())
    # Token-equality (not substring) avoids false hits like "up" in "support".
    tokens = set(re.findall(r"[a-z]+", norm))
    pos = sum(w for term, w in _POS_LEXICON.items() if term in tokens)
    neg = sum(w for term, w in _NEG_LEXICON.items() if term in tokens)
    pos += sum(w for phrase, w in _POS_PHRASES.items() if phrase in norm)
    neg += sum(w for phrase, w in _NEG_PHRASES.items() if phrase in norm)
    # tanh saturation: a couple of strong words approach +/-1 without clipping noise.
    return max(-1.0, min(1.0, math.tanh((pos - neg) / 1.5)))


def _score_texts_keyword_fallback(texts: list[str]) -> list[float]:
    return [_keyword_score_one(t) for t in texts]


def _blend_finbert_keyword(finbert: float, keyword: float) -> float:
    """Combine FinBERT's general tone with the crypto-domain lexicon.

    When the lexicon is confident (|keyword| >= 0.4) and disagrees in sign with
    FinBERT, trust the domain lexicon — this is what stops headlines like
    "Altcoin Market Cap Roundtrips 900 Days" from reading as bullish. Otherwise
    FinBERT leads with the lexicon as a nudge.
    """
    disagree = finbert == 0.0 or (finbert > 0) != (keyword > 0)
    if abs(keyword) >= 0.4 and disagree:
        blended = 0.7 * keyword + 0.3 * finbert
    else:
        blended = 0.6 * finbert + 0.4 * keyword
    return max(-1.0, min(1.0, blended))


async def score_texts(texts: list[str]) -> list[float]:
    if not texts: return []
    if not HF_TOKEN: return _score_texts_keyword_fallback(texts)

    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        async with httpx.AsyncClient() as client:
            response = await client.post(HF_API_URL, headers=headers, json={"inputs": texts}, timeout=15.0)
            if response.status_code != 200:
                logger.warning(f"HF API Error {response.status_code}: {response.text}")
                return _score_texts_keyword_fallback(texts)

            results = response.json()
            # FinBERT returns one list of label-dicts PER input. Normalise a single
            # input ([{...},{...}]) to the batch shape ([[{...},{...}]]).
            if results and isinstance(results[0], dict):
                results = [results]

            def _signed(label_dicts) -> float:
                # Signed score = P(positive) - P(negative); captures intensity AND
                # certainty far better than picking the single top label.
                probs = {d.get("label", "").lower(): d.get("score", 0.0) for d in label_dicts}
                return float(probs.get("positive", 0.0) - probs.get("negative", 0.0))

            finbert_scores = [_signed(item) for item in results]
            # Guard against any shape surprise so we never return misaligned scores.
            if len(finbert_scores) != len(texts):
                logger.warning("HF API returned mismatched length; using keyword fallback")
                return _score_texts_keyword_fallback(texts)
            # Correct FinBERT's general tone with crypto-domain knowledge.
            keyword_scores = _score_texts_keyword_fallback(texts)
            return [_blend_finbert_keyword(f, k) for f, k in zip(finbert_scores, keyword_scores)]
    except Exception as e:
        logger.warning(f"HF API Call failed: {e}")
        return _score_texts_keyword_fallback(texts)


def get_dominant_topic(texts: list[str]) -> str:
    categories = {
        "price_action": ["price", "ath", "dump", "rally", "surge", "crash", "pump"],
        "regulation": ["sec", "regulation", "ban", "law", "government", "legal"],
        "etf": ["etf", "blackrock", "spot", "approval", "fund"],
        "adoption": ["adopt", "payment", "merchant", "country", "integration"],
        "hack_or_fud": ["hack", "scam", "exploit", "stolen", "fraud", "vulnerability"],
        "mining": ["mining", "hashrate", "miner", "difficulty", "block"],
        "macro": ["fed", "inflation", "interest", "economy", "dollar"],
    }

    combined_text = " ".join(texts).lower()

    max_hits = -1
    dominant = "price_action"

    for category, keywords in categories.items():
        hits = sum(combined_text.count(kw) for kw in keywords)
        if hits > max_hits:
            max_hits = hits
            dominant = category

    return dominant


async def _fetch_btc_change_24h(client: httpx.AsyncClient) -> float | None:
    """BTC 24h % change, trying several public APIs in order. CoinGecko's free tier
    rate-limits/blocks many cloud IPs (e.g. Render), so we fall back to exchange
    tickers that are reliable from datacenter IPs."""
    sources = [
        ("coingecko",
         "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true",
         lambda j: float(j["bitcoin"]["usd_24h_change"])),
        ("coinbase",
         "https://api.exchange.coinbase.com/products/BTC-USD/stats",
         lambda j: (float(j["last"]) - float(j["open"])) / float(j["open"]) * 100.0),
        ("kraken",
         "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
         lambda j: (lambda d: (float(d["c"][0]) - float(d["o"])) / float(d["o"]) * 100.0)(
             list(j["result"].values())[0])),
        ("binance",
         "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT",
         lambda j: float(j["priceChangePercent"])),
    ]
    for name, url, parse in sources:
        try:
            r = await client.get(url)
            if r.status_code == 200:
                val = parse(r.json())
                logger.info(f"BTC 24h change via {name}: {val:+.2f}%")
                return val
            logger.warning(f"BTC price source {name} returned {r.status_code}")
        except Exception as e:
            logger.warning(f"BTC price source {name} failed: {e}")
    return None


async def fetch_sentiment_data() -> dict:
    global PREV_SENTIMENT

    def fetch_rss() -> list:
        feeds = [
            "https://feeds.feedburner.com/CoinDesk",
            "https://cointelegraph.com/rss",
            "https://bitcoinmagazine.com/.rss/full/",
            "https://decrypt.co/feed",
            "https://bitcoinist.com/feed/",
            "https://www.newsbtc.com/feed/",
            "https://cryptoslate.com/feed/",
        ]
        items = []
        for feed_url in feeds:
            try:
                parsed = feedparser.parse(feed_url)
                for entry in parsed.entries[:5]:
                    raw_title = entry.title if hasattr(entry, "title") else ""
                    summary = entry.get("summary", "")
                    url = entry.get("link", "")
                    clean_title = _clean_rss_text(raw_title, "")        # display: headline only
                    scoring_text = _clean_rss_text(raw_title, summary)  # scoring: + summary context
                    items.append({"title": clean_title, "text": scoring_text, "url": url})
            except Exception as e:
                logger.error(f"Error fetching RSS {feed_url}: {e}")
        return items

    loop = asyncio.get_event_loop()
    rss_items = await loop.run_in_executor(None, fetch_rss)
    rss_titles = [i["title"] for i in rss_items]
    rss_texts  = [i["text"]  for i in rss_items]
    rss_urls   = [i["url"]   for i in rss_items]

    reddit_titles, reddit_texts, reddit_urls = [], [], []
    try:
        headers = {"User-Agent": "mempool-sentiment-bot/1.0 (by /u/anonymous)"}
        async with httpx.AsyncClient(
            headers=headers, follow_redirects=True
        ) as reddit_client:
            subs = ["Bitcoin"]
            for sub in subs:
                resp = await reddit_client.get(
                    f"https://www.reddit.com/r/{sub}/hot.json?limit=10", timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    posts = data.get("data", {}).get("children", [])
                    for post in posts:
                        title = post.get("data", {}).get("title", "")
                        permalink = post.get("data", {}).get("permalink", "")
                        if title:
                            reddit_titles.append(title)
                            reddit_texts.append(title)
                            reddit_urls.append(f"https://reddit.com{permalink}" if permalink else "")
    except Exception as e:
        logger.warning(f"Reddit public JSON fetch failed: {e}")

    fng_score = 0.0
    github_commit_score = 0.0
    btc_price_score = 0.0
    btc_change_24h = None
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers={"User-Agent": "mempool-sentiment-bot/1.0"}) as client:
            # 1. Fear and Greed
            try:
                fng_resp = await client.get("https://api.alternative.me/fng/?limit=1")
                if fng_resp.status_code == 200:
                    data = fng_resp.json()
                    val = int(data["data"][0]["value"])
                    fng_score = (val - 50) / 50.0
            except Exception as e:
                logger.warning(f"Error fetching Fear and Greed: {e}")

            # 1b. BTC price momentum (24h) — the dominant market-direction signal.
            # A +/-4% day maps to fully bullish/bearish so sentiment tracks price.
            btc_change_24h = await _fetch_btc_change_24h(client)
            if btc_change_24h is not None:
                btc_price_score = max(-1.0, min(1.0, btc_change_24h / 4.0))

            # 2. GitHub Commits
            try:
                gh_resp = await client.get("https://api.github.com/repos/bitcoin/bitcoin/commits?per_page=10")
                if gh_resp.status_code == 200:
                    commits = gh_resp.json()
                    now = datetime.datetime.now(timezone.utc)
                    recent_commits = 0
                    for c in commits:
                        date_str = c["commit"]["committer"]["date"]
                        c_date = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        if (now - c_date).total_seconds() < 86400:
                            recent_commits += 1
                    github_commit_score = min(0.2, recent_commits * 0.04)
            except Exception as e:
                logger.warning(f"Error fetching GitHub commits: {e}")
    except Exception as e:
        logger.error(f"Error in macro data fetch: {e}")

    rss_scores = await score_texts(rss_texts) if rss_texts else []
    reddit_scores = await score_texts(reddit_texts) if reddit_texts else []

    global LATEST_NEWS
    all_items = [_mk_item("News", t, s, u) for t, s, u in zip(rss_titles, rss_scores, rss_urls)]
    all_items += [_mk_item("Reddit", t, s, u) for t, s, u in zip(reddit_titles, reddit_scores, reddit_urls)]
    all_items.sort(key=lambda x: x["impact"], reverse=True)
    LATEST_NEWS = all_items[:4]

    article_scores = (
        [s * 1.0 for s in rss_scores]
        + [s * 1.3 for s in reddit_scores]
    )
    news_avg = (sum(article_scores) / len(article_scores)) if article_scores else 0.0

    # Blend the independent signals. BTC price momentum dominates so the displayed
    # sentiment moves the same direction as the live market.
    components = [(news_avg, 1.0)]
    if btc_price_score != 0.0:
        components.append((btc_price_score, 5.0))      # dominant: live price trend
    if fng_score != 0.0:
        components.append((fng_score, 1.0))            # macro fear/greed (slow, secondary)
    if github_commit_score != 0.0:
        components.append((github_commit_score, 0.5))  # dev momentum (mild)

    blend_weight = sum(w for _, w in components)
    current_avg = (
        sum(s * w for s, w in components) / blend_weight if blend_weight > 0 else 0.0
    )
    current_avg = max(-1.0, min(1.0, current_avg))

    all_texts = rss_titles + reddit_titles

    if current_avg == 0.0 and len(all_texts) == 0:
        # If API failed or returned nothing, drift slightly so it doesn't freeze
        current_avg = PREV_SENTIMENT + random.gauss(0, 0.015)
        current_avg = max(-1.0, min(1.0, current_avg))
        # Keep recent topic to avoid empty state. LATEST_NEWS holds dicts
        # ({"source","text","score"}), so extract the text before scoring topics.
        dominant_topic = (
            get_dominant_topic([item["title"] for item in LATEST_NEWS])
            if LATEST_NEWS
            else "price_action"
        )
    else:
        dominant_topic = get_dominant_topic(all_texts)

    score_velocity = current_avg - PREV_SENTIMENT
    PREV_SENTIMENT = current_avg

    return {
        "score": current_avg,
        "score_velocity": score_velocity,
        "article_volume": max(len(all_texts), 5 + int(random.uniform(0, 10))),
        "dominant_topic": dominant_topic,
        "source_weight": 1.3 if reddit_texts else 1.0,
        "all_texts": all_texts,
    }


def _synthetic_mempool_core() -> dict:
    """Realistic 2026-regime synthetic mempool snapshot. Used as a fallback when
    the live mempool.space API is unreachable so the dashboard stays dynamic."""
    now = datetime.datetime.now(timezone.utc)
    median_fee_rate = _sample_fee_layer(now.hour, now.weekday()) + random.gauss(0, 0.5)
    median_fee_rate = min(FEE_CAP, max(ORDINALS_FLOOR, median_fee_rate))

    sess = _session(now.hour)
    if sess == "us_open":
        tx_count = int(random.uniform(30_000, 70_000) + random.gauss(0, 2000))
    elif sess == "peak":
        tx_count = int(random.uniform(20_000, 50_000) + random.gauss(0, 1500))
    else:
        tx_count = int(random.uniform(8_000, 25_000) + random.gauss(0, 1000))

    avg_tx_size_bytes = random.uniform(250, 480)
    total_size_mb = min(80.0, max(1.5, tx_count * avg_tx_size_bytes / 1_000_000))

    p10 = max(ORDINALS_FLOOR, median_fee_rate * random.uniform(0.50, 0.80))
    p90 = min(FEE_CAP, median_fee_rate * random.uniform(1.2, 2.0))
    high_fee_pct = min(100.0, max(0.0, (median_fee_rate - 10) * 3))

    return {
        "tx_count": tx_count,
        "total_size_mb": total_size_mb,
        "median_fee_rate": median_fee_rate,
        "p10": p10,
        "p90": p90,
        "high_fee_pct": high_fee_pct,
        "avg_tx_size_bytes": avg_tx_size_bytes,
        "source": "synthetic",
    }


async def _real_mempool_core() -> dict | None:
    """Fetch a live snapshot from mempool.space. Returns None if unreachable so
    the caller can fall back to synthetic generation."""
    try:
        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True,
            headers={"User-Agent": "satsense/1.0"},
        ) as client:
            resp = await client.get("https://mempool.space/api/mempool")
            if resp.status_code != 200:
                logger.warning(f"mempool.space returned {resp.status_code}")
                return None
            mp = resp.json()
    except Exception as e:
        logger.warning(f"mempool.space fetch failed: {e}")
        return None

    tx_count = int(mp.get("count", 0) or 0)
    total_vsize = float(mp.get("vsize", 0) or 0.0)  # vbytes
    histogram = mp.get("fee_histogram") or []       # list of [feerate, vsize]
    if tx_count <= 0 or not histogram:
        return None

    full_v = sum(float(v) for _, v in histogram) or 1.0
    high_fee_pct = 100.0 * sum(float(v) for fr, v in histogram if float(fr) >= 10.0) / full_v

    # Percentiles over the *confirmation window* (top of the mempool by feerate —
    # the txs that would actually be mined in the next ~6 blocks). Percentiles over
    # the whole backlog collapse to the 1 sat/vB dust floor and carry no signal.
    WINDOW_VB = 6_000_000.0  # ~6 blocks of vsize
    desc = sorted(
        ((float(fr), float(v)) for fr, v in histogram), key=lambda x: x[0], reverse=True
    )
    window, acc = [], 0.0
    for fr, v in desc:
        window.append((fr, v))
        acc += v
        if acc >= WINDOW_VB:
            break
    pairs = sorted(window, key=lambda x: x[0])  # ascending for the percentile walk
    total_v = sum(v for _, v in pairs) or 1.0

    def _pct(p: float) -> float:
        target = p * total_v
        cum = 0.0
        for fr, v in pairs:
            cum += v
            if cum >= target:
                return fr
        return pairs[-1][0]

    median_fee_rate = _pct(0.50)
    p10 = _pct(0.10)
    p90 = _pct(0.90)

    return {
        "tx_count": tx_count,
        "total_size_mb": total_vsize / 1_000_000.0,
        "median_fee_rate": min(FEE_CAP, max(ORDINALS_FLOOR, median_fee_rate)),
        "p10": min(FEE_CAP, max(ORDINALS_FLOOR, p10)),
        "p90": min(FEE_CAP, max(ORDINALS_FLOOR, p90)),
        "high_fee_pct": high_fee_pct,
        "avg_tx_size_bytes": (total_vsize / tx_count) if tx_count else 350.0,
        "source": "mempool.space",
    }


async def fetch_mempool_data() -> dict:
    # Try the live mempool.space API first; fall back to the synthetic 2026-regime
    # engine if it is blocked, rate-limited, or unreachable.
    core = await _real_mempool_core()
    if core is None:
        logger.info("Using synthetic fallback for mempool data")
        core = _synthetic_mempool_core()

    tx_count        = core["tx_count"]
    median_fee_rate = core["median_fee_rate"]
    total_size_mb   = core["total_size_mb"]
    p10             = core["p10"]
    p90             = core["p90"]
    high_fee_pct    = core["high_fee_pct"]
    avg_tx_size_bytes = core["avg_tx_size_bytes"]

    tx_arrival_rate = tx_count / 5.0

    fee_cluster = 0
    if "kmeans" in MODELS:
        try:
            X_clust = np.array([[median_fee_rate, tx_count, total_size_mb]])
            fee_cluster = int(MODELS["kmeans"].predict(X_clust)[0])
        except Exception as e:
            logger.error(f"KMeans prediction failed: {e}")
            fee_cluster = _assign_fee_cluster(median_fee_rate)
    else:
        fee_cluster = _assign_fee_cluster(median_fee_rate)

    return {
        "tx_count": tx_count,
        "total_size_mb": total_size_mb,
        "median_fee_rate": median_fee_rate,
        "p10": p10,
        "p90": p90,
        "high_fee_pct": high_fee_pct,
        "avg_tx_size_bytes": avg_tx_size_bytes,
        "tx_arrival_rate": tx_arrival_rate,
        "fee_cluster": fee_cluster,
    }


def _temporal_flags(hour: int, weekday: int) -> tuple:
    """Return (is_peak, is_asian, is_floor, is_weekend) booleans."""
    is_peak    = 1 if 13 <= hour <= 19 else 0   # EU/US overlap
    is_asian   = 1 if 0  <= hour <= 4  else 0   # Asian session
    is_floor   = 1 if 7  <= hour <= 11 else 0   # Daily quiet window
    is_weekend = 1 if weekday >= 5 else 0
    return is_peak, is_asian, is_floor, is_weekend


def run_prediction(feature_dict: dict) -> dict:
    req_models = ["rf_1block", "rf_3block", "rf_6block", "scaler"]
    if all(m in MODELS for m in req_models):
        try:
            scaler = MODELS["scaler"]
            now = datetime.datetime.now(timezone.utc)
            hour, weekday = now.hour, now.weekday()
            is_peak, is_asian, is_floor, is_weekend = _temporal_flags(hour, weekday)

            # fee_iqr: use p10/p90 if caller supplies them, else estimate from median
            median = feature_dict.get("median_fee_rate", 1.0)
            p10 = feature_dict.get("p10", median * 0.6)
            p90 = feature_dict.get("p90", median * 1.8)
            fee_iqr = max(0.0, p90 - p10)

            # vsize_per_tx: bytes per transaction (proxy for large-vs-small batches)
            tx_count   = feature_dict.get("tx_count", 1)
            total_vb   = feature_dict.get("total_size_mb", 0.0) * 1_000_000
            vsize_per_tx = feature_dict.get("avg_tx_size_bytes") or (total_vb / max(tx_count, 1))

            X = np.array(
                [
                    [
                        tx_count,
                        median,
                        feature_dict.get("total_size_mb", 0.0),
                        feature_dict.get("fee_cluster", 0),
                        feature_dict.get("sentiment_score", 0.0),
                        feature_dict.get("sentiment_velocity", 0.0),
                        feature_dict.get("article_volume", 0),
                        hour,
                        weekday,
                        is_peak,
                        is_asian,
                        is_floor,
                        is_weekend,
                        fee_iqr,
                        vsize_per_tx,
                    ]
                ]
            )

            X_scaled = scaler.transform(X)

            pred_1 = MODELS["rf_1block"].predict(X_scaled)[0]
            pred_3 = MODELS["rf_3block"].predict(X_scaled)[0]
            pred_6 = MODELS["rf_6block"].predict(X_scaled)[0]

            confs = []
            for m in ["rf_1block", "rf_3block", "rf_6block"]:
                rf = MODELS[m]
                preds = np.array([tree.predict(X_scaled)[0] for tree in rf.estimators_])
                mean = preds.mean()
                std = preds.std()
                conf = 1.0 - (std / mean) if mean > 0 else 0.0
                confs.append(max(0.0, min(1.0, conf)))

            confidence = float(np.mean(confs))
            version_path = os.path.join(MODEL_PATH, "version.txt")
            if os.path.exists(version_path):
                with open(version_path, "r") as f:
                    model_version = f.read().strip()
            else:
                model_version = "rf-trained"

            return {
                "fee_1block": round(max(1.0, float(pred_1)), 2),
                "fee_3block": round(max(1.0, float(pred_3)), 2),
                "fee_6block": round(max(1.0, float(pred_6)), 2),
                "confidence": confidence,
                "model_version": model_version,
            }
        except Exception as e:
            logger.error(f"Error in prediction: {e}")

    median_fee_rate = feature_dict.get("median_fee_rate", 1.0)
    return {
        "fee_1block": round(max(1.0, float(median_fee_rate * 1.12)), 2),
        "fee_3block": round(max(1.0, float(median_fee_rate * 0.95)), 2),
        "fee_6block": round(max(1.0, float(median_fee_rate * 0.75)), 2),
        "confidence": 0.0,
        "model_version": "fallback-rules",
    }


@app.get("/health")
def health_endpoint():
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "ai_engine": "huggingface" if HF_TOKEN else "keyword_fallback",
    }


@app.post("/snapshot")
async def snapshot_endpoint():
    if not supabase_client:
        return {"error": "Supabase client not initialized"}

    mempool_data, sentiment_data = await asyncio.gather(
        fetch_mempool_data(), fetch_sentiment_data()
    )

    m_resp = (
        supabase_client.table("mempool_snapshots")
        .insert(
            {
                "tx_count": mempool_data["tx_count"],
                "total_size_mb": mempool_data["total_size_mb"],
                "median_fee_rate": mempool_data["median_fee_rate"],
                "p10_fee_rate": mempool_data["p10"],
                "p90_fee_rate": mempool_data["p90"],
                "high_fee_pct": mempool_data["high_fee_pct"],
                "avg_tx_size_bytes": mempool_data["avg_tx_size_bytes"],
                "tx_arrival_rate": mempool_data["tx_arrival_rate"],
                "fee_cluster": mempool_data["fee_cluster"],
            }
        )
        .execute()
    )
    if not m_resp.data:
        return {"error": "Failed to insert mempool snapshot"}
    mempool_id = m_resp.data[0]["id"]

    s_resp = (
        supabase_client.table("sentiment_snapshots")
        .insert(
            {
                "score": sentiment_data["score"],
                "score_velocity": sentiment_data["score_velocity"],
                "article_volume": sentiment_data["article_volume"],
                "dominant_topic": sentiment_data["dominant_topic"],
                "source_weight": sentiment_data["source_weight"],
            }
        )
        .execute()
    )
    if not s_resp.data:
        return {"error": "Failed to insert sentiment snapshot"}
    sentiment_id = s_resp.data[0]["id"]

    now = datetime.datetime.now(timezone.utc)
    feature_payload = {
        "mempool_snapshot_id": mempool_id,
        "sentiment_snapshot_id": sentiment_id,
        "tx_count": mempool_data["tx_count"],
        "median_fee_rate": mempool_data["median_fee_rate"],
        "total_size_mb": mempool_data["total_size_mb"],
        "fee_cluster": mempool_data["fee_cluster"],
        "sentiment_score": sentiment_data["score"],
        "sentiment_velocity": sentiment_data["score_velocity"],
        "article_volume": sentiment_data["article_volume"],
    }
    f_resp = supabase_client.table("features").insert(feature_payload).execute()
    if not f_resp.data:
        return {"error": "Failed to insert features"}
    feature_id = f_resp.data[0]["id"]

    pred_payload = feature_payload.copy()
    pred_payload["hour_of_day"] = now.hour
    pred_payload["day_of_week"] = now.weekday()
    # Pass through the real fee-dispersion + tx-size signals so the model sees the
    # same fee_iqr / vsize_per_tx it was trained on (avoids train/serve skew).
    pred_payload["p10"] = mempool_data["p10"]
    pred_payload["p90"] = mempool_data["p90"]
    pred_payload["avg_tx_size_bytes"] = mempool_data["avg_tx_size_bytes"]
    pred_dict = run_prediction(pred_payload)

    p_resp = (
        supabase_client.table("predictions")
        .insert(
            {
                "feature_id": feature_id,
                "fee_1block": pred_dict["fee_1block"],
                "fee_3block": pred_dict["fee_3block"],
                "fee_6block": pred_dict["fee_6block"],
                "confidence": pred_dict["confidence"],
                "model_version": pred_dict["model_version"],
            }
        )
        .execute()
    )
    if not p_resp.data:
        return {"error": "Failed to insert prediction"}

    return {
        "feature_id": feature_id,
        "prediction": pred_dict,
        "mempool": mempool_data,
        "sentiment": sentiment_data,
    }


@app.post("/train")
def train_endpoint():
    if not supabase_client:
        return {"error": "Supabase not configured"}

    # Pull features and their associated mempool snapshot (for p10/p90 IQR)
    resp = (
        supabase_client.table("features")
        .select(
            "*, "
            "predictions(*, actuals(*)), "
            "mempool_snapshots!mempool_snapshot_id(p10_fee_rate, p90_fee_rate, avg_tx_size_bytes)"
        )
        .order("captured_at")
        .execute()
    )

    data = resp.data
    if len(data) < 50:
        return {"error": "not enough data", "rows": len(data)}

    records = []
    for d in data:
        preds = d.get("predictions", [])
        pres = (
            preds[0]
            if isinstance(preds, list) and preds
            else preds if isinstance(preds, dict) else {}
        )

        actuals = pres.get("actuals", []) if pres else []
        acts = (
            actuals[0]
            if isinstance(actuals, list) and actuals
            else actuals if isinstance(actuals, dict) else {}
        )

        if not pres and not acts:
            continue

        try:
            dt = datetime.datetime.fromisoformat(
                d.get("captured_at", "").replace("Z", "+00:00")
            )
            hour_of_day = dt.hour
            day_of_week = dt.weekday()
        except Exception:
            hour_of_day = 0
            day_of_week = 0

        is_peak, is_asian, is_floor, is_weekend = _temporal_flags(hour_of_day, day_of_week)

        # p10/p90 from joined mempool_snapshots; fall back to estimates if missing
        ms = d.get("mempool_snapshots") or {}
        median_fee = d.get("median_fee_rate", 1.0)
        p10 = ms.get("p10_fee_rate") or (median_fee * 0.6)
        p90 = ms.get("p90_fee_rate") or (median_fee * 1.8)
        fee_iqr = max(0.0, float(p90) - float(p10))

        # vsize per tx — high value = large consolidation txs; low = small retail
        tx_count  = d.get("tx_count", 1) or 1
        total_vb  = d.get("total_size_mb", 0.0) * 1_000_000
        avg_tx_sz = ms.get("avg_tx_size_bytes") or (total_vb / tx_count)
        vsize_per_tx = float(avg_tx_sz) if avg_tx_sz else (total_vb / tx_count)

        r = {
            # Base mempool features
            "tx_count":          d.get("tx_count", 0),
            "median_fee_rate":   median_fee,
            "total_size_mb":     d.get("total_size_mb", 0.0),
            "fee_cluster":       d.get("fee_cluster", 0),
            # Sentiment features
            "sentiment_score":   d.get("sentiment_score", 0.0),
            "sentiment_velocity": d.get("sentiment_velocity", 0.0),
            "article_volume":    d.get("article_volume", 0),
            # Raw temporal
            "hour_of_day":       hour_of_day,
            "day_of_week":       day_of_week,
            # Derived temporal flags (high-signal for the model)
            "is_peak_hour":      is_peak,
            "is_asian_session":  is_asian,
            "is_floor_period":   is_floor,
            "is_weekend":        is_weekend,
            # Fee dispersion — IQR as a congestion signal
            "fee_iqr":           fee_iqr,
            # Tx-size proxy — large batches vs. small retail
            "vsize_per_tx":      vsize_per_tx,
            "f_id":              d.get("id"),
        }

        actual_fee = acts.get("actual_fee_paid")
        r["y_1block"] = (
            actual_fee if actual_fee is not None else pres.get("fee_1block", 1.0)
        )
        r["y_3block"] = pres.get("fee_3block", 1.0)
        r["y_6block"] = pres.get("fee_6block", 1.0)

        records.append(r)

    df = pd.DataFrame(records)
    if len(df) < 50:
        return {"error": "not enough complete data", "rows": len(df)}

    features_cols = [
        # Mempool state
        "tx_count",
        "median_fee_rate",
        "total_size_mb",
        "fee_cluster",
        # Sentiment
        "sentiment_score",
        "sentiment_velocity",
        "article_volume",
        # Temporal (raw)
        "hour_of_day",
        "day_of_week",
        # Temporal (derived flags) — the key new signals
        "is_peak_hour",
        "is_asian_session",
        "is_floor_period",
        "is_weekend",
        # Fee dispersion
        "fee_iqr",
        # Tx-type proxy
        "vsize_per_tx",
    ]
    X = df[features_cols]
    y_1 = df["y_1block"]
    y_3 = df["y_3block"]
    y_6 = df["y_6block"]

    X_train, X_test, y1_train, y1_test, y3_train, y3_test, y6_train, y6_test = (
        train_test_split(X, y_1, y_3, y_6, test_size=0.2, random_state=42)
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf1 = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=3, random_state=42
    )
    rf3 = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=3, random_state=42
    )
    rf6 = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=3, random_state=42
    )

    rf1.fit(X_train_scaled, y1_train)
    rf3.fit(X_train_scaled, y3_train)
    rf6.fit(X_train_scaled, y6_train)

    km = KMeans(n_clusters=5, random_state=42)
    km.fit(X_train[["median_fee_rate", "tx_count", "total_size_mb"]])

    rmse1 = float(math.sqrt(mean_squared_error(y1_test, rf1.predict(X_test_scaled))))
    rmse3 = float(math.sqrt(mean_squared_error(y3_test, rf3.predict(X_test_scaled))))
    rmse6 = float(math.sqrt(mean_squared_error(y6_test, rf6.predict(X_test_scaled))))

    mae1 = float(mean_absolute_error(y1_test, rf1.predict(X_test_scaled)))
    mae3 = float(mean_absolute_error(y3_test, rf3.predict(X_test_scaled)))
    mae6 = float(mean_absolute_error(y6_test, rf6.predict(X_test_scaled)))

    version = f"v{int(datetime.datetime.now(timezone.utc).timestamp())}"

    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(rf1,    os.path.join(MODEL_PATH, "rf_1block.pkl"))
    joblib.dump(rf3,    os.path.join(MODEL_PATH, "rf_3block.pkl"))
    joblib.dump(rf6,    os.path.join(MODEL_PATH, "rf_6block.pkl"))
    joblib.dump(km,     os.path.join(MODEL_PATH, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

    with open(os.path.join(MODEL_PATH, "version.txt"), "w") as f:
        f.write(version)

    for fname, obj in [
        ("rf_1block", rf1),
        ("rf_3block", rf3),
        ("rf_6block", rf6),
        ("kmeans",    km),
        ("scaler",    scaler),
    ]:
        MODELS[fname] = obj

    df["new_cluster"] = km.predict(df[["median_fee_rate", "tx_count", "total_size_mb"]])
    updates = []
    for _, row in df.iterrows():
        updates.append({"id": row["f_id"], "fee_cluster": int(row["new_cluster"])})

    for i in range(0, len(updates), 100):
        batch = updates[i : i + 100]
        supabase_client.table("features").upsert(batch).execute()

    return {
        "status": "trained",
        "rows_used": len(X_train),
        "version": version,
        "feature_count": len(features_cols),
        "rmse": {"1block": rmse1, "3block": rmse3, "6block": rmse6},
        "mae":  {"1block": mae1,  "3block": mae3,  "6block": mae6},
    }


#
# Bootstrap helpers — calibrated to April 2026 Bitcoin mempool regime
#

# 2026 Efficiency Floor: Lightning/L2 migration has pushed mainchain clear.
# Filler (Ordinals/Runes) sustains a 1-3 sat/vB baseline.
ORDINALS_FLOOR = 1.5   # sat/vB  (was 6.0 — that was the 2023 Inscription Wars era)
FEE_CAP        = 50.0  # sat/vB  hard ceiling for 2026 (extreme events only)

#  Three-layer probability tables by session 
# Each session defines (p_filler, p_utility, p_priority)
# Filler   = 1-3 sat/vB  (Ordinals / low-priority consolidation)
# Utility  = 4-15 sat/vB (retail, exchange, Lightning topups)
# Priority = 15-40 sat/vB (L2 settlement batches, institutional)

_LAYER_PROBS = {
    "quiet":   (0.70, 0.28, 0.02),   # 04-08 UTC — mempool drains
    "us_open": (0.12, 0.65, 0.23),   # 12-16 UTC — aggressive build
    "peak":    (0.18, 0.60, 0.22),   # 18-22 UTC — saturation
    "asian":   (0.45, 0.50, 0.05),   # 00-04 UTC — secondary bump
    "other":   (0.40, 0.55, 0.05),   # everything else
}

def _session(hour: int) -> str:
    if  4 <= hour <  8: return "quiet"
    if 12 <= hour < 16: return "us_open"
    if 18 <= hour < 22: return "peak"
    if  0 <= hour <  4: return "asian"
    return "other"


def _sample_fee_layer(hour: int, weekday: int) -> float:
    """
    Sample a fee rate by first choosing which layer is active,
    then sampling within that layer's range.
    Filler:   1.0 – 3.0  sat/vB
    Utility:  4.0 – 15.0 sat/vB  (intraday-shaped)
    Priority: 15.0 – 40.0 sat/vB (L2 / institutional)
    """
    sess = _session(hour)
    p_fill, p_util, p_prio = _LAYER_PROBS[sess]

    # Weekends shift weight toward filler
    if weekday >= 5:
        p_fill = min(1.0, p_fill + 0.20)
        p_prio = max(0.0, p_prio - 0.15)
        p_util = max(0.0, 1.0 - p_fill - p_prio)

    layer = random.choices(
        ["filler", "utility", "priority"],
        weights=[p_fill, p_util, p_prio]
    )[0]

    if layer == "filler":
        return random.uniform(ORDINALS_FLOOR, 3.0)
    elif layer == "utility":
        # Utility fee peaks during us_open/peak, lower during quiet
        if sess == "quiet":
            return random.uniform(2.5, 8.0)
        elif sess in ("us_open", "peak"):
            return random.uniform(6.0, 15.0)
        else:
            return random.uniform(3.0, 10.0)
    else:  # priority — L2 settlement / institutional
        return random.uniform(15.0, 38.0)


def _is_last_friday(ts: datetime.datetime) -> bool:
    """True if `ts` falls on the last Friday of its month."""
    last_day = _calendar.monthrange(ts.year, ts.month)[1]
    return ts.weekday() == 4 and (last_day - ts.day) < 7


@app.get("/bootstrap")
def bootstrap_endpoint():
    if not supabase_client:
        return {"error": "Supabase not configured"}

    now = datetime.datetime.now(timezone.utc)
    rows = 4032  # 14 days × 288 five-minute intervals

    sentiment = 0.0
    records_to_insert = []

    #  Build congestion-spike windows 
    # ~8 organic spikes over 14 days; each lasts 2–6 hours (24-72 5-min ticks)
    spike_windows: list[tuple[int, int]] = []
    current_step = 0
    while current_step < rows:
        if random.random() < (8 / rows):
            dur = random.randint(24, 72)
            spike_windows.append((current_step, current_step + dur))
            current_step += dur
        current_step += 1


    #  State Initialization for Simulation 
    cur_fee = ORDINALS_FLOOR
    cur_txs = 15000.0
    cur_size = 5.0
    time_since_block = 0
    
    for i in range(rows - 1, -1, -1):
        ts       = now - datetime.timedelta(minutes=5 * i)
        step_idx = rows - 1 - i
        hour     = ts.hour
        weekday  = ts.weekday()
        
        in_spike     = any(st <= step_idx <= en for st, en in spike_windows)
        is_last_fri  = _is_last_friday(ts)
        is_sun_clear = (weekday == 6 and 22 <= hour) or (weekday == 0 and hour < 2)
        
        #  Target State Calculation 
        if in_spike:
            target_fee = random.uniform(25.0, 42.0)
            target_txs = random.uniform(40000, 65000)
            inertia = 0.15 # slower move into spikes
        elif is_sun_clear:
            target_fee = ORDINALS_FLOOR
            target_txs = random.uniform(3000, 8000)
            inertia = 0.3 # faster clear
        elif is_last_fri:
            target_fee = random.uniform(12.0, 22.0)
            target_txs = random.uniform(25000, 45000)
            inertia = 0.1
        else:
            # Baseline based on session
            target_fee = _sample_fee_layer(hour, weekday)
            sess = _session(hour)
            if sess == "us_open": target_txs = random.uniform(35000, 55000)
            elif sess == "peak": target_txs = random.uniform(25000, 45000)
            elif sess == "quiet": target_txs = random.uniform(5000, 12000)
            else: target_txs = random.uniform(12000, 25000)
            inertia = 0.08 # very smooth transitions

        #  State Update (Persistence/Inertia) 
        # Move current state toward target (EMA style)
        cur_fee = cur_fee + (target_fee - cur_fee) * inertia + random.gauss(0, 0.2)
        cur_txs = cur_txs + (target_txs - cur_txs) * inertia + random.gauss(0, 500)
        
        # Clamp
        cur_fee = min(FEE_CAP, max(ORDINALS_FLOOR, cur_fee))
        cur_txs = max(1000, cur_txs)
        
        #  Block Mining Logic (Sawtooth V-Shape) 
        # Blocks happen every 10 mins on average (Poisson-ish)
        # Every 2 steps (10 mins), a block is mined
        time_since_block += 5
        mined_this_step = False
        if time_since_block >= 10:
            if random.random() < 0.8: # 80% chance every 10 mins
                mined_this_step = True
                time_since_block = 0
        
        # Growth: transactions arrive
        # Drains: blocks clear ~2500 txs / 1.5 MB
        if mined_this_step:
            drain_txs = random.randint(2200, 2800)
            cur_txs = max(2000, cur_txs - drain_txs)
            
        # Recalculate size based on tx count and session-based size
        avg_tx_size = 350.0 # base 2026 SegWit avg
        if in_spike: avg_tx_size = 800.0 # heavy L2 batches
        elif weekday >= 5: avg_tx_size = 280.0 # light retail
        
        cur_size = (cur_txs * avg_tx_size / 1_000_000.0)
        cur_size = min(80.0, max(1.0, cur_size))

        median_fee_rate = cur_fee
        tx_count = int(cur_txs)
        total_size_mb = round(cur_size, 3)

        #  Fee dispersion (IQR) 
        # During quiet/filler periods p10≈p90 (tight spread)
        # During spikes p90 rockets while p10 stays at filler floor
        if in_spike:
            p10 = max(ORDINALS_FLOOR, median_fee_rate * random.uniform(0.25, 0.45))
            p90 = min(FEE_CAP, median_fee_rate * random.uniform(2.0, 3.5))
        elif median_fee_rate < 5:  # filler-dominated
            p10 = max(ORDINALS_FLOOR * 0.8, median_fee_rate * random.uniform(0.70, 0.90))
            p90 = median_fee_rate * random.uniform(1.2, 1.8)
        else:
            p10 = max(ORDINALS_FLOOR * 0.8, median_fee_rate * random.uniform(0.50, 0.70))
            p90 = min(FEE_CAP, median_fee_rate * random.uniform(1.5, 2.5))

        high_fee_pct    = min(100.0, max(0.0, (median_fee_rate - 10) * 3))
        tx_arrival_rate = tx_count / 5.0

        fee_cluster = _assign_fee_cluster(median_fee_rate)

        #  Sentiment: drifts; spikes + expiry push bearish 
        sentiment += random.gauss(0, 0.025)
        if in_spike or is_last_fri:
            sentiment += (-0.35 - sentiment) * 0.25
        sentiment = max(-1.0, min(1.0, sentiment))

        score_velocity = random.gauss(0, 0.012)

        # News volume peaks during EU/US hours; spikes around events
        article_volume = int(
            4 + 14 * (1 if 13 <= hour <= 20 else 0.3)
            + (8 if in_spike or is_last_fri else 0)
            + random.uniform(-2, 2)
        )
        article_volume = max(1, article_volume)

        cats    = ["price_action", "regulation", "etf", "adoption",
                   "hack_or_fud", "mining", "macro"]
        weights = [1] * len(cats)
        if in_spike or is_last_fri:
            weights[cats.index("hack_or_fud")] = 5
            weights[cats.index("price_action")] = 3
        dominant_topic = random.choices(cats, weights=weights)[0]

        #  Fee predictions 
        sent_mult = (
            1.04 if sentiment > 0.4
            else 0.96 if sentiment < -0.4
            else 1.0
        )
        fee_1block = max(
            ORDINALS_FLOOR,
            min(FEE_CAP, median_fee_rate * random.uniform(1.05, 1.20) * sent_mult + random.gauss(0, 1.0)),
        )
        fee_3block = max(
            ORDINALS_FLOOR,
            min(FEE_CAP, median_fee_rate * random.uniform(0.85, 1.05) * sent_mult + random.gauss(0, 0.8)),
        )
        fee_6block = max(
            ORDINALS_FLOOR,
            min(FEE_CAP, median_fee_rate * random.uniform(0.65, 0.85) * sent_mult + random.gauss(0, 0.5)),
        )

        actual_fee        = max(ORDINALS_FLOOR, min(FEE_CAP, fee_1block + random.gauss(0, 1.5)))
        blocks_to_confirm = random.randint(5, 10) if in_spike else random.randint(1, 3)

        records_to_insert.append(
            {
                "ts": ts.isoformat(),
                "m": {
                    "tx_count":          tx_count,
                    "total_size_mb":     total_size_mb,
                    "median_fee_rate":   median_fee_rate,
                    "p10_fee_rate":      round(p10, 2),
                    "p90_fee_rate":      round(p90, 2),
                    "high_fee_pct":      high_fee_pct,
                    "avg_tx_size_bytes": avg_tx_size,
                    "tx_arrival_rate":   tx_arrival_rate,
                    "fee_cluster":       fee_cluster,
                },
                "s": {
                    "score":          sentiment,
                    "score_velocity": score_velocity,
                    "article_volume": article_volume,
                    "dominant_topic": dominant_topic,
                    "source_weight":  1.1,
                },
                "f": {
                    "tx_count":           tx_count,
                    "median_fee_rate":    median_fee_rate,
                    "total_size_mb":      total_size_mb,
                    "fee_cluster":        fee_cluster,
                    "sentiment_score":    sentiment,
                    "sentiment_velocity": score_velocity,
                    "article_volume":     article_volume,
                },
                "p": {
                    "fee_1block":    round(fee_1block, 2),
                    "fee_3block":    round(fee_3block, 2),
                    "fee_6block":    round(fee_6block, 2),
                    "confidence":    0.5,
                    "model_version": "synthetic-v2",
                },
                "a": {
                    "actual_fee_paid":   round(actual_fee, 2),
                    "blocks_to_confirm": blocks_to_confirm,
                },
            }
        )

    logger.info(f"Generated {len(records_to_insert)} rows. Starting inserts...")

    for i in range(0, len(records_to_insert), 100):
        batch = records_to_insert[i : i + 100]

        m_batch = [{"captured_at": r["ts"], **r["m"]} for r in batch]
        m_res = supabase_client.table("mempool_snapshots").insert(m_batch).execute()

        s_batch = [{"captured_at": r["ts"], **r["s"]} for r in batch]
        s_res = supabase_client.table("sentiment_snapshots").insert(s_batch).execute()

        f_batch = []
        for j, r in enumerate(batch):
            f_doc = {"captured_at": r["ts"], **r["f"]}
            f_doc["mempool_snapshot_id"] = m_res.data[j]["id"]
            f_doc["sentiment_snapshot_id"] = s_res.data[j]["id"]
            f_batch.append(f_doc)
        f_res = supabase_client.table("features").insert(f_batch).execute()

        p_batch = []
        for j, r in enumerate(batch):
            p_doc = {"predicted_at": r["ts"], **r["p"]}
            p_doc["feature_id"] = f_res.data[j]["id"]
            p_batch.append(p_doc)
        p_res = supabase_client.table("predictions").insert(p_batch).execute()

        a_batch = []
        for j, r in enumerate(batch):
            a_doc = {"confirmed_at": r["ts"], **r["a"]}
            a_doc["prediction_id"] = p_res.data[j]["id"]
            a_batch.append(a_doc)
        supabase_client.table("actuals").insert(a_batch).execute()

        if (i + 100) % 500 == 0:
            logger.info(f"Inserted {i+100} rows.")

    logger.info("Calling /train internally...")
    train_result = train_endpoint()

    return {"inserted": 4032, "training": train_result}


@app.get("/news")
async def get_news():
    if not LATEST_NEWS:
        await fetch_sentiment_data()
    return LATEST_NEWS[:3]
