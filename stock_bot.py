"""
AI Stock News + Trend Scoring Bot
Research and paper-trading only. No real trades.

Install:
pip install pandas numpy yfinance requests vaderSentiment pytrends streamlit

Optional:
Set API keys in environment variables:
NEWS_API_KEY
GNEWS_API_KEY

Run:
python stock_bot.py

Dashboard:
streamlit run stock_bot.py
"""

import os
import sqlite3
import json
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import yfinance as yf

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None


# =========================
# CONFIG
# =========================

WATCHLIST = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMD": "AMD",
    "TSLA": "Tesla",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet Google",
    "AMZN": "Amazon",
    "PLTR": "Palantir",
    "COIN": "Coinbase",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
    "AVGO": "Broadcom",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "ORCL": "Oracle",
    "ADBE": "Adobe",
    "UBER": "Uber",
    "SHOP": "Shopify",
    "LLY": "Eli Lilly",
    "UNH": "UnitedHealth",
    "V": "Visa",
    "MA": "Mastercard",
    "COST": "Costco",
    "WMT": "Walmart",
    "HD": "Home Depot",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
}

DB_PATH = "stock_bot.db"

STARTING_CASH = 10_000
BUY_THRESHOLD = 75
SELL_THRESHOLD = 45
EMERGENCY_DROP = 25
MAX_POSITION_SIZE = 0.10

ENABLE_REAL_TRADING = False
ALLOW_EARNINGS_TRADES = False

NEWS_NOISE_TERMS = [
    "mlb", "nba", "nfl", "nhl", "soccer", "baseball", "football",
    "basketball", "cricket", "ufc", "wwe", "celebrity", "movie",
    "music", "album", "concert", "lyrics", "recipe", "horoscope"
]

MARKET_TERMS = [
    "stock", "shares", "earnings", "revenue", "profit", "forecast",
    "guidance", "analyst", "upgrade", "downgrade", "price target",
    "market", "nasdaq", "nyse", "investor", "dividend", "valuation",
    "ai", "chip", "cloud", "sales", "quarter", "growth"
]


# =========================
# DATABASE
# =========================

class Database:
    def __init__(self, path: str = DB_PATH):
        self.conn = sqlite3.connect(path)
        self.create_tables()

    def create_tables(self):
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            return_1d REAL,
            return_5d REAL,
            return_20d REAL,
            ma_5 REAL,
            ma_20 REAL,
            ma_50 REAL,
            rsi REAL,
            volume_ratio REAL,
            PRIMARY KEY (ticker, date)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            title TEXT,
            source TEXT,
            url TEXT UNIQUE,
            published_at TEXT,
            sentiment REAL,
            surprise_type TEXT,
            created_at TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            ticker TEXT,
            date TEXT,
            final_score REAL,
            news_score REAL,
            trend_score REAL,
            momentum_score REAL,
            volume_score REAL,
            macro_risk_score REAL,
            earnings_risk_score REAL,
            confidence_score REAL,
            signal TEXT,
            reasons TEXT,
            PRIMARY KEY (ticker, date)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            action TEXT,
            shares REAL,
            price REAL,
            cash_after REAL,
            portfolio_value REAL,
            reason TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            ticker TEXT PRIMARY KEY,
            shares REAL,
            avg_price REAL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS cash (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash REAL
        )
        """)

        cur.execute("""
        INSERT OR IGNORE INTO cash (id, cash)
        VALUES (1, ?)
        """, (STARTING_CASH,))

        self.conn.commit()

    def save_market_data(self, ticker: str, df: pd.DataFrame):
        cur = self.conn.cursor()

        for date, row in df.iterrows():
            cur.execute("""
            INSERT OR REPLACE INTO market_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                str(date.date()),
                safe_float(row.get("Open")),
                safe_float(row.get("High")),
                safe_float(row.get("Low")),
                safe_float(row.get("Close")),
                safe_float(row.get("Volume")),
                safe_float(row.get("return_1d")),
                safe_float(row.get("return_5d")),
                safe_float(row.get("return_20d")),
                safe_float(row.get("ma_5")),
                safe_float(row.get("ma_20")),
                safe_float(row.get("ma_50")),
                safe_float(row.get("rsi")),
                safe_float(row.get("volume_ratio")),
            ))

        self.conn.commit()

    def save_news(self, item: Dict):
        cur = self.conn.cursor()

        try:
            cur.execute("""
            INSERT OR IGNORE INTO news
            (ticker, title, source, url, published_at, sentiment, surprise_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["ticker"],
                item["title"],
                item["source"],
                item["url"],
                item["published_at"],
                item.get("sentiment"),
                item.get("surprise_type"),
                now_date(),
            ))
            self.conn.commit()
        except Exception:
            pass

    def save_score(self, score: Dict):
        cur = self.conn.cursor()

        cur.execute("""
        INSERT OR REPLACE INTO scores
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            score["ticker"],
            score["date"],
            score["final_score"],
            score["news_score"],
            score["trend_score"],
            score["momentum_score"],
            score["volume_score"],
            score["macro_risk_score"],
            score["earnings_risk_score"],
            score["confidence_score"],
            score["signal"],
            json.dumps(score["reasons"]),
        ))

        self.conn.commit()

    def latest_market_row(self, ticker: str) -> Optional[Dict]:
        cur = self.conn.cursor()

        cur.execute("""
        SELECT *
        FROM market_data
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 1
        """, (ticker,))

        row = cur.fetchone()

        if not row:
            return None

        cols = [x[0] for x in cur.description]
        return dict(zip(cols, row))

    def recent_news(self, ticker: str, days: int = 3) -> List[Dict]:
        cur = self.conn.cursor()

        cutoff = dt.datetime.now() - dt.timedelta(days=days)

        cur.execute("""
        SELECT ticker, title, source, url, published_at, sentiment, surprise_type
        FROM news
        WHERE ticker = ?
        AND published_at >= ?
        ORDER BY published_at DESC
        LIMIT 20
        """, (ticker, cutoff.isoformat()))

        rows = cur.fetchall()
        cols = [x[0] for x in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def latest_score(self, ticker: str) -> Optional[Dict]:
        cur = self.conn.cursor()

        cur.execute("""
        SELECT *
        FROM scores
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 1
        """, (ticker,))

        row = cur.fetchone()

        if not row:
            return None

        cols = [x[0] for x in cur.description]
        data = dict(zip(cols, row))

        try:
            data["reasons"] = json.loads(data["reasons"])
        except Exception:
            data["reasons"] = []

        return data

    def previous_score(self, ticker: str) -> Optional[Dict]:
        cur = self.conn.cursor()

        cur.execute("""
        SELECT *
        FROM scores
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 1 OFFSET 1
        """, (ticker,))

        row = cur.fetchone()

        if not row:
            return None

        cols = [x[0] for x in cur.description]
        return dict(zip(cols, row))

    def get_cash(self) -> float:
        cur = self.conn.cursor()
        cur.execute("SELECT cash FROM cash WHERE id = 1")
        return float(cur.fetchone()[0])

    def set_cash(self, cash: float):
        cur = self.conn.cursor()
        cur.execute("UPDATE cash SET cash = ? WHERE id = 1", (cash,))
        self.conn.commit()

    def get_position(self, ticker: str) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT ticker, shares, avg_price FROM portfolio WHERE ticker = ?", (ticker,))
        row = cur.fetchone()

        if not row:
            return None

        return {
            "ticker": row[0],
            "shares": float(row[1]),
            "avg_price": float(row[2]),
        }

    def upsert_position(self, ticker: str, shares: float, avg_price: float):
        cur = self.conn.cursor()

        if shares <= 0:
            cur.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))
        else:
            cur.execute("""
            INSERT OR REPLACE INTO portfolio
            VALUES (?, ?, ?)
            """, (ticker, shares, avg_price))

        self.conn.commit()

    def log_trade(self, trade: Dict):
        cur = self.conn.cursor()

        cur.execute("""
        INSERT INTO paper_trades
        (date, ticker, action, shares, price, cash_after, portfolio_value, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["date"],
            trade["ticker"],
            trade["action"],
            trade["shares"],
            trade["price"],
            trade["cash_after"],
            trade["portfolio_value"],
            trade["reason"],
        ))

        self.conn.commit()


# =========================
# HELPERS
# =========================

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def now_date() -> str:
    return dt.datetime.now().isoformat()


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def normalize_percent(value: float, low: float, high: float) -> float:
    if value is None or math.isnan(value):
        return 50

    value = max(low, min(high, value))
    return ((value - low) / (high - low)) * 100


# =========================
# MARKET DATA
# =========================

class MarketDataCollector:
    def __init__(self, db: Database):
        self.db = db

    def fetch(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

        if df.empty:
            return df

        # yfinance sometimes returns MultiIndex columns like ("Close", "AAPL").
        # Flatten them so the rest of the app can read normal names like Close, Open, Volume.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df.set_index("Date")

        df["return_1d"] = df["Close"].pct_change()
        df["return_5d"] = df["Close"].pct_change(5)
        df["return_20d"] = df["Close"].pct_change(20)

        df["ma_5"] = df["Close"].rolling(5).mean()
        df["ma_20"] = df["Close"].rolling(20).mean()
        df["ma_50"] = df["Close"].rolling(50).mean()

        df["rsi"] = self.calculate_rsi(df["Close"])
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        self.db.save_market_data(ticker, df)
        return df

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


# =========================
# NEWS
# =========================

class NewsCollector:
    def __init__(self, db: Database):
        self.db = db
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.gnews_api_key = os.getenv("GNEWS_API_KEY")
        self.sentiment = SentimentEngine()

    def fetch_for_ticker(self, ticker: str, company_name: str):
        articles = []

        if self.news_api_key:
            articles.extend(self.fetch_newsapi(ticker, company_name))

        if self.gnews_api_key:
            articles.extend(self.fetch_gnews(ticker, company_name))

        articles.extend(self.fetch_rss_fallback(ticker, company_name))

        filtered_articles = []

        for article in articles:
            if not is_relevant_stock_news(article, ticker, company_name):
                continue

            article["sentiment"] = self.sentiment.score(article["title"])
            article["surprise_type"] = detect_surprise_type(article["title"])
            self.db.save_news(article)
            filtered_articles.append(article)

        return filtered_articles

    def fetch_newsapi(self, ticker: str, company_name: str) -> List[Dict]:
        query = f'"{company_name}" OR "{ticker}" stock earnings revenue guidance'

        url = "https://newsapi.org/v2/everything"

        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": self.news_api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        except Exception:
            return []

        articles = []

        for item in data.get("articles", []):
            articles.append({
                "ticker": ticker,
                "title": item.get("title") or "",
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "url": item.get("url") or "",
                "published_at": item.get("publishedAt") or now_date(),
            })

        return articles

    def fetch_gnews(self, ticker: str, company_name: str) -> List[Dict]:
        query = f'{company_name} {ticker} stock earnings'

        url = "https://gnews.io/api/v4/search"

        params = {
            "q": query,
            "lang": "en",
            "max": 20,
            "apikey": self.gnews_api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        except Exception:
            return []

        articles = []

        for item in data.get("articles", []):
            articles.append({
                "ticker": ticker,
                "title": item.get("title") or "",
                "source": item.get("source", {}).get("name", "GNews"),
                "url": item.get("url") or "",
                "published_at": item.get("publishedAt") or now_date(),
            })

        return articles

    def fetch_rss_fallback(self, ticker: str, company_name: str) -> List[Dict]:
        query = f'"{company_name}"+{ticker}+stock+earnings+shares' 
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        try:
            response = requests.get(url, timeout=10)
            text = response.text
        except Exception:
            return []

        articles = []
        parts = text.split("<item>")[1:20]

        for part in parts:
            title = extract_between(part, "<title>", "</title>")
            link = extract_between(part, "<link>", "</link>")
            pub_date = extract_between(part, "<pubDate>", "</pubDate>")

            if title:
                articles.append({
                    "ticker": ticker,
                    "title": clean_html(title),
                    "source": "Google News RSS",
                    "url": link or f"rss:{ticker}:{hash(title)}",
                    "published_at": parse_rss_date(pub_date),
                })

        return articles



def is_relevant_stock_news(article: Dict, ticker: str, company_name: str) -> bool:
    title = (article.get("title") or "").lower()
    source = (article.get("source") or "").lower()
    company_words = [w.lower() for w in company_name.replace("-", " ").split() if len(w) > 2]

    if any(term in title or term in source for term in NEWS_NOISE_TERMS):
        return False

    has_company_match = ticker.lower() in title or any(word in title for word in company_words)
    has_market_context = any(term in title for term in MARKET_TERMS)

    return has_company_match and has_market_context

def extract_between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except Exception:
        return ""


def clean_html(text: str) -> str:
    return (
        text.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def parse_rss_date(value: str) -> str:
    try:
        parsed = dt.datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
        return parsed.isoformat()
    except Exception:
        return now_date()


# =========================
# SENTIMENT
# =========================

class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

    def score(self, text: str) -> float:
        if not text:
            return 0

        if self.analyzer:
            return float(self.analyzer.polarity_scores(text)["compound"])

        positive_words = [
            "beat", "beats", "surge", "surges", "growth", "record",
            "upgrade", "bullish", "profit", "strong", "rally", "higher",
            "partnership", "launch", "approval", "outperform"
        ]

        negative_words = [
            "miss", "misses", "drop", "falls", "lawsuit", "bearish",
            "downgrade", "weak", "loss", "cut", "layoff", "probe",
            "investigation", "warning", "risk", "lower"
        ]

        t = text.lower()
        pos = sum(1 for word in positive_words if word in t)
        neg = sum(1 for word in negative_words if word in t)

        if pos + neg == 0:
            return 0

        return (pos - neg) / (pos + neg)


def detect_surprise_type(text: str) -> str:
    t = text.lower()

    positive_surprise = [
        "beats expectations", "better than expected", "raises guidance",
        "unexpected profit", "surprise profit", "record revenue",
        "stronger than expected"
    ]

    negative_surprise = [
        "misses expectations", "worse than expected", "cuts guidance",
        "unexpected loss", "surprise loss", "weaker than expected",
        "warning", "plunges"
    ]

    expected_positive = [
        "expected growth", "anticipated growth", "forecast profit"
    ]

    expected_negative = [
        "expected loss", "anticipated decline", "forecast decline"
    ]

    if any(x in t for x in positive_surprise):
        return "unexpected_positive"

    if any(x in t for x in negative_surprise):
        return "unexpected_negative"

    if any(x in t for x in expected_positive):
        return "expected_positive"

    if any(x in t for x in expected_negative):
        return "expected_negative"

    return "unclear"


# =========================
# SOCIAL TRENDS
# =========================

class TrendCollector:
    def __init__(self):
        self.pytrends_available = TrendReq is not None

    def google_trend_score(self, keyword: str) -> float:
        if not self.pytrends_available:
            return 50

        try:
            pytrends = TrendReq(hl="en-US", tz=360)
            pytrends.build_payload([keyword], timeframe="now 7-d")
            df = pytrends.interest_over_time()

            if df.empty or keyword not in df.columns:
                return 50

            recent = df[keyword].tail(2).mean()
            average = df[keyword].mean()

            if average == 0:
                return 50

            ratio = recent / average

            if ratio >= 1.8:
                return 90
            if ratio >= 1.4:
                return 75
            if ratio >= 1.1:
                return 60
            if ratio >= 0.8:
                return 50
            if ratio >= 0.5:
                return 35

            return 20

        except Exception:
            return 50


# =========================
# SCORING
# =========================

class StockScorer:
    def __init__(self, db: Database):
        self.db = db
        self.trends = TrendCollector()

    def score_stock(self, ticker: str, company_name: str) -> Dict:
        market = self.db.latest_market_row(ticker)
        news = self.db.recent_news(ticker)

        if not market:
            return self.empty_score(ticker, "No market data available")

        news_score = self.calculate_news_score(news)
        trend_score = self.trends.google_trend_score(company_name)
        momentum_score = self.calculate_momentum_score(market)
        volume_score = self.calculate_volume_score(market)
        macro_risk_score = self.calculate_macro_risk_score(news)
        earnings_risk_score = self.calculate_earnings_risk_score(news)
        confidence_score = self.calculate_confidence_score(news, market)

        final_score = (
            news_score * 0.30 +
            trend_score * 0.15 +
            momentum_score * 0.20 +
            volume_score * 0.10 +
            macro_risk_score * 0.10 +
            earnings_risk_score * 0.10 +
            confidence_score * 0.05
        )

        signal = self.get_signal(ticker, final_score, market)

        reasons = self.generate_reasons(
            news_score,
            trend_score,
            momentum_score,
            volume_score,
            macro_risk_score,
            earnings_risk_score,
            confidence_score,
            signal,
            news,
            market,
        )

        score = {
            "ticker": ticker,
            "date": now_date()[:10],
            "final_score": round(final_score, 2),
            "news_score": round(news_score, 2),
            "trend_score": round(trend_score, 2),
            "momentum_score": round(momentum_score, 2),
            "volume_score": round(volume_score, 2),
            "macro_risk_score": round(macro_risk_score, 2),
            "earnings_risk_score": round(earnings_risk_score, 2),
            "confidence_score": round(confidence_score, 2),
            "signal": signal,
            "reasons": reasons,
        }

        self.db.save_score(score)
        return score

    def empty_score(self, ticker: str, reason: str) -> Dict:
        score = {
            "ticker": ticker,
            "date": now_date()[:10],
            "final_score": 50,
            "news_score": 50,
            "trend_score": 50,
            "momentum_score": 50,
            "volume_score": 50,
            "macro_risk_score": 50,
            "earnings_risk_score": 50,
            "confidence_score": 0,
            "signal": "HOLD",
            "reasons": [reason],
        }

        self.db.save_score(score)
        return score

    def calculate_news_score(self, news: List[Dict]) -> float:
        if not news:
            return 50

        weighted_scores = []

        for item in news:
            sentiment = item.get("sentiment") or 0
            surprise = item.get("surprise_type")

            base = 50 + sentiment * 50

            if surprise == "unexpected_positive":
                base += 15
            elif surprise == "unexpected_negative":
                base -= 20
            elif surprise == "expected_positive":
                base += 5
            elif surprise == "expected_negative":
                base -= 5

            weighted_scores.append(clamp(base))

        return float(np.mean(weighted_scores))

    def calculate_momentum_score(self, market: Dict) -> float:
        r5 = market.get("return_5d")
        r20 = market.get("return_20d")
        close = market.get("close")
        ma20 = market.get("ma_20")
        rsi = market.get("rsi")

        score = 50

        if r5 is not None:
            score += normalize_percent(r5, -0.10, 0.10) * 0.25 - 12.5

        if r20 is not None:
            score += normalize_percent(r20, -0.20, 0.20) * 0.35 - 17.5

        if close and ma20:
            score += 10 if close > ma20 else -10

        if rsi:
            if rsi > 80:
                score -= 15
            elif rsi > 70:
                score -= 8
            elif rsi < 25:
                score -= 10
            elif 45 <= rsi <= 65:
                score += 5

        return clamp(score)

    def calculate_volume_score(self, market: Dict) -> float:
        volume_ratio = market.get("volume_ratio")
        daily_return = market.get("return_1d")

        if not volume_ratio:
            return 50

        if volume_ratio > 2 and daily_return and daily_return > 0:
            return 85

        if volume_ratio > 2 and daily_return and daily_return < 0:
            return 25

        if volume_ratio > 1.5:
            return 65

        if volume_ratio < 0.7:
            return 40

        return 50

    def calculate_macro_risk_score(self, news: List[Dict]) -> float:
        if not news:
            return 50

        risk_terms = [
            "inflation", "rate hike", "higher rates", "recession",
            "tariff", "war", "regulation", "lawsuit", "probe",
            "federal reserve", "cpi", "unemployment"
        ]

        joined = " ".join([x.get("title", "").lower() for x in news])
        risk_hits = sum(1 for term in risk_terms if term in joined)

        if risk_hits >= 5:
            return 20
        if risk_hits >= 3:
            return 35
        if risk_hits >= 1:
            return 45

        return 65

    def calculate_earnings_risk_score(self, news: List[Dict]) -> float:
        if not news:
            return 60

        earnings_terms = [
            "earnings", "guidance", "revenue", "profit", "eps",
            "quarterly results", "forecast"
        ]

        joined = " ".join([x.get("title", "").lower() for x in news])
        hits = sum(1 for term in earnings_terms if term in joined)

        if hits >= 3 and not ALLOW_EARNINGS_TRADES:
            return 35

        if hits >= 1:
            return 50

        return 65

    def calculate_confidence_score(self, news: List[Dict], market: Dict) -> float:
        score = 40

        if len(news) >= 10:
            score += 30
        elif len(news) >= 5:
            score += 20
        elif len(news) >= 1:
            score += 10

        if market.get("close") and market.get("ma_20"):
            score += 15

        if market.get("volume_ratio"):
            score += 15

        return clamp(score)

    def get_signal(self, ticker: str, score: float, market: Dict) -> str:
        close = market.get("close")
        ma20 = market.get("ma_20")

        previous = self.db.previous_score(ticker)

        if previous:
            previous_score = previous.get("final_score")
            if previous_score is not None and previous_score - score >= EMERGENCY_DROP:
                return "EMERGENCY_SELL"

        if score >= BUY_THRESHOLD:
            if close and ma20 and close > ma20:
                return "BUY"
            return "WATCH"

        if score <= SELL_THRESHOLD:
            return "SELL"

        return "HOLD"

    def generate_reasons(
        self,
        news_score,
        trend_score,
        momentum_score,
        volume_score,
        macro_risk_score,
        earnings_risk_score,
        confidence_score,
        signal,
        news,
        market,
    ) -> List[str]:
        reasons = []

        if news_score >= 70:
            reasons.append("News sentiment is strongly bullish.")
        elif news_score <= 40:
            reasons.append("News sentiment is bearish or risk-heavy.")
        else:
            reasons.append("News sentiment is neutral.")

        if momentum_score >= 70:
            reasons.append("Price momentum is strong.")
        elif momentum_score <= 40:
            reasons.append("Price momentum is weak.")
        else:
            reasons.append("Price momentum is neutral.")

        if trend_score >= 70:
            reasons.append("Search trend interest is rising.")
        elif trend_score <= 40:
            reasons.append("Search trend interest is weak.")
        else:
            reasons.append("Search trend interest is normal.")

        if volume_score >= 70:
            reasons.append("Volume is unusually strong on positive movement.")
        elif volume_score <= 40:
            reasons.append("Volume suggests caution.")

        if macro_risk_score <= 40:
            reasons.append("Macro or regulatory risk is showing in recent headlines.")

        if earnings_risk_score <= 40:
            reasons.append("Earnings-related risk is elevated.")

        if news:
            reasons.append(f"Top headline: {news[0].get('title')}")

        reasons.append(f"Final signal: {signal}")

        return reasons[:5]


# =========================
# PAPER TRADER
# =========================

class PaperTrader:
    def __init__(self, db: Database):
        self.db = db

    def portfolio_value(self) -> float:
        cash = self.db.get_cash()
        cur = self.db.conn.cursor()

        cur.execute("SELECT ticker, shares FROM portfolio")
        positions = cur.fetchall()

        total = cash

        for ticker, shares in positions:
            market = self.db.latest_market_row(ticker)

            if market and market.get("close"):
                total += float(shares) * float(market["close"])

        return total

    def execute_signal(self, score: Dict):
        ticker = score["ticker"]
        signal = score["signal"]

        market = self.db.latest_market_row(ticker)

        if not market or not market.get("close"):
            return

        price = float(market["close"])
        cash = self.db.get_cash()
        total_value = self.portfolio_value()
        position = self.db.get_position(ticker)

        if signal == "BUY":
            if position:
                return

            max_dollars = total_value * MAX_POSITION_SIZE
            dollars_to_spend = min(cash, max_dollars)

            if dollars_to_spend < 50:
                return

            shares = dollars_to_spend / price
            cash_after = cash - dollars_to_spend

            self.db.set_cash(cash_after)
            self.db.upsert_position(ticker, shares, price)

            self.db.log_trade({
                "date": now_date(),
                "ticker": ticker,
                "action": "BUY",
                "shares": shares,
                "price": price,
                "cash_after": cash_after,
                "portfolio_value": self.portfolio_value(),
                "reason": "Score crossed buy threshold.",
            })

        elif signal in ["SELL", "EMERGENCY_SELL"]:
            if not position:
                return

            shares = position["shares"]
            proceeds = shares * price
            cash_after = cash + proceeds

            self.db.set_cash(cash_after)
            self.db.upsert_position(ticker, 0, 0)

            self.db.log_trade({
                "date": now_date(),
                "ticker": ticker,
                "action": signal,
                "shares": shares,
                "price": price,
                "cash_after": cash_after,
                "portfolio_value": self.portfolio_value(),
                "reason": "Score dropped below sell threshold.",
            })


# =========================
# BACKTEST
# =========================

class Backtester:
    def __init__(self):
        pass

    def simple_momentum_backtest(
        self,
        ticker: str,
        period: str = "2y",
        buy_threshold_return_20d: float = 0.08,
        sell_threshold_return_20d: float = -0.03,
    ) -> Dict:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

        if df.empty:
            return {"ticker": ticker, "error": "No data"}

        df["return_20d"] = df["Close"].pct_change(20)
        df["daily_return"] = df["Close"].pct_change()

        cash = 10_000
        shares = 0
        trades = []

        for date, row in df.iterrows():
            price = float(row["Close"])
            signal_metric = row["return_20d"]

            if pd.isna(signal_metric):
                continue

            if shares == 0 and signal_metric >= buy_threshold_return_20d:
                shares = cash / price
                cash = 0
                trades.append(("BUY", date, price))

            elif shares > 0 and signal_metric <= sell_threshold_return_20d:
                cash = shares * price
                shares = 0
                trades.append(("SELL", date, price))

        final_price = float(df["Close"].iloc[-1])
        final_value = cash + shares * final_price

        buy_hold = 10_000 / float(df["Close"].iloc[0]) * final_price

        return {
            "ticker": ticker,
            "strategy_final_value": round(final_value, 2),
            "strategy_return_percent": round((final_value / 10_000 - 1) * 100, 2),
            "buy_hold_final_value": round(buy_hold, 2),
            "buy_hold_return_percent": round((buy_hold / 10_000 - 1) * 100, 2),
            "number_of_trades": len(trades),
            "trades": trades[-10:],
        }


# =========================
# MAIN BOT
# =========================

class StockBot:
    def __init__(self):
        self.db = Database()
        self.market = MarketDataCollector(self.db)
        self.news = NewsCollector(self.db)
        self.scorer = StockScorer(self.db)
        self.trader = PaperTrader(self.db)

    def run_daily_update(self):
        results = []

        for ticker, company in WATCHLIST.items():
            print(f"\nUpdating {ticker} - {company}")

            try:
                self.market.fetch(ticker)
                print("Market data updated.")
            except Exception as e:
                print(f"Market data failed: {e}")

            try:
                articles = self.news.fetch_for_ticker(ticker, company)
                print(f"News articles found: {len(articles)}")
            except Exception as e:
                print(f"News failed: {e}")

            try:
                score = self.scorer.score_stock(ticker, company)
                print(f"Score: {score['final_score']} | Signal: {score['signal']}")
                for reason in score["reasons"]:
                    print(f"- {reason}")

                self.trader.execute_signal(score)
                results.append(score)

            except Exception as e:
                print(f"Scoring failed: {e}")

        print("\nPortfolio value:", round(self.trader.portfolio_value(), 2))
        return results



# =========================
# ALGORITHM SIMULATOR
# =========================

def run_algorithm_simulation(starting_cash: float = 10000, days: int = 180) -> Dict:
    prices = {}
    score_rows = []

    for ticker, company in WATCHLIST.items():
        try:
            df = yf.download(ticker, period="90d", progress=False, auto_adjust=True)

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.tail(days + 20).copy()

            df["return_1d"] = df["Close"].pct_change()
            df["return_5d"] = df["Close"].pct_change(5)
            df["return_20d"] = df["Close"].pct_change(20)
            df["ma_20"] = df["Close"].rolling(20).mean()
            df["ma_50"] = df["Close"].rolling(50).mean()
            df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

            prices[ticker] = df

        except Exception:
            continue

    if not prices:
        return {
            "summary": {},
            "portfolio_history": pd.DataFrame(),
            "trades": pd.DataFrame(),
        }

    common_dates = sorted(set.intersection(*[set(df.index) for df in prices.values()]))
    common_dates = common_dates[-days:]

    cash = starting_cash
    positions = {}
    portfolio_history = []
    trades = []

    for current_date in common_dates:
        daily_scores = []

        for ticker, df in prices.items():
            if current_date not in df.index:
                continue

            row = df.loc[current_date]

            price = float(row["Close"])
            r5 = float(row.get("return_5d") or 0)
            r20 = float(row.get("return_20d") or 0)
            volume_ratio = float(row.get("volume_ratio") or 1)
            ma20 = float(row.get("ma_20") or 0)

            momentum_score = 50
            momentum_score += normalize_percent(r5, -0.10, 0.10) * 0.30 - 15
            momentum_score += normalize_percent(r20, -0.20, 0.20) * 0.40 - 20

            if ma20 and price > ma20:
                momentum_score += 10
            else:
                momentum_score -= 5

            volume_score = 50
            if volume_ratio > 1.8 and r5 > 0:
                volume_score = 75
            elif volume_ratio > 1.8 and r5 < 0:
                volume_score = 35
            elif volume_ratio > 1.2:
                volume_score = 60

            simulated_score = clamp(momentum_score * 0.70 + volume_score * 0.30)

            daily_scores.append({
                "ticker": ticker,
                "price": price,
                "score": simulated_score,
                "r5": r5,
                "r20": r20,
                "volume_ratio": volume_ratio,
            })

        daily_scores = sorted(daily_scores, key=lambda x: x["score"], reverse=True)

        # Sell weak positions
        for ticker in list(positions.keys()):
            matching = [x for x in daily_scores if x["ticker"] == ticker]
            if not matching:
                continue

            data = matching[0]

            if data["score"] < 45:
                shares = positions[ticker]["shares"]
                proceeds = shares * data["price"]
                cash += proceeds

                trades.append({
                    "Date": current_date.date(),
                    "Ticker": ticker,
                    "Action": "SELL",
                    "Price": round(data["price"], 2),
                    "Shares": round(shares, 4),
                    "Value": round(proceeds, 2),
                    "Score": round(data["score"], 1),
                })

                del positions[ticker]

        # Buy top names with score above 62
        candidates = [x for x in daily_scores if x["score"] >= 62 and x["ticker"] not in positions]
        max_positions = 5

        for data in candidates[:max_positions]:
            if len(positions) >= max_positions:
                break

            allocation = starting_cash * 0.20
            spend = min(cash, allocation)

            if spend < 100:
                continue

            shares = spend / data["price"]
            cash -= spend

            positions[data["ticker"]] = {
                "shares": shares,
                "avg_price": data["price"],
            }

            trades.append({
                "Date": current_date.date(),
                "Ticker": data["ticker"],
                "Action": "BUY",
                "Price": round(data["price"], 2),
                "Shares": round(shares, 4),
                "Value": round(spend, 2),
                "Score": round(data["score"], 1),
            })

        stock_value = 0

        for ticker, pos in positions.items():
            if ticker in prices and current_date in prices[ticker].index:
                stock_value += pos["shares"] * float(prices[ticker].loc[current_date]["Close"])

        total_value = cash + stock_value

        portfolio_history.append({
            "Date": current_date.date(),
            "Portfolio Value": round(total_value, 2),
            "Cash": round(cash, 2),
            "Invested": round(stock_value, 2),
            "Return %": round((total_value / starting_cash - 1) * 100, 2),
            "Open Positions": len(positions),
        })

    portfolio_df = pd.DataFrame(portfolio_history)
    trades_df = pd.DataFrame(trades)

    if portfolio_df.empty:
        summary = {}
    else:
        final_value = float(portfolio_df["Portfolio Value"].iloc[-1])
        peak = float(portfolio_df["Portfolio Value"].max())
        low = float(portfolio_df["Portfolio Value"].min())
        total_return = (final_value / starting_cash - 1) * 100

        summary = {
            "starting_cash": starting_cash,
            "final_value": round(final_value, 2),
            "profit_loss": round(final_value - starting_cash, 2),
            "return_percent": round(total_return, 2),
            "peak_value": round(peak, 2),
            "lowest_value": round(low, 2),
            "number_of_trades": len(trades_df),
            "open_positions": int(portfolio_df["Open Positions"].iloc[-1]),
        }

    explanations = []

    if not portfolio_df.empty:
        temp = portfolio_df.copy()
        temp["Daily Change"] = temp["Portfolio Value"].diff()
        temp["Daily Change %"] = temp["Portfolio Value"].pct_change() * 100
        major_moves = temp.dropna().copy()
        major_moves = major_moves.reindex(major_moves["Daily Change %"].abs().sort_values(ascending=False).index).head(8)

        for _, row in major_moves.iterrows():
            change = float(row["Daily Change"])
            change_pct = float(row["Daily Change %"])
            day = row["Date"]

            same_day_trades = trades_df[trades_df["Date"].astype(str) == str(day)] if not trades_df.empty else pd.DataFrame()

            if change > 0:
                reason = "Portfolio increased mainly because held positions moved higher."
            else:
                reason = "Portfolio decreased mainly because held positions moved lower or the algorithm reduced exposure."

            if not same_day_trades.empty:
                bought = same_day_trades[same_day_trades["Action"] == "BUY"]["Ticker"].tolist()
                sold = same_day_trades[same_day_trades["Action"] == "SELL"]["Ticker"].tolist()

                trade_notes = []
                if bought:
                    trade_notes.append("bought " + ", ".join(bought[:5]))
                if sold:
                    trade_notes.append("sold " + ", ".join(sold[:5]))

                if trade_notes:
                    reason += " The algorithm " + " and ".join(trade_notes) + "."

            if abs(change_pct) > 4:
                reason += " A move this large often happens during broad market volatility, earnings reactions, rate/inflation news, or sector-wide news."

            explanations.append({
                "Date": day,
                "Portfolio Change": round(change, 2),
                "Portfolio Change %": round(change_pct, 2),
                "Estimated Reason": reason,
            })

    explanations_df = pd.DataFrame(explanations)

    return {
        "summary": summary,
        "portfolio_history": portfolio_df,
        "trades": trades_df,
        "explanations": explanations_df,
    }


# =========================
# DASHBOARD
# =========================

def run_dashboard():
    import streamlit as st

    st.set_page_config(
        page_title="Ethan Yen AI Market Terminal",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .main-title {
        font-size: 40px;
        font-weight: 900;
        letter-spacing: -1.2px;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 15px;
        color: #9ca3af;
        margin-bottom: 22px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(125, 125, 125, 0.08);
        border: 1px solid rgba(125, 125, 125, 0.18);
        padding: 14px 16px;
        border-radius: 14px;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">Ethan Yen AI Market Terminal</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Live prices, AI scores, market momentum, news sentiment, charts, and algorithm paper-trading simulator</div>', unsafe_allow_html=True)

    db = Database()
    trader = PaperTrader(db)

    st.sidebar.title("Dashboard Controls")
st.sidebar.caption("Use refresh when you want the newest prices, news, and AI scores.")
    auto_refresh = True
    selected_period = st.sidebar.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    selected_chart_type = st.sidebar.selectbox("Chart type", ["Close price", "Volume", "Close + moving averages"], index=0)

    if st.sidebar.button("Refresh All Data"):
        with st.spinner("Updating prices, news, trends, and scores..."):
            bot = StockBot()
            bot.run_daily_update()
        st.success("Data refreshed.")
        st.rerun()

    existing_rows = []
    for ticker in WATCHLIST:
        if db.latest_market_row(ticker) and db.latest_score(ticker):
            existing_rows.append(ticker)

    if auto_refresh and len(existing_rows) == 0:
        with st.spinner("First launch: loading live prices, news, and scores. This may take a bit."):
            bot = StockBot()
            bot.run_daily_update()
        st.rerun()

    rows = []

    for ticker, company in WATCHLIST.items():
        score = db.latest_score(ticker)
        market = db.latest_market_row(ticker)

        if score and market:
            rows.append({
                "Ticker": ticker,
                "Company": company,
                "Price": float(market.get("close") or 0),
                "1D %": float(market.get("return_1d") or 0) * 100,
                "5D %": float(market.get("return_5d") or 0) * 100,
                "20D %": float(market.get("return_20d") or 0) * 100,
                "Volume": float(market.get("volume") or 0),
                "Vol Ratio": float(market.get("volume_ratio") or 0),
                "20D MA": float(market.get("ma_20") or 0),
                "RSI": float(market.get("rsi") or 0),
                "Score": float(score.get("final_score") or 0),
                "Signal": score.get("signal"),
                "News": float(score.get("news_score") or 0),
                "Trend": float(score.get("trend_score") or 0),
                "Momentum": float(score.get("momentum_score") or 0),
                "Volume Score": float(score.get("volume_score") or 0),
                "Macro Risk": float(score.get("macro_risk_score") or 0),
            })

    if not rows:
        st.warning("No data yet. Click Refresh All Data in the sidebar.")
        return

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)

    buy_count = int((df["Signal"] == "BUY").sum())
    watch_count = int((df["Signal"] == "WATCH").sum())
    sell_count = int(df["Signal"].isin(["SELL", "EMERGENCY_SELL"]).sum())
    avg_score = float(df["Score"].mean())

    top = df.iloc[0]
    bottom = df.iloc[-1]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Portfolio", f"${trader.portfolio_value():,.2f}")
    m2.metric("Cash", f"${db.get_cash():,.2f}")
    m3.metric("Stocks", len(df))
    m4.metric("Avg Score", f"{avg_score:.1f}")
    m5.metric("Top", f"{top['Ticker']}", f"{top['Score']:.1f}")
    m6.metric("Weakest", f"{bottom['Ticker']}", f"{bottom['Score']:.1f}")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Market Dashboard",
        "Stock Deep Dive",
        "Algorithm Simulator",
        "News & Explanations",
    ])

    with tab1:
        st.subheader("Market Signal Table")

        display_df = df.copy()
        display_df["Signal"] = display_df["Signal"].map({
            "BUY": "🟢 BUY",
            "WATCH": "🔵 WATCH",
            "HOLD": "🟡 HOLD",
            "SELL": "🔴 SELL",
            "EMERGENCY_SELL": "🔴 EMERGENCY SELL",
        }).fillna(display_df["Signal"])

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "1D %": st.column_config.NumberColumn("1D %", format="%.2f%%"),
                "5D %": st.column_config.NumberColumn("5D %", format="%.2f%%"),
                "20D %": st.column_config.NumberColumn("20D %", format="%.2f%%"),
                "Volume": st.column_config.NumberColumn("Volume", format="compact"),
                "Vol Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2fx"),
                "20D MA": st.column_config.NumberColumn("20D MA", format="$%.2f"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%.1f"),
            },
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("AI Score Ranking")
            st.bar_chart(df.set_index("Ticker")["Score"], width="stretch")

        with c2:
            st.subheader("Price Movement")
            st.line_chart(df.set_index("Ticker")[["1D %", "5D %", "20D %"]], width="stretch")

    with tab2:
        selected = st.selectbox("Choose a stock to inspect", df["Ticker"].tolist(), index=0)

        selected_score = db.latest_score(selected)
        selected_market = db.latest_market_row(selected)

        if selected_market and selected_score:
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Price", f"${float(selected_market.get('close') or 0):,.2f}")
            p2.metric("1D", f"{float(selected_market.get('return_1d') or 0) * 100:.2f}%")
            p3.metric("5D", f"{float(selected_market.get('return_5d') or 0) * 100:.2f}%")
            p4.metric("20D", f"{float(selected_market.get('return_20d') or 0) * 100:.2f}%")
            p5.metric("AI Score", f"{float(selected_score.get('final_score') or 0):.1f}")

        chart_data = yf.download(selected, period=selected_period, progress=False, auto_adjust=True)

        if not chart_data.empty:
            if isinstance(chart_data.columns, pd.MultiIndex):
                chart_data.columns = chart_data.columns.get_level_values(0)

            if selected_chart_type == "Close price":
                st.line_chart(chart_data["Close"], width="stretch")
            elif selected_chart_type == "Volume":
                st.bar_chart(chart_data["Volume"], width="stretch")
            else:
                chart_data["MA20"] = chart_data["Close"].rolling(20).mean()
                chart_data["MA50"] = chart_data["Close"].rolling(50).mean()
                st.line_chart(chart_data[["Close", "MA20", "MA50"]], width="stretch")

        st.subheader("Score Breakdown")

        if selected_score:
            breakdown = pd.DataFrame({
                "Component": ["News", "Trend", "Momentum", "Volume", "Macro Risk", "Earnings Risk", "Confidence"],
                "Score": [
                    selected_score.get("news_score"),
                    selected_score.get("trend_score"),
                    selected_score.get("momentum_score"),
                    selected_score.get("volume_score"),
                    selected_score.get("macro_risk_score"),
                    selected_score.get("earnings_risk_score"),
                    selected_score.get("confidence_score"),
                ],
            }).set_index("Component")

            st.bar_chart(breakdown, width="stretch")

    with tab3:
        st.subheader("Algorithm Trading Simulator")
        st.caption("Simulates how the algorithm would have traded over the selected period using historical price momentum and volume. News explanations are estimated from market movement, not true historical news archives.")

        sim_cash = st.number_input("Starting cash", min_value=1000, max_value=1000000, value=10000, step=1000)
        sim_days = st.slider("Simulation period", min_value=30, max_value=180, value=180)

        if st.button("Run Algorithm Simulation"):
            with st.spinner("Running historical paper-trading simulation..."):
                result = run_algorithm_simulation(starting_cash=sim_cash, days=sim_days)

            summary = result["summary"]
            history = result["portfolio_history"]
            trades = result["trades"]

            if not summary or history.empty:
                st.error("Simulation failed because price data was unavailable.")
            else:
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Start", f"${summary['starting_cash']:,.2f}")
                s2.metric("Final Value", f"${summary['final_value']:,.2f}")
                s3.metric("Profit / Loss", f"${summary['profit_loss']:,.2f}", f"{summary['return_percent']:.2f}%")
                s4.metric("Peak Value", f"${summary['peak_value']:,.2f}")
                s5.metric("Trades", summary["number_of_trades"])

                st.subheader("Portfolio Value Trend")
                st.line_chart(history.set_index("Date")[["Portfolio Value", "Cash", "Invested"]], width="stretch")

                st.subheader("Portfolio Return %")
                st.line_chart(history.set_index("Date")["Return %"], width="stretch")

                st.subheader("Major Portfolio Moves")
                explanations = result.get("explanations", pd.DataFrame())

                if explanations.empty:
                    st.info("No major portfolio moves detected.")
                else:
                    st.dataframe(explanations, width="stretch", hide_index=True)

                st.subheader("Daily Portfolio History")
                st.dataframe(history, width="stretch", hide_index=True)

                st.subheader("Simulated Trades")
                if trades.empty:
                    st.info("No trades were triggered during the simulation.")
                else:
                    st.dataframe(trades, width="stretch", hide_index=True)

    with tab4:
        st.subheader("Recent Filtered Headlines")
        selected_news_ticker = st.selectbox("Choose stock for news", df["Ticker"].tolist(), key="news_select")
        news = db.recent_news(selected_news_ticker)

        if news:
            for item in news[:10]:
                title = item.get("title")
                url = item.get("url")
                source = item.get("source")
                sentiment = item.get("sentiment")

                if url:
                    st.markdown(f"**[{title}]({url})**")
                else:
                    st.markdown(f"**{title}**")

                st.caption(f"Source: {source} | Sentiment: {sentiment}")
        else:
            st.info("No recent filtered headlines for this stock.")

        st.subheader("All Signal Explanations")

        for ticker in df["Ticker"]:
            score = db.latest_score(ticker)
            market = db.latest_market_row(ticker)

            if not score or not market:
                continue

            with st.expander(f"{ticker} | ${float(market.get('close') or 0):,.2f} | Score {float(score.get('final_score') or 0):.1f} | {score.get('signal')}"):
                for reason in score.get("reasons", []):
                    st.write(f"- {reason}")


# =========================
# UTILITY COMMANDS
# =========================

def run_backtest():
    backtester = Backtester()
    results = []

    for ticker in WATCHLIST:
        result = backtester.simple_momentum_backtest(ticker)
        results.append(result)

    df = pd.DataFrame(results)
    print(df[[
        "ticker",
        "strategy_final_value",
        "strategy_return_percent",
        "buy_hold_final_value",
        "buy_hold_return_percent",
        "number_of_trades",
    ]])


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    import sys

    if "streamlit" in sys.modules:
        run_dashboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_backtest()
    else:
        bot = StockBot()
        bot.run_daily_update()