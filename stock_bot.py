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
    "META": "Meta",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "PLTR": "Palantir",
    "COIN": "Coinbase",
    "JPM": "JPMorgan",
    "XOM": "Exxon Mobil",
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

        for article in articles:
            article["sentiment"] = self.sentiment.score(article["title"])
            article["surprise_type"] = detect_surprise_type(article["title"])
            self.db.save_news(article)

        return articles

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
        query = f"{company_name}+{ticker}+stock"
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
# DASHBOARD
# =========================

def run_dashboard():
    import streamlit as st

    db = Database()

    st.set_page_config(page_title="AI Stock Signal Bot", layout="wide")

    st.title("AI Stock News + Trend Scoring Bot")
    st.caption("Research and paper-trading only. No real trades.")

    rows = []

    for ticker, company in WATCHLIST.items():
        score = db.latest_score(ticker)
        market = db.latest_market_row(ticker)

        if score and market:
            rows.append({
                "Ticker": ticker,
                "Company": company,
                "Price": market.get("close"),
                "Final Score": score.get("final_score"),
                "Signal": score.get("signal"),
                "News": score.get("news_score"),
                "Trend": score.get("trend_score"),
                "Momentum": score.get("momentum_score"),
                "Volume": score.get("volume_score"),
                "Macro Risk": score.get("macro_risk_score"),
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No scores yet. Run: python stock_bot.py")

    st.subheader("Portfolio")
    trader = PaperTrader(db)
    st.metric("Paper Portfolio Value", f"${trader.portfolio_value():,.2f}")
    st.metric("Cash", f"${db.get_cash():,.2f}")

    st.subheader("Latest Explanations")

    for ticker in WATCHLIST:
        score = db.latest_score(ticker)

        if not score:
            continue

        with st.expander(f"{ticker} | Score {score['final_score']} | {score['signal']}"):
            for reason in score.get("reasons", []):
                st.write(f"- {reason}")

            news = db.recent_news(ticker)

            if news:
                st.write("Recent headlines:")
                for item in news[:5]:
                    st.write(f"- {item.get('title')}")


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