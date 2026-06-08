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
from pathlib import Path
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
# ENV / SECRET HELPERS
# =========================

def load_local_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


def secret_value(name: str, default: str = "") -> str:
    """Read from environment variables first, then Streamlit secrets if available."""
    value = os.getenv(name)
    if value:
        return str(value)

    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default


load_local_env_file()


# =========================
# CONFIG
# =========================

WATCHLIST = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMD": "Advanced Micro Devices",
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
    "BRK-B": "Berkshire Hathaway",
    "TSM": "Taiwan Semiconductor",
    "ASML": "ASML Holding",
    "AMAT": "Applied Materials",
    "MU": "Micron Technology",
    "ARM": "Arm Holdings",
    "QCOM": "Qualcomm",
    "TXN": "Texas Instruments",
    "INTU": "Intuit",
    "NOW": "ServiceNow",
    "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike",
    "VRT": "Vertiv",
    "MRVL": "Marvell Technology",
    "VEEV": "Veeva Systems",
    "BKNG": "Booking Holdings",
    "ABNB": "Airbnb",
    "DIS": "Disney",
    "PEP": "PepsiCo",
    "MCD": "McDonald's",
    "CAT": "Caterpillar",
    "GE": "GE Aerospace",
    "RTX": "RTX",
    "NKE": "Nike",
    "PFE": "Pfizer",
    "ABBV": "AbbVie",
    "ISRG": "Intuitive Surgical",
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


COMPANY_DESCRIPTIONS = {
    "AAPL": "Apple makes iPhones, Macs, wearables, services, and consumer software. It is a mega-cap technology company driven by hardware cycles, services revenue, AI features, and consumer demand.",
    "MSFT": "Microsoft sells cloud infrastructure, Windows, Office, enterprise software, gaming, and AI tools. Its stock is heavily influenced by Azure growth, enterprise spending, and AI adoption.",
    "NVDA": "Nvidia designs GPUs and AI accelerators used in data centers, gaming, and professional computing. Its stock is closely tied to AI infrastructure demand and chip supply.",
    "AMD": "AMD designs CPUs, GPUs, and data center chips. It competes with Nvidia and Intel and is influenced by AI chip demand, PC cycles, and server growth.",
    "TSLA": "Tesla makes electric vehicles, batteries, energy storage products, and autonomous driving software. Its stock moves on deliveries, margins, robotaxi expectations, and EV demand.",
    "META": "Meta owns Facebook, Instagram, WhatsApp, Threads, and major AI/reality-labs projects. Its stock is driven by ad growth, user engagement, AI spending, and margins.",
    "GOOGL": "Alphabet owns Google Search, YouTube, Android, Google Cloud, and AI products. Its stock is driven by ad demand, cloud growth, AI competition, and regulation.",
    "AMZN": "Amazon runs e-commerce, logistics, advertising, Prime, and AWS cloud. Its stock is driven by AWS growth, retail margins, ads, and consumer spending.",
    "PLTR": "Palantir builds data analytics and AI platforms for governments and enterprises. Its stock is driven by AI software demand, contract growth, and valuation expectations.",
    "COIN": "Coinbase is a crypto exchange and infrastructure company. Its stock is heavily influenced by Bitcoin prices, crypto trading volume, regulation, and institutional adoption.",
    "JPM": "JPMorgan Chase is one of the largest banks in the world. Its stock is driven by interest rates, credit risk, loan growth, investment banking, and the broader economy.",
    "XOM": "Exxon Mobil is a global energy company focused on oil, natural gas, refining, and chemicals. Its stock is driven by oil prices, energy demand, and geopolitical risk.",
    "AVGO": "Broadcom designs semiconductors and infrastructure software. Its stock is driven by AI networking chips, custom silicon, cloud demand, and software integration.",
    "NFLX": "Netflix is a global streaming entertainment company. Its stock is driven by subscriber growth, ad-tier adoption, pricing power, and content spending.",
    "CRM": "Salesforce sells cloud software for sales, service, marketing, and data. Its stock is driven by enterprise software demand, AI tools, margins, and subscription growth.",
    "ORCL": "Oracle sells enterprise software, databases, and cloud infrastructure. Its stock is driven by cloud growth, AI data-center demand, and enterprise contracts.",
    "ADBE": "Adobe sells creative, document, and marketing software. Its stock is driven by subscription growth, AI creative tools, and competition.",
    "UBER": "Uber operates ride-sharing, delivery, freight, and mobility platforms. Its stock is driven by bookings growth, margins, autonomous vehicle risk, and consumer demand.",
    "SHOP": "Shopify provides e-commerce software and merchant tools. Its stock is driven by online retail growth, merchant services, payments, and operating leverage.",
    "LLY": "Eli Lilly is a pharmaceutical company known for diabetes, obesity, and other medicines. Its stock is driven by drug demand, trial data, approvals, and pricing.",
    "UNH": "UnitedHealth is a large health insurance and healthcare services company. Its stock is influenced by medical costs, regulation, enrollment, and healthcare margins.",
    "V": "Visa runs one of the largest global payment networks. Its stock is driven by consumer spending, cross-border transactions, and payment volume.",
    "MA": "Mastercard operates a global payments network. Its stock is driven by consumer spending, international travel, and payment volume growth.",
    "COST": "Costco runs membership-based warehouse stores. Its stock is driven by same-store sales, membership renewals, margins, and consumer strength.",
    "WMT": "Walmart is a global retail giant. Its stock is driven by grocery demand, e-commerce growth, margins, and consumer spending.",
    "HD": "Home Depot is a major home improvement retailer. Its stock is driven by housing activity, renovation demand, interest rates, and consumer confidence.",
    "BRK-B": "Berkshire Hathaway is a diversified holding company led by Warren Buffett's investment model. It owns insurance, railroads, utilities, industrial businesses, and a large stock portfolio.",
    "TSM": "Taiwan Semiconductor Manufacturing is the world's leading chip foundry. It manufactures advanced chips for companies like Nvidia, Apple, AMD, and others.",
    "ASML": "ASML makes extreme ultraviolet lithography machines used to manufacture advanced semiconductors. It is critical to the global chip supply chain.",
    "AMAT": "Applied Materials sells equipment and services used in semiconductor manufacturing. Its stock is tied to chip factory spending and AI hardware demand.",
    "MU": "Micron makes memory and storage chips. Its stock is cyclical and driven by DRAM/NAND pricing, AI memory demand, and data center spending.",
    "ARM": "Arm designs chip architecture used in mobile devices, data centers, and AI hardware. Its stock is influenced by licensing growth and semiconductor demand.",
    "QCOM": "Qualcomm designs mobile and wireless chips. Its stock is driven by smartphone cycles, automotive chips, AI devices, and licensing revenue.",
    "TXN": "Texas Instruments makes analog and embedded chips used across industrial, automotive, and electronics markets.",
    "INTU": "Intuit makes TurboTax, QuickBooks, Credit Karma, and financial software. Its stock is driven by small business activity and tax/software demand.",
    "NOW": "ServiceNow provides workflow automation software for enterprises. Its stock is driven by cloud software demand, AI workflow tools, and subscription growth.",
    "PANW": "Palo Alto Networks is a cybersecurity company. Its stock is driven by enterprise security spending, platform adoption, and cyber risk.",
    "CRWD": "CrowdStrike provides cloud-based cybersecurity software. Its stock is driven by endpoint security demand, retention, and enterprise growth.",
    "VRT": "Vertiv provides power and cooling infrastructure for data centers. Its stock is closely linked to AI data center construction and electrical infrastructure demand.",
    "MRVL": "Marvell designs chips for data centers, networking, storage, and custom AI silicon. Its stock is driven by AI infrastructure demand and cloud chip projects.",
    "VEEV": "Veeva provides cloud software for life sciences companies. Its stock is driven by pharmaceutical software spending and regulated industry demand.",
    "BKNG": "Booking Holdings runs travel platforms like Booking.com, Priceline, Agoda, and OpenTable. Its stock is driven by global travel demand.",
    "ABNB": "Airbnb operates a marketplace for short-term stays and experiences. Its stock is driven by travel demand, supply growth, regulation, and bookings.",
    "DIS": "Disney owns film studios, theme parks, ESPN, streaming, and consumer brands. Its stock is driven by parks, streaming profitability, sports, and content.",
    "PEP": "PepsiCo sells beverages and snacks globally. Its stock is defensive and driven by pricing power, volumes, and consumer staples demand.",
    "MCD": "McDonald's is a global fast-food franchise company. Its stock is driven by same-store sales, pricing power, margins, and global consumer demand.",
    "CAT": "Caterpillar makes construction, mining, and industrial equipment. Its stock is cyclical and driven by infrastructure, commodities, and global growth.",
    "GE": "GE Aerospace makes jet engines and aerospace systems. Its stock is driven by aviation demand, engine services, and defense/commercial aircraft cycles.",
    "RTX": "RTX is an aerospace and defense company. Its stock is driven by defense spending, aircraft systems, missiles, and engine demand.",
    "NKE": "Nike sells athletic footwear, apparel, and equipment. Its stock is driven by consumer demand, brand strength, margins, and China/US sales.",
    "PFE": "Pfizer is a pharmaceutical company. Its stock is driven by drug pipeline progress, approvals, vaccine demand, and patent cycles.",
    "ABBV": "AbbVie is a pharmaceutical company focused on immunology, oncology, aesthetics, and neuroscience. Its stock is driven by drug sales and pipeline execution.",
    "ISRG": "Intuitive Surgical makes robotic surgery systems. Its stock is driven by procedure growth, system placements, and healthcare technology adoption.",
    "SPY": "SPY tracks the S&P 500 index and is used as a broad-market benchmark.",
    "QQQ": "QQQ tracks the Nasdaq-100 index and is used as a growth/technology-heavy benchmark.",
}

MACRO_RSS_FEEDS = {
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "SEC": "https://www.sec.gov/news/pressreleases.rss",
    "Treasury": "https://home.treasury.gov/news/press-releases/feed",
    "White House": "https://www.whitehouse.gov/feed/",
    "BLS": "https://www.bls.gov/feed/news_release.rss",
    "BEA": "https://www.bea.gov/news/current-releases/rss.xml",
}

MACRO_POSITIVE_TERMS = [
    "rate cut", "cuts rates", "cooling inflation", "inflation eased",
    "strong jobs", "gdp growth", "soft landing", "lower yields",
    "consumer confidence rises", "productivity rises"
]

MACRO_NEGATIVE_TERMS = [
    "rate hike", "higher rates", "inflation rose", "hot inflation",
    "recession", "unemployment rises", "jobless claims rise",
    "bank stress", "default", "shutdown", "tariff", "sanctions",
    "geopolitical risk", "war", "oil spike", "trade war",
    "export controls", "antitrust", "government investigation"
]

MACRO_GOOGLE_NEWS_QUERIES = [
    "Trump tariffs stock market",
    "Trump trade policy markets",
    "White House tariffs stocks",
    "Federal Reserve interest rates stock market",
    "inflation CPI stock market",
    "jobs report stock market",
    "GDP growth stock market",
    "Treasury yields tech stocks",
    "new IPO stock market",
    "Nasdaq IPO upcoming",
    "SEC investigation public company",
    "geopolitical risk stock market",
    "oil prices stock market",
    "AI regulation stock market",
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
        CREATE TABLE IF NOT EXISTS macro_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            source TEXT,
            url TEXT UNIQUE,
            published_at TEXT,
            sentiment REAL,
            impact_score REAL,
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

    def save_macro_news(self, item: Dict):
        cur = self.conn.cursor()

        try:
            cur.execute("""
            INSERT OR IGNORE INTO macro_news
            (title, source, url, published_at, sentiment, impact_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                item["title"],
                item["source"],
                item["url"],
                item["published_at"],
                item.get("sentiment"),
                item.get("impact_score"),
                now_date(),
            ))
            self.conn.commit()
        except Exception:
            pass

    def recent_macro_news(self, days: int = 5) -> List[Dict]:
        cur = self.conn.cursor()
        cutoff = dt.datetime.now() - dt.timedelta(days=days)

        cur.execute("""
        SELECT title, source, url, published_at, sentiment, impact_score
        FROM macro_news
        WHERE published_at >= ?
        ORDER BY published_at DESC
        LIMIT 50
        """, (cutoff.isoformat(),))

        rows = cur.fetchall()
        cols = [x[0] for x in cur.description]
        return [dict(zip(cols, row)) for row in rows]

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
# MACRO / GOVERNMENT NEWS
# =========================

class MacroNewsCollector:
    def __init__(self, db: Database):
        self.db = db
        self.sentiment = SentimentEngine()

    def fetch_all(self) -> List[Dict]:
        all_items = []

        for source, url in MACRO_RSS_FEEDS.items():
            all_items.extend(self.fetch_rss(source, url))

        all_items.extend(self.fetch_google_macro_searches())
        all_items.extend(self.fetch_gdelt_macro_news())

        saved_items = []

        for item in all_items:
            title = item.get("title", "")
            item["sentiment"] = self.sentiment.score(title)
            item["impact_score"] = self.score_macro_impact(title)
            self.db.save_macro_news(item)
            saved_items.append(item)

        return saved_items

    def fetch_google_macro_searches(self) -> List[Dict]:
        items = []

        for query in MACRO_GOOGLE_NEWS_QUERIES:
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            items.extend(self.fetch_rss(f"Google News: {query}", url))

        return items

    def fetch_gdelt_macro_news(self) -> List[Dict]:
        query = "(stock market OR Nasdaq OR S&P 500 OR inflation OR Federal Reserve OR tariffs OR IPO OR Treasury yields OR oil prices)"
        url = "https://api.gdeltproject.org/api/v2/doc/doc"

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 50,
            "sort": "HybridRel",
        }

        try:
            response = requests.get(url, params=params, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            data = response.json()
        except Exception:
            return []

        items = []

        for article in data.get("articles", []):
            title = article.get("title") or ""
            link = article.get("url") or ""

            if not title:
                continue

            items.append({
                "title": title,
                "source": "GDELT Global News",
                "url": link or f"gdelt:{hash(title)}",
                "published_at": now_date(),
            })

        return items

    def fetch_rss(self, source: str, url: str) -> List[Dict]:
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "EthanYenStockBot/1.0"}
            )
            text = response.text
        except Exception:
            return []

        items = []
        parts = text.split("<item>")[1:30]

        for part in parts:
            title = clean_html(extract_between(part, "<title>", "</title>"))
            link = extract_between(part, "<link>", "</link>")
            pub_date = extract_between(part, "<pubDate>", "</pubDate>")

            if not title:
                continue

            items.append({
                "title": title,
                "source": source,
                "url": link or f"macro:{source}:{hash(title)}",
                "published_at": parse_rss_date(pub_date),
            })

        return items

    def score_macro_impact(self, title: str) -> float:
        t = title.lower()
        score = 50

        for term in MACRO_POSITIVE_TERMS:
            if term in t:
                score += 12

        for term in MACRO_NEGATIVE_TERMS:
            if term in t:
                score -= 14

        return clamp(score)


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

        articles.extend(self.fetch_yahoo_finance_news(ticker))
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

    def fetch_yahoo_finance_news(self, ticker: str) -> List[Dict]:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            text = response.text
        except Exception:
            return []

        articles = []
        parts = text.split("<item>")[1:20]

        for part in parts:
            title = clean_html(extract_between(part, "<title>", "</title>"))
            link = extract_between(part, "<link>", "</link>")
            pub_date = extract_between(part, "<pubDate>", "</pubDate>")

            if title:
                articles.append({
                    "ticker": ticker,
                    "title": title,
                    "source": "Yahoo Finance RSS",
                    "url": link or f"yahoo:{ticker}:{hash(title)}",
                    "published_at": parse_rss_date(pub_date),
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
        macro_news = self.db.recent_macro_news(days=7)

        risk_terms = [
            "inflation", "rate hike", "higher rates", "recession",
            "tariff", "war", "regulation", "lawsuit", "probe",
            "federal reserve", "cpi", "unemployment", "shutdown",
            "sanctions", "oil", "treasury yield"
        ]

        company_titles = " ".join([x.get("title", "").lower() for x in news])
        macro_titles = " ".join([x.get("title", "").lower() for x in macro_news])

        joined = company_titles + " " + macro_titles
        risk_hits = sum(1 for term in risk_terms if term in joined)

        if macro_news:
            macro_score = float(np.mean([x.get("impact_score") or 50 for x in macro_news]))
        else:
            macro_score = 50

        if risk_hits >= 6:
            macro_score -= 20
        elif risk_hits >= 3:
            macro_score -= 10
        elif risk_hits >= 1:
            macro_score -= 5

        return clamp(macro_score)

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
        self.macro = MacroNewsCollector(self.db)
        self.scorer = StockScorer(self.db)
        self.trader = PaperTrader(self.db)

    def run_daily_update(self):
        results = []

        try:
            macro_items = self.macro.fetch_all()
            print(f"Macro/government news items found: {len(macro_items)}")
        except Exception as e:
            print(f"Macro news failed: {e}")

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
    market_context = {}

    # Download enough data for 180 trading days plus indicator warmup.
    download_period = "1y"

    context_symbols = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "^VIX": "VIX volatility index",
        "^TNX": "10Y treasury yield",
        "CL=F": "Oil futures",
        "BTC-USD": "Bitcoin",
    }

    all_symbols = dict(WATCHLIST)
    all_symbols.update(context_symbols)

    for ticker, company in all_symbols.items():
        try:
            df = yf.download(ticker, period=download_period, progress=False, auto_adjust=True)

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.copy()
            df["return_1d"] = df["Close"].pct_change()
            df["return_5d"] = df["Close"].pct_change(5)
            df["return_20d"] = df["Close"].pct_change(20)
            df["ma_20"] = df["Close"].rolling(20).mean()
            df["ma_50"] = df["Close"].rolling(50).mean()
            df["ma_200"] = df["Close"].rolling(200).mean()
            df["volatility_20d"] = df["return_1d"].rolling(20).std()
            df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

            if ticker in WATCHLIST:
                prices[ticker] = df
            else:
                market_context[ticker] = df

        except Exception:
            continue

    if not prices:
        return {
            "summary": {},
            "portfolio_history": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "explanations": pd.DataFrame(),
            "benchmark": pd.DataFrame(),
        }

    common_dates = sorted(set.intersection(*[set(df.index) for df in prices.values()]))
    common_dates = common_dates[-days:]

    cash = starting_cash
    positions = {}
    portfolio_history = []
    trades = []
    max_positions = 5
    max_position_fraction = 0.20
    stop_loss = -0.08
    trailing_stop = -0.10

    def get_market_regime(current_date):
        spy = market_context.get("SPY")
        qqq = market_context.get("QQQ")
        vix = market_context.get("^VIX")
        tnx = market_context.get("^TNX")

        score = 50
        notes = []

        if spy is not None and current_date in spy.index:
            row = spy.loc[current_date]
            if row["Close"] > row["ma_50"]:
                score += 15
                notes.append("SPY traded above its 50-day moving average.")
            else:
                score -= 15
                notes.append("SPY traded below its 50-day moving average.")

            if float(row.get("return_5d") or 0) < -0.03:
                score -= 10
                notes.append("SPY had weak 5-day momentum.")

        if qqq is not None and current_date in qqq.index:
            row = qqq.loc[current_date]
            if row["Close"] > row["ma_50"]:
                score += 10
                notes.append("QQQ traded above its 50-day moving average.")
            else:
                score -= 10
                notes.append("QQQ traded below its 50-day moving average.")

            if float(row.get("return_5d") or 0) < -0.04:
                score -= 10
                notes.append("QQQ had weak 5-day momentum.")

        if vix is not None and current_date in vix.index:
            vix_5d = float(vix.loc[current_date].get("return_5d") or 0)
            if vix_5d > 0.15:
                score -= 15
                notes.append("VIX rose sharply, suggesting risk-off market pressure.")

        if tnx is not None and current_date in tnx.index:
            rate_5d = float(tnx.loc[current_date].get("return_5d") or 0)
            if rate_5d > 0.08:
                score -= 8
                notes.append("10-year Treasury yield rose, which can pressure growth stocks.")

        return clamp(score), notes

    for current_date in common_dates:
        regime_score, regime_notes = get_market_regime(current_date)
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
            ma50 = float(row.get("ma_50") or 0)
            vol = float(row.get("volatility_20d") or 0)

            momentum_score = 50
            momentum_score += normalize_percent(r5, -0.10, 0.10) * 0.25 - 12.5
            momentum_score += normalize_percent(r20, -0.20, 0.20) * 0.35 - 17.5

            if ma20 and price > ma20:
                momentum_score += 8
            else:
                momentum_score -= 6

            if ma50 and price > ma50:
                momentum_score += 8
            else:
                momentum_score -= 8

            volume_score = 50
            if volume_ratio > 1.8 and r5 > 0:
                volume_score = 75
            elif volume_ratio > 1.8 and r5 < 0:
                volume_score = 30
            elif volume_ratio > 1.2:
                volume_score = 60

            volatility_penalty = 0
            if vol > 0.045:
                volatility_penalty = 10
            elif vol > 0.035:
                volatility_penalty = 5

            simulated_score = clamp(
                momentum_score * 0.55 +
                volume_score * 0.20 +
                regime_score * 0.25 -
                volatility_penalty
            )

            daily_scores.append({
                "ticker": ticker,
                "price": price,
                "score": simulated_score,
                "r5": r5,
                "r20": r20,
                "volume_ratio": volume_ratio,
                "volatility": vol,
            })

        daily_scores = sorted(daily_scores, key=lambda x: x["score"], reverse=True)

        # Sell rules: score breakdown, stop loss, trailing stop, and weak market regime.
        for ticker in list(positions.keys()):
            matching = [x for x in daily_scores if x["ticker"] == ticker]
            if not matching:
                continue

            data = matching[0]
            pos = positions[ticker]
            current_return = (data["price"] / pos["avg_price"]) - 1
            pos["highest_price"] = max(pos.get("highest_price", pos["avg_price"]), data["price"])
            drawdown_from_high = (data["price"] / pos["highest_price"]) - 1

            sell_reason = None

            if data["score"] < 48:
                sell_reason = "algorithm score dropped below risk threshold"
            elif current_return <= stop_loss:
                sell_reason = "stop loss triggered to protect capital"
            elif drawdown_from_high <= trailing_stop:
                sell_reason = "trailing stop triggered to lock gains or limit downside"
            elif current_return >= 0.18:
                sell_reason = "take-profit rule triggered after strong gain"
            elif regime_score < 40 and data["score"] < 58:
                sell_reason = "market regime turned defensive"

            if sell_reason:
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
                    "Reason": sell_reason,
                })

                del positions[ticker]

        # Buy rules: only buy in neutral or positive market regime.
        if regime_score >= 50:
            candidates = [
                x for x in daily_scores
                if x["score"] >= 68
                and x["ticker"] not in positions
                and x["volatility"] < 0.05
                and x["r5"] > -0.04
                and x["r20"] > -0.08
            ]

            for data in candidates[:max_positions]:
                if len(positions) >= max_positions:
                    break

                # Smaller allocation in weaker markets.
                regime_multiplier = 1.0 if regime_score >= 65 else 0.50
                spend = min(cash, starting_cash * max_position_fraction * regime_multiplier)

                if spend < 100:
                    continue

                shares = spend / data["price"]
                cash -= spend

                positions[data["ticker"]] = {
                    "shares": shares,
                    "avg_price": data["price"],
                    "highest_price": data["price"],
                }

                trades.append({
                    "Date": current_date.date(),
                    "Ticker": data["ticker"],
                    "Action": "BUY",
                    "Price": round(data["price"], 2),
                    "Shares": round(shares, 4),
                    "Value": round(spend, 2),
                    "Score": round(data["score"], 1),
                    "Reason": "strong score with acceptable market regime",
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
            "Market Regime Score": round(regime_score, 1),
            "Market Notes": " ".join(regime_notes[:3]),
        })

    portfolio_df = pd.DataFrame(portfolio_history)
    trades_df = pd.DataFrame(trades)

    benchmark_rows = []
    for bench in ["SPY", "QQQ"]:
        dfb = market_context.get(bench)
        if dfb is not None and common_dates[0] in dfb.index and common_dates[-1] in dfb.index:
            start_price = float(dfb.loc[common_dates[0]]["Close"])
            end_price = float(dfb.loc[common_dates[-1]]["Close"])
            final_value = starting_cash * (end_price / start_price)
            benchmark_rows.append({
                "Benchmark": bench,
                "Final Value": round(final_value, 2),
                "Return %": round((final_value / starting_cash - 1) * 100, 2),
            })

    benchmark_df = pd.DataFrame(benchmark_rows)

    if portfolio_df.empty:
        summary = {}
    else:
        final_value = float(portfolio_df["Portfolio Value"].iloc[-1])
        peak = float(portfolio_df["Portfolio Value"].max())
        low = float(portfolio_df["Portfolio Value"].min())
        total_return = (final_value / starting_cash - 1) * 100
        max_drawdown = ((low / peak) - 1) * 100 if peak else 0

        summary = {
            "starting_cash": starting_cash,
            "final_value": round(final_value, 2),
            "profit_loss": round(final_value - starting_cash, 2),
            "return_percent": round(total_return, 2),
            "peak_value": round(peak, 2),
            "lowest_value": round(low, 2),
            "max_drawdown_percent": round(max_drawdown, 2),
            "number_of_trades": len(trades_df),
            "open_positions": int(portfolio_df["Open Positions"].iloc[-1]),
        }

    explanations = []

    if not portfolio_df.empty:
        temp = portfolio_df.copy()
        temp["Daily Change"] = temp["Portfolio Value"].diff()
        temp["Daily Change %"] = temp["Portfolio Value"].pct_change() * 100
        major_moves = temp.dropna().copy()
        major_moves = major_moves.reindex(major_moves["Daily Change %"].abs().sort_values(ascending=False).index).head(10)

        for _, row in major_moves.iterrows():
            change = float(row["Daily Change"])
            change_pct = float(row["Daily Change %"])
            day = row["Date"]
            notes = row.get("Market Notes", "")

            same_day_trades = trades_df[trades_df["Date"].astype(str) == str(day)] if not trades_df.empty else pd.DataFrame()

            context_parts = []

            for symbol, label in {
                "SPY": "S&P 500",
                "QQQ": "Nasdaq 100",
                "^VIX": "VIX",
                "^TNX": "10Y yield",
                "CL=F": "Oil",
                "BTC-USD": "Bitcoin",
            }.items():
                ctx = market_context.get(symbol)
                if ctx is not None:
                    match_dates = [d for d in ctx.index if str(d.date()) == str(day)]
                    if match_dates:
                        d = match_dates[0]
                        r1 = float(ctx.loc[d].get("return_1d") or 0) * 100
                        if abs(r1) >= 1.0 or symbol in ["^VIX", "^TNX"]:
                            context_parts.append(f"{label} moved {r1:.2f}%")

            if change > 0:
                reason = "Portfolio gained as the algorithm's held stocks benefited from the day's market setup."
            else:
                reason = "Portfolio fell as the market setup turned weaker for the algorithm's held stocks."

            if context_parts:
                reason += " Market context: " + "; ".join(context_parts[:4]) + "."

            if notes:
                reason += " Regime notes: " + notes

            if not same_day_trades.empty:
                bought = same_day_trades[same_day_trades["Action"] == "BUY"]["Ticker"].tolist()
                sold = same_day_trades[same_day_trades["Action"] == "SELL"]["Ticker"].tolist()

                if bought:
                    reason += " The algorithm added exposure to " + ", ".join(bought[:5]) + "."
                if sold:
                    reason += " The algorithm reduced risk by selling " + ", ".join(sold[:5]) + "."

            reason += " This is market-data based reasoning, not a verified archived news headline."

            explanations.append({
                "Date": day,
                "Portfolio Change": round(change, 2),
                "Portfolio Change %": round(change_pct, 2),
                "Market Regime Score": row.get("Market Regime Score"),
                "Estimated Market Reason": reason,
            })

    explanations_df = pd.DataFrame(explanations)

    return {
        "summary": summary,
        "portfolio_history": portfolio_df,
        "trades": trades_df,
        "explanations": explanations_df,
        "benchmark": benchmark_df,
    }




# =========================
# RISK-FIRST ALGORITHM HELPERS
# =========================

def calculate_risk_adjusted_score(score: Dict, market: Dict) -> float:
    base = float(score.get("final_score") or 50)
    momentum = float(score.get("momentum_score") or 50)
    macro = float(score.get("macro_risk_score") or 50)
    volume_score = float(score.get("volume_score") or 50)

    r1 = float(market.get("return_1d") or 0)
    r5 = float(market.get("return_5d") or 0)
    r20 = float(market.get("return_20d") or 0)
    close = float(market.get("close") or 0)
    ma20 = float(market.get("ma_20") or 0)
    ma50 = float(market.get("ma_50") or 0)
    rsi = float(market.get("rsi") or 50)
    vol_ratio = float(market.get("volume_ratio") or 1)

    adjusted = base

    # Reward healthy momentum, not chaotic spikes.
    if close and ma20 and close > ma20:
        adjusted += 4
    else:
        adjusted -= 8

    if close and ma50 and close > ma50:
        adjusted += 6
    else:
        adjusted -= 10

    # Avoid catching falling knives.
    if r5 < -0.06:
        adjusted -= 12
    elif r5 < -0.03:
        adjusted -= 6

    if r20 < -0.12:
        adjusted -= 15
    elif r20 < -0.07:
        adjusted -= 8

    # Avoid extreme overbought entries.
    if rsi > 80:
        adjusted -= 12
    elif rsi > 75:
        adjusted -= 8
    elif 45 <= rsi <= 65:
        adjusted += 4

    # Volume confirmation.
    if vol_ratio > 1.5 and r1 > 0:
        adjusted += 5
    elif vol_ratio > 1.5 and r1 < 0:
        adjusted -= 8

    # Macro filter.
    if macro < 40:
        adjusted -= 10
    elif macro > 60:
        adjusted += 4

    return round(clamp(adjusted), 2)

def calculate_trade_plan(score: Dict, market: Dict) -> Dict:
    price = float(market.get("close") or 0)
    risk_adj = calculate_risk_adjusted_score(score, market)
    rsi = float(market.get("rsi") or 50)
    r20 = float(market.get("return_20d") or 0)

    if risk_adj >= 75:
        action = "BUY"
    elif risk_adj >= 65:
        action = "WATCH"
    elif risk_adj <= 42:
        action = "SELL / AVOID"
    else:
        action = "HOLD"

    stop_loss = price * 0.92 if price else 0
    trailing_stop = "10%"
    take_profit_1 = price * 1.10 if price else 0
    take_profit_2 = price * 1.18 if price else 0

    if rsi > 75:
        action = "WAIT - OVEREXTENDED"

    if r20 < -0.12:
        action = "AVOID - DOWNTREND"

    return {
        "Risk Adjusted Score": risk_adj,
        "Plan": action,
        "Stop Loss": round(stop_loss, 2),
        "Trailing Stop": trailing_stop,
        "Take Profit 1": round(take_profit_1, 2),
        "Take Profit 2": round(take_profit_2, 2),
    }


# =========================
# REGIME + RELATIVE STRENGTH + ALLOCATION ENGINE
# =========================

def get_market_regime_snapshot() -> Dict:
    symbols = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "^VIX": "VIX",
        "^TNX": "10Y Yield",
    }

    data = {}

    for symbol, name in symbols.items():
        try:
            df = yf.download(symbol, period="1y", progress=False, auto_adjust=True)

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df["ma_20"] = df["Close"].rolling(20).mean()
            df["ma_50"] = df["Close"].rolling(50).mean()
            df["ma_200"] = df["Close"].rolling(200).mean()
            df["return_5d"] = df["Close"].pct_change(5)
            df["return_20d"] = df["Close"].pct_change(20)

            latest = df.iloc[-1]

            data[symbol] = {
                "name": name,
                "price": float(latest["Close"]),
                "ma_20": float(latest["ma_20"]) if not pd.isna(latest["ma_20"]) else None,
                "ma_50": float(latest["ma_50"]) if not pd.isna(latest["ma_50"]) else None,
                "ma_200": float(latest["ma_200"]) if not pd.isna(latest["ma_200"]) else None,
                "return_5d": float(latest["return_5d"]) if not pd.isna(latest["return_5d"]) else 0,
                "return_20d": float(latest["return_20d"]) if not pd.isna(latest["return_20d"]) else 0,
            }

        except Exception:
            pass

    score = 50
    reasons = []

    spy = data.get("SPY")
    qqq = data.get("QQQ")
    vix = data.get("^VIX")
    tnx = data.get("^TNX")

    for item, label in [(spy, "SPY"), (qqq, "QQQ")]:
        if not item:
            continue

        if item["ma_50"] and item["price"] > item["ma_50"]:
            score += 12
            reasons.append(f"{label} is above its 50-day moving average.")
        else:
            score -= 12
            reasons.append(f"{label} is below its 50-day moving average.")

        if item["ma_200"] and item["price"] > item["ma_200"]:
            score += 10
            reasons.append(f"{label} is above its 200-day moving average.")
        else:
            score -= 10
            reasons.append(f"{label} is below its 200-day moving average.")

        if item["return_20d"] > 0.04:
            score += 6
            reasons.append(f"{label} has positive 20-day momentum.")
        elif item["return_20d"] < -0.04:
            score -= 8
            reasons.append(f"{label} has weak 20-day momentum.")

    if vix:
        if vix["price"] > 30:
            score -= 25
            reasons.append("VIX is above 30, suggesting panic or high volatility.")
        elif vix["price"] > 22:
            score -= 12
            reasons.append("VIX is elevated, suggesting risk-off conditions.")
        elif vix["price"] < 17:
            score += 8
            reasons.append("VIX is low, suggesting calmer market conditions.")

    if tnx and tnx["return_20d"] > 0.10:
        score -= 8
        reasons.append("10Y yield has risen sharply, which can pressure growth stocks.")

    score = clamp(score)

    if score >= 75:
        regime = "BULL"
        max_exposure = 0.95
        cash_minimum = 0.05
    elif score >= 55:
        regime = "NEUTRAL"
        max_exposure = 0.70
        cash_minimum = 0.30
    elif score >= 35:
        regime = "BEAR"
        max_exposure = 0.35
        cash_minimum = 0.65
    else:
        regime = "PANIC"
        max_exposure = 0.10
        cash_minimum = 0.90

    return {
        "regime": regime,
        "score": round(score, 1),
        "max_exposure": max_exposure,
        "cash_minimum": cash_minimum,
        "reasons": reasons[:6],
        "data": data,
    }

def calculate_relative_strength(ticker: str) -> Dict:
    try:
        stock = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        spy = yf.download("SPY", period="6mo", progress=False, auto_adjust=True)
        qqq = yf.download("QQQ", period="6mo", progress=False, auto_adjust=True)

        for df in [stock, spy, qqq]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if stock.empty or spy.empty or qqq.empty:
            return {"RS vs SPY": 0, "RS vs QQQ": 0, "RS Score": 50}

        stock_20 = stock["Close"].pct_change(20).iloc[-1]
        spy_20 = spy["Close"].pct_change(20).iloc[-1]
        qqq_20 = qqq["Close"].pct_change(20).iloc[-1]

        rs_spy = (stock_20 - spy_20) * 100
        rs_qqq = (stock_20 - qqq_20) * 100

        rs_score = 50 + (rs_spy * 2.0) + (rs_qqq * 1.5)

        return {
            "RS vs SPY": round(float(rs_spy), 2),
            "RS vs QQQ": round(float(rs_qqq), 2),
            "RS Score": round(clamp(float(rs_score)), 1),
        }

    except Exception:
        return {"RS vs SPY": 0, "RS vs QQQ": 0, "RS Score": 50}

def calculate_portfolio_heat(pro_df: pd.DataFrame, regime: Dict) -> Dict:
    if pro_df.empty:
        return {"heat": 0, "label": "EMPTY", "warnings": []}

    allocation = pro_df["Suggested Allocation %"].sum()
    high_risk_count = int(pro_df["Risk"].isin(["HIGH", "VERY HIGH"]).sum())
    avg_score = float(pro_df["Score"].mean())
    cash_minimum = regime.get("cash_minimum", 0.30) * 100

    sector_exposure = (
        pro_df[pro_df["Suggested Allocation %"] > 0]
        .groupby("Sector")["Suggested Allocation %"]
        .sum()
        .sort_values(ascending=False)
    )

    heat = 0
    warnings = []

    heat += min(allocation, 100) * 0.45
    heat += high_risk_count * 4

    if regime["regime"] in ["BEAR", "PANIC"]:
        heat += 25
        warnings.append("Market regime is defensive, so high exposure is risky.")

    if allocation > (100 - cash_minimum):
        heat += 20
        warnings.append("Suggested allocation is above the regime-based exposure limit.")

    if not sector_exposure.empty and sector_exposure.iloc[0] > 35:
        heat += 15
        warnings.append(f"Portfolio is concentrated in {sector_exposure.index[0]}.")

    if avg_score < 55:
        heat += 10
        warnings.append("Average score is not strong enough for aggressive exposure.")

    heat = clamp(heat)

    if heat >= 80:
        label = "VERY HOT"
    elif heat >= 60:
        label = "HOT"
    elif heat >= 35:
        label = "MODERATE"
    else:
        label = "COOL"

    return {
        "heat": round(heat, 1),
        "label": label,
        "warnings": warnings,
    }

def capital_allocation_engine(pro_df: pd.DataFrame, regime: Dict) -> pd.DataFrame:
    if pro_df.empty:
        return pd.DataFrame()

    df = pro_df.copy()

    df["RS Score"] = df["Ticker"].apply(lambda t: calculate_relative_strength(t)["RS Score"])
    df["Combined Allocation Score"] = (
        df["Risk Adjusted Score"] * 0.45 +
        df["Confidence"] * 0.20 +
        df["RS Score"] * 0.25 +
        df["Score"] * 0.10
    )

    risk_multiplier = {
        "LOW": 1.0,
        "MEDIUM": 0.8,
        "HIGH": 0.55,
        "VERY HIGH": 0.30,
    }

    df["Risk Multiplier"] = df["Risk"].map(risk_multiplier).fillna(0.5)
    df["Adjusted Weight Score"] = df["Combined Allocation Score"] * df["Risk Multiplier"]

    df = df[
        (df["Trade Plan"].isin(["BUY", "WATCH"])) &
        (df["Risk Adjusted Score"] >= 62) &
        (df["Confidence"] >= 45)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    max_exposure = regime.get("max_exposure", 0.70) * 100
    max_single = 14 if regime["regime"] == "BULL" else 9 if regime["regime"] == "NEUTRAL" else 5

    total_weight = df["Adjusted Weight Score"].sum()

    if total_weight <= 0:
        return pd.DataFrame()

    df["Target Allocation %"] = df["Adjusted Weight Score"] / total_weight * max_exposure
    df["Target Allocation %"] = df["Target Allocation %"].clip(upper=max_single)

    # Re-normalize after capping.
    capped_total = df["Target Allocation %"].sum()
    if capped_total > max_exposure:
        df["Target Allocation %"] = df["Target Allocation %"] / capped_total * max_exposure

    df = df.sort_values("Target Allocation %", ascending=False)

    return df[[
        "Ticker", "Company", "Sector", "Target Allocation %",
        "Risk Adjusted Score", "RS Score", "Confidence", "Risk", "Trade Plan"
    ]]

# =========================
# PROFESSIONAL ANALYTICS
# =========================

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Semiconductors", "AMD": "Semiconductors",
    "AVGO": "Semiconductors", "TSM": "Semiconductors", "ASML": "Semiconductors", "AMAT": "Semiconductors",
    "MU": "Semiconductors", "ARM": "Semiconductors", "QCOM": "Semiconductors", "TXN": "Semiconductors",
    "META": "Technology", "GOOGL": "Technology", "AMZN": "Consumer/Cloud", "NFLX": "Consumer/Media",
    "CRM": "Cloud Software", "ORCL": "Cloud Software", "ADBE": "Cloud Software", "NOW": "Cloud Software",
    "PANW": "Cybersecurity", "CRWD": "Cybersecurity", "PLTR": "AI/Data", "VRT": "AI Infrastructure",
    "MRVL": "AI Infrastructure", "VEEV": "Healthcare Software", "TSLA": "Consumer/EV",
    "UBER": "Consumer/Platform", "SHOP": "Consumer/Platform", "COIN": "Crypto/Finance",
    "JPM": "Financials", "V": "Financials", "MA": "Financials", "BRK-B": "Financials",
    "LLY": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare", "ISRG": "Healthcare",
    "XOM": "Energy", "COST": "Consumer Staples", "WMT": "Consumer Staples", "PEP": "Consumer Staples",
    "HD": "Consumer Cyclical", "MCD": "Consumer", "BKNG": "Travel", "ABNB": "Travel",
    "DIS": "Media", "CAT": "Industrials", "GE": "Aerospace/Defense", "RTX": "Aerospace/Defense",
    "NKE": "Consumer", "SPY": "Benchmark", "QQQ": "Benchmark",
}

def get_risk_rating(market: Dict) -> str:
    r20 = abs(float(market.get("return_20d") or 0))
    r5 = abs(float(market.get("return_5d") or 0))
    vol_ratio = float(market.get("volume_ratio") or 1)
    rsi = float(market.get("rsi") or 50)

    risk_points = 0

    if r20 > 0.20:
        risk_points += 3
    elif r20 > 0.12:
        risk_points += 2
    elif r20 > 0.07:
        risk_points += 1

    if r5 > 0.08:
        risk_points += 2
    elif r5 > 0.05:
        risk_points += 1

    if vol_ratio > 2:
        risk_points += 2
    elif vol_ratio > 1.5:
        risk_points += 1

    if rsi > 75 or rsi < 25:
        risk_points += 2
    elif rsi > 68 or rsi < 32:
        risk_points += 1

    if risk_points >= 6:
        return "VERY HIGH"
    if risk_points >= 4:
        return "HIGH"
    if risk_points >= 2:
        return "MEDIUM"
    return "LOW"

def get_confidence_rating(score: Dict, market: Dict) -> float:
    components = [
        float(score.get("news_score") or 50),
        float(score.get("trend_score") or 50),
        float(score.get("momentum_score") or 50),
        float(score.get("volume_score") or 50),
        float(score.get("macro_risk_score") or 50),
    ]

    final_score = float(score.get("final_score") or 50)
    agreement = 100 - np.std(components)
    data_quality = float(score.get("confidence_score") or 50)

    confidence = agreement * 0.55 + data_quality * 0.30

    if final_score >= 70 or final_score <= 40:
        confidence += 10

    if market.get("close") and market.get("volume"):
        confidence += 5

    return round(clamp(confidence), 1)

def suggested_allocation(score: Dict, risk: str) -> float:
    final_score = float(score.get("final_score") or 50)

    if final_score < 60:
        return 0.0

    base = (final_score - 55) / 45 * 10

    if risk == "VERY HIGH":
        base *= 0.35
    elif risk == "HIGH":
        base *= 0.55
    elif risk == "MEDIUM":
        base *= 0.75

    return round(clamp(base, 0, 10), 1)

def expected_return_range(score: Dict, market: Dict) -> str:
    final_score = float(score.get("final_score") or 50)
    momentum = float(score.get("momentum_score") or 50)
    macro = float(score.get("macro_risk_score") or 50)

    estimate = (final_score - 50) * 0.35 + (momentum - 50) * 0.15 + (macro - 50) * 0.10

    low = estimate - 5
    high = estimate + 5

    return f"{low:.1f}% to {high:.1f}%"

def bull_bear_case(ticker: str, score: Dict, market: Dict, news: List[Dict]) -> Dict:
    company = WATCHLIST.get(ticker, ticker)
    final_score = float(score.get("final_score") or 50)
    momentum = float(score.get("momentum_score") or 50)
    news_score = float(score.get("news_score") or 50)
    macro = float(score.get("macro_risk_score") or 50)
    r20 = float(market.get("return_20d") or 0) * 100

    bull = []
    bear = []

    if news_score >= 65:
        bull.append("Recent company news sentiment is positive.")
    elif news_score <= 40:
        bear.append("Recent company news sentiment is weak or risk-heavy.")

    if momentum >= 65:
        bull.append("Price momentum is strong compared with recent history.")
    elif momentum <= 40:
        bear.append("Price momentum is weak.")

    if macro >= 60:
        bull.append("Macro conditions are not strongly negative for this stock.")
    elif macro <= 40:
        bear.append("Macro or government/economic news is adding risk.")

    if r20 > 8:
        bull.append(f"The stock is up {r20:.1f}% over 20 trading days.")
    elif r20 < -8:
        bear.append(f"The stock is down {abs(r20):.1f}% over 20 trading days.")

    if news:
        bull.append(f"Important headline: {news[0].get('title')}")

    if not bull:
        bull.append(f"{company} could benefit if its sector strengthens and market risk stays controlled.")

    if not bear:
        bear.append("Main risk is that sentiment, momentum, or macro conditions weaken after the signal.")

    if final_score >= 70:
        recommendation = "BUY / STRONG WATCH"
    elif final_score >= 60:
        recommendation = "WATCH"
    elif final_score <= 45:
        recommendation = "AVOID / SELL"
    else:
        recommendation = "HOLD / NEUTRAL"

    return {
        "bull": bull[:4],
        "bear": bear[:4],
        "recommendation": recommendation,
    }

def build_professional_table(db: Database, watchlist: Dict) -> pd.DataFrame:
    rows = []

    for ticker, company in watchlist.items():
        score = db.latest_score(ticker)
        market = db.latest_market_row(ticker)

        if not score or not market:
            continue

        risk = get_risk_rating(market)
        confidence = get_confidence_rating(score, market)
        allocation = suggested_allocation(score, risk)
        trade_plan = calculate_trade_plan(score, market)

        rs = calculate_relative_strength(ticker)

        rows.append({
            "Ticker": ticker,
            "Company": company,
            "Sector": SECTOR_MAP.get(ticker, "Other"),
            "Relative Strength vs SPY": rs["RS vs SPY"],
            "Relative Strength vs QQQ": rs["RS vs QQQ"],
            "Relative Strength Score": rs["RS Score"],
            "Price": float(market.get("close") or 0),
            "Score": float(score.get("final_score") or 0),
            "Risk Adjusted Score": trade_plan["Risk Adjusted Score"],
            "Trade Plan": trade_plan["Plan"],
            "Stop Loss": trade_plan["Stop Loss"],
            "Take Profit 1": trade_plan["Take Profit 1"],
            "Take Profit 2": trade_plan["Take Profit 2"],
            "Signal": score.get("signal"),
            "Confidence": confidence,
            "Risk": risk,
            "Suggested Allocation %": allocation,
            "Expected 6M Return": expected_return_range(score, market),
            "1D %": float(market.get("return_1d") or 0) * 100,
            "5D %": float(market.get("return_5d") or 0) * 100,
            "20D %": float(market.get("return_20d") or 0) * 100,
            "RSI": float(market.get("rsi") or 0),
            "Volume Ratio": float(market.get("volume_ratio") or 0),
        })

    return pd.DataFrame(rows)

def portfolio_attribution_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    buys = trades[trades["Action"] == "BUY"].copy()
    sells = trades[trades["Action"] == "SELL"].copy()

    if buys.empty or sells.empty:
        return pd.DataFrame()

    rows = []

    for ticker in sorted(set(buys["Ticker"]).intersection(set(sells["Ticker"]))):
        first_buy = buys[buys["Ticker"] == ticker].iloc[0]
        last_sell = sells[sells["Ticker"] == ticker].iloc[-1]

        buy_value = float(first_buy["Value"])
        sell_value = float(last_sell["Value"])
        pnl = sell_value - buy_value

        rows.append({
            "Ticker": ticker,
            "Approx P/L": round(pnl, 2),
            "Buy Price": first_buy["Price"],
            "Sell Price": last_sell["Price"],
        })

    return pd.DataFrame(rows).sort_values("Approx P/L", ascending=False) if rows else pd.DataFrame()

def get_upcoming_earnings_placeholder(ticker: str) -> str:
    return "Add Finnhub/Polygon key for accurate earnings date"

def get_analyst_placeholder(ticker: str) -> str:
    return "Add Finnhub/Polygon/Benzinga key for analyst upgrades"


def automation_trade_plan_table(db: Database, watchlist: Dict, regime: Optional[Dict] = None, test_mode: str = "Off", test_ticker: Optional[str] = None) -> pd.DataFrame:
    if regime is None:
        regime = get_market_regime_snapshot()

    rows = []
    portfolio_value = PaperTrader(db).portfolio_value()
    cash = db.get_cash()
    current_exposure = 0 if portfolio_value <= 0 else max(0, min(1, (portfolio_value - cash) / portfolio_value))

    for ticker, company in watchlist.items():
        score = db.latest_score(ticker)
        market = db.latest_market_row(ticker)

        if not score or not market:
            continue

        final_score = float(score.get("final_score") or 0)
        risk_adjusted = calculate_risk_adjusted_score(score, market)
        confidence = get_confidence_rating(score, market)
        risk = get_risk_rating(market)
        signal = score.get("signal")
        position = db.get_position(ticker)

        if test_mode != "Off" and ticker == test_ticker:
            if test_mode == "Force Approved Buy":
                signal = "BUY"
                final_score = max(final_score, 88)
                risk_adjusted = max(risk_adjusted, 82)
                confidence = max(confidence, 78)
            elif test_mode == "Force Blocked Buy":
                signal = "BUY"
                final_score = max(final_score, 78)
                risk_adjusted = min(risk_adjusted, 55)
                confidence = min(confidence, 52)
            elif test_mode == "Force Sell":
                signal = "SELL"
                position = position or {"ticker": ticker, "shares": 1, "avg_price": close}

        decision = "HOLD"
        allowed = False
        suggested_dollars = 0.0
        reasons = []

        close = float(market.get("close") or 0)
        ma20 = float(market.get("ma_20") or 0)
        ma50 = float(market.get("ma_50") or 0)
        rsi = float(market.get("rsi") or 50)

        if signal == "BUY" and not position:
            decision = "BUY CANDIDATE"
            allowed = True

            if final_score < BUY_THRESHOLD:
                allowed = False
                reasons.append(f"score {final_score:.1f} below buy threshold")
            if risk_adjusted < 68:
                allowed = False
                reasons.append(f"risk-adjusted score {risk_adjusted:.1f} below 68")
            if confidence < 60:
                allowed = False
                reasons.append(f"confidence {confidence:.1f} below 60")
            if regime["score"] < 45:
                allowed = False
                reasons.append("market regime too weak")
            if close and ma20 and close < ma20:
                allowed = False
                reasons.append("below 20-day moving average")
            if close and ma50 and close < ma50:
                allowed = False
                reasons.append("below 50-day moving average")
            if rsi > 78:
                allowed = False
                reasons.append("RSI too overbought")
            if current_exposure >= regime["max_exposure"]:
                allowed = False
                reasons.append("portfolio exposure already too high")
            if cash <= portfolio_value * regime["cash_minimum"]:
                allowed = False
                reasons.append("cash reserve too low")

            if allowed:
                base_fraction = 0.03
                if final_score >= 85 and confidence >= 75 and risk_adjusted >= 78:
                    base_fraction = 0.10
                elif final_score >= 78 and confidence >= 68:
                    base_fraction = 0.07
                elif final_score >= 72:
                    base_fraction = 0.05

                if regime["score"] < 55:
                    base_fraction *= 0.50
                elif regime["score"] > 70:
                    base_fraction *= 1.10

                if risk == "VERY HIGH":
                    base_fraction *= 0.35
                elif risk == "HIGH":
                    base_fraction *= 0.55
                elif risk == "MEDIUM":
                    base_fraction *= 0.75

                suggested_dollars = min(cash, portfolio_value * min(base_fraction, MAX_POSITION_SIZE))
                reasons.append("approved by automation rules")
            else:
                decision = "BLOCKED BUY"

        elif signal in ["SELL", "EMERGENCY_SELL"] and position:
            decision = "SELL"
            allowed = True
            reasons.append("sell signal active and position exists")

        elif signal == "BUY" and position:
            decision = "HOLD EXISTING"
            reasons.append("already holding position")

        else:
            decision = "HOLD"
            reasons.append("no executable signal")

        rows.append({
            "Ticker": ticker,
            "Company": company,
            "Signal": signal,
            "Decision": decision,
            "Allowed": allowed,
            "Suggested Dollars": round(float(suggested_dollars), 2),
            "Score": round(final_score, 1),
            "Risk Adjusted Score": round(risk_adjusted, 1),
            "Confidence": round(confidence, 1),
            "Risk": risk,
            "Position Exists": bool(position),
            "Main Reason": "; ".join(reasons[:3]),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    priority = {
        "SELL": 0,
        "BUY CANDIDATE": 1,
        "BLOCKED BUY": 2,
        "HOLD EXISTING": 3,
        "HOLD": 4,
    }
    out["_priority"] = out["Decision"].map(priority).fillna(9)
    out = out.sort_values(["_priority", "Risk Adjusted Score", "Confidence"], ascending=[True, False, False])
    return out.drop(columns=["_priority"])



def alpaca_headers() -> Dict:
    return {
        "APCA-API-KEY-ID": secret_value("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": secret_value("ALPACA_SECRET_KEY", ""),
        "Content-Type": "application/json",
    }


def alpaca_base_url() -> str:
    return secret_value("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def alpaca_keys_ready() -> bool:
    return bool(secret_value("ALPACA_API_KEY") and secret_value("ALPACA_SECRET_KEY"))


def fetch_alpaca_account() -> Dict:
    if not alpaca_keys_ready():
        return {
            "ok": False,
            "error": "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.",
        }

    try:
        url = f"{alpaca_base_url()}/v2/account"
        r = requests.get(url, headers=alpaca_headers(), timeout=15)
        data = r.json()
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "data": data,
            "error": data.get("message") if isinstance(data, dict) else None,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


def submit_alpaca_paper_order(ticker: str, side: str, dollars: float = 0.0) -> Dict:
    if secret_value("ENABLE_BROKER_ORDERS", secret_value("ENABLE_LIVE_ORDERS", "false")).lower() != "true":
        return {
            "ok": False,
            "blocked": True,
            "error": "ENABLE_BROKER_ORDERS is false, so no Alpaca order was sent.",
        }

    if not alpaca_keys_ready():
        return {
            "ok": False,
            "error": "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.",
        }

    if not ticker:
        return {
            "ok": False,
            "error": "Missing ticker.",
        }

    payload = {
        "symbol": ticker,
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }

    if side == "buy":
        if dollars <= 0:
            return {
                "ok": False,
                "error": "Buy order needs dollars greater than 0.",
            }
        payload["notional"] = round(float(dollars), 2)
    else:
        payload["qty"] = "all"

    try:
        url = f"{alpaca_base_url()}/v2/orders"
        r = requests.post(url, headers=alpaca_headers(), json=payload, timeout=15)
        try:
            data = r.json()
        except Exception:
            data = {"raw_text": r.text}

        return {
            "ok": 200 <= r.status_code < 300,
            "status_code": r.status_code,
            "data": data,
            "payload": payload,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "payload": payload,
        }



def automation_safety_snapshot(db: Database, plan_df: pd.DataFrame) -> Dict:
    trader = PaperTrader(db)
    portfolio_value = trader.portfolio_value()
    cash = db.get_cash()
    exposure = 0 if portfolio_value <= 0 else max(0, min(1, (portfolio_value - cash) / portfolio_value))

    approved = plan_df[plan_df["Decision"].isin(["BUY CANDIDATE", "SELL"])] if not plan_df.empty else pd.DataFrame()
    approved_buy_dollars = 0.0
    if not approved.empty and "Suggested Dollars" in approved.columns:
        approved_buy_dollars = float(approved[approved["Decision"] == "BUY CANDIDATE"]["Suggested Dollars"].sum())

    warnings = []

    if exposure > 0.80:
        warnings.append("Portfolio exposure is above 80%.")
    if cash < portfolio_value * 0.10:
        warnings.append("Cash is below 10% of portfolio value.")
    if approved_buy_dollars > portfolio_value * 0.25:
        warnings.append("Approved buy dollars exceed 25% of portfolio value.")
    if len(approved) > 5:
        warnings.append("More than 5 actions are approved in one run.")

    status = "OK"
    if warnings:
        status = "CAUTION"
    if exposure > 0.95 or approved_buy_dollars > portfolio_value * 0.40:
        status = "BLOCK"

    return {
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(cash, 2),
        "exposure_percent": round(exposure * 100, 1),
        "approved_actions": int(len(approved)),
        "approved_buy_dollars": round(approved_buy_dollars, 2),
        "status": status,
        "warnings": warnings,
    }


def get_recent_trade_journal(db: Database, limit: int = 50) -> pd.DataFrame:
    try:
        cur = db.conn.cursor()
        cur.execute("""
            SELECT date, ticker, action, shares, price, cash_after, portfolio_value, reason
            FROM paper_trades
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        cols = [x[0] for x in cur.description]
        return pd.DataFrame([dict(zip(cols, row)) for row in rows])
    except Exception:
        return pd.DataFrame()



def fetch_alpaca_orders(status: str = "all", limit: int = 50) -> Dict:
    if not alpaca_keys_ready():
        return {
            "ok": False,
            "error": "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.",
            "data": [],
        }

    try:
        url = f"{alpaca_base_url()}/v2/orders"
        params = {
            "status": status,
            "limit": limit,
            "direction": "desc",
        }
        r = requests.get(url, headers=alpaca_headers(), params=params, timeout=15)
        data = r.json()
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "data": data if isinstance(data, list) else [],
            "error": data.get("message") if isinstance(data, dict) else None,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "data": [],
        }


def fetch_alpaca_positions() -> Dict:
    if not alpaca_keys_ready():
        return {
            "ok": False,
            "error": "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.",
            "data": [],
        }

    try:
        url = f"{alpaca_base_url()}/v2/positions"
        r = requests.get(url, headers=alpaca_headers(), timeout=15)
        data = r.json()
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "data": data if isinstance(data, list) else [],
            "error": data.get("message") if isinstance(data, dict) else None,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "data": [],
        }


def alpaca_orders_dataframe(raw_orders: List[Dict]) -> pd.DataFrame:
    rows = []
    for order in raw_orders:
        rows.append({
            "Submitted At": order.get("submitted_at"),
            "Symbol": order.get("symbol"),
            "Side": order.get("side"),
            "Type": order.get("type"),
            "Status": order.get("status"),
            "Qty": order.get("qty"),
            "Notional": order.get("notional"),
            "Filled Qty": order.get("filled_qty"),
            "Filled Avg Price": order.get("filled_avg_price"),
            "Time In Force": order.get("time_in_force"),
            "Order ID": order.get("id"),
        })
    return pd.DataFrame(rows)


def alpaca_positions_dataframe(raw_positions: List[Dict]) -> pd.DataFrame:
    rows = []
    for pos in raw_positions:
        rows.append({
            "Symbol": pos.get("symbol"),
            "Qty": pos.get("qty"),
            "Market Value": pos.get("market_value"),
            "Avg Entry Price": pos.get("avg_entry_price"),
            "Current Price": pos.get("current_price"),
            "Unrealized P/L": pos.get("unrealized_pl"),
            "Unrealized P/L %": pos.get("unrealized_plpc"),
            "Side": pos.get("side"),
        })
    return pd.DataFrame(rows)


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

    if "custom_watchlist" not in st.session_state:
        st.session_state.custom_watchlist = {}

    active_watchlist = dict(WATCHLIST)
    active_watchlist.update(st.session_state.custom_watchlist)

    st.sidebar.title("Dashboard Controls")
    st.sidebar.caption("Refresh pulls newest prices, company news, official macro feeds, and AI scores.")
    auto_refresh = True
    selected_period = st.sidebar.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    selected_chart_type = st.sidebar.selectbox("Chart type", ["Close price", "Volume", "Close + moving averages"], index=0)

    st.sidebar.divider()
    st.sidebar.subheader("Add a Stock")
    custom_ticker = st.sidebar.text_input("Ticker symbol", placeholder="Example: SMCI").upper().strip()
    custom_name = st.sidebar.text_input("Company name", placeholder="Example: Super Micro Computer").strip()

    if st.sidebar.button("Add Stock to Dashboard"):
        if custom_ticker:
            if not custom_name:
                custom_name = custom_ticker
            st.session_state.custom_watchlist[custom_ticker] = custom_name

            with st.spinner(f"Adding {custom_ticker}..."):
                try:
                    market = MarketDataCollector(db)
                    news = NewsCollector(db)
                    scorer = StockScorer(db)

                    market.fetch(custom_ticker)
                    news.fetch_for_ticker(custom_ticker, custom_name)
                    scorer.score_stock(custom_ticker, custom_name)
                    st.success(f"Added {custom_ticker}.")
                except Exception as e:
                    st.error(f"Could not add {custom_ticker}: {e}")
            st.rerun()

    if st.sidebar.button("Refresh All Data"):
        with st.spinner("Updating prices, news, trends, and scores..."):
            bot = StockBot()
            bot.run_daily_update()

            for t, name in st.session_state.custom_watchlist.items():
                try:
                    market = MarketDataCollector(db)
                    news = NewsCollector(db)
                    scorer = StockScorer(db)
                    market.fetch(t)
                    news.fetch_for_ticker(t, name)
                    scorer.score_stock(t, name)
                except Exception:
                    pass

        st.success("Data refreshed.")
        st.rerun()

    existing_rows = []
    for ticker in active_watchlist:
        if db.latest_market_row(ticker) and db.latest_score(ticker):
            existing_rows.append(ticker)

    if auto_refresh and len(existing_rows) == 0:
        with st.spinner("First launch: loading live prices, news, and scores. This may take a bit."):
            bot = StockBot()
            bot.run_daily_update()
        st.rerun()

    rows = []

    for ticker, company in active_watchlist.items():
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

    st.subheader("Portfolio Change Tracker")
    st.caption("This will become more useful after the app has multiple days of saved paper-trading history. Right now it is the first live-tracking stage.")

    current_value = trader.portfolio_value()
    start_value = STARTING_CASH
    portfolio_change = current_value - start_value
    portfolio_change_pct = (current_value / start_value - 1) * 100 if start_value else 0

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Starting Value", f"${start_value:,.2f}")
    pc2.metric("Current Value", f"${current_value:,.2f}")
    pc3.metric("Total Change", f"${portfolio_change:,.2f}", f"{portfolio_change_pct:.2f}%")

    if abs(portfolio_change) < 1:
        st.info("No real portfolio movement yet because the app has just started tracking. As paper trades happen, this section will show gains, losses, and portfolio changes.")

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Market Dashboard",
        "Stock Deep Dive",
        "Algorithm Simulator",
        "News & Explanations",
        "Risk & Allocation",
        "Automation Trade Plan",
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

        st.info(COMPANY_DESCRIPTIONS.get(selected, f"{selected} is a user-added stock. The dashboard will score it using price, volume, trend, macro, and available company news."))

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
        st.subheader("Official Macro / Government News")
        macro_news = db.recent_macro_news(days=7)

        if macro_news:
            macro_df = pd.DataFrame(macro_news)
            st.dataframe(macro_df[["source", "title", "impact_score", "sentiment", "published_at"]], width="stretch", hide_index=True)
        else:
            st.info("No recent macro/government news saved yet. Click Refresh All Data.")

        st.subheader("Recent Filtered Company Headlines")
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
                st.write("Company description:")
                st.info(COMPANY_DESCRIPTIONS.get(ticker, f"{ticker} is a user-added stock. The app will analyze its price, volume, trend, and available news data."))
                st.write("Signal reasons:")
                for reason in score.get("reasons", []):
                    st.write(f"- {reason}")



    with tab5:
        st.subheader("Market Regime & Capital Allocation Engine")
        st.caption("This section decides how aggressive the algorithm should be based on market conditions, relative strength, and portfolio heat.")

        regime = get_market_regime_snapshot()

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Market Regime", regime["regime"])
        r2.metric("Regime Score", f"{regime['score']:.1f}/100")
        r3.metric("Max Exposure", f"{regime['max_exposure'] * 100:.0f}%")
        r4.metric("Minimum Cash", f"{regime['cash_minimum'] * 100:.0f}%")

        st.write("Regime reasons:")
        for reason in regime["reasons"]:
            st.write(f"- {reason}")

        pro_df = build_professional_table(db, active_watchlist)

        if pro_df.empty:
            st.info("No allocation data available yet. Refresh data first.")
        else:
            allocation_df = capital_allocation_engine(pro_df, regime)
            heat = calculate_portfolio_heat(pro_df, regime)

            h1, h2 = st.columns(2)
            h1.metric("Portfolio Heat", f"{heat['heat']:.1f}/100")
            h2.metric("Heat Label", heat["label"])

            if heat["warnings"]:
                st.warning(" ".join(heat["warnings"]))
            else:
                st.success("Portfolio heat is controlled under the current regime.")

            st.subheader("Capital Allocation Plan")

            if allocation_df.empty:
                st.info("No stocks qualify for allocation under the current risk rules. Cash is preferred.")
            else:
                st.dataframe(
                    allocation_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Target Allocation %": st.column_config.NumberColumn("Target Allocation %", format="%.1f%%"),
                        "Risk Adjusted Score": st.column_config.ProgressColumn("Risk Adjusted Score", min_value=0, max_value=100, format="%.1f"),
                        "RS Score": st.column_config.ProgressColumn("Relative Strength", min_value=0, max_value=100, format="%.1f"),
                        "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
                    },
                )

                st.bar_chart(allocation_df.set_index("Ticker")["Target Allocation %"], width="stretch")

            st.subheader("Relative Strength Rankings")
            rs_cols = [
                "Ticker", "Company", "Sector",
                "Relative Strength vs SPY", "Relative Strength vs QQQ",
                "Relative Strength Score", "Risk Adjusted Score", "Confidence", "Risk"
            ]
            available_cols = [c for c in rs_cols if c in pro_df.columns]
            st.dataframe(
                pro_df[available_cols].sort_values("Relative Strength Score", ascending=False),
                width="stretch",
                hide_index=True,
            )


    with tab6:
        st.subheader("Automation Trade Plan")
        st.caption("This is the bridge toward future automated buying and selling. It does not place real orders. It shows what the bot would approve, block, hold, or sell based on current scores and risk rules.")

        broker_mode = secret_value("BROKER_MODE", "database_paper")
        live_orders = secret_value("ENABLE_BROKER_ORDERS", secret_value("ENABLE_LIVE_ORDERS", "false")).lower() == "true"
        manual_approval = secret_value("MANUAL_APPROVAL_REQUIRED", "true").lower() == "true"

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Broker Mode", broker_mode)
        a2.metric("Broker Orders", "ENABLED" if live_orders else "BLOCKED")
        a3.metric("Manual Approval", "ON" if manual_approval else "OFF")
        a4.metric("Automation Status", "SAFE MODE" if not live_orders else "LIVE RISK")

        if live_orders:
            st.error("Live orders are enabled. Only use this after long paper testing and broker safety checks.")
        else:
            st.success("Safe mode is active. This tab plans trades but does not send real broker orders.")

        st.divider()
        st.subheader("Force Test Mode")
        st.caption("Use this only to test whether the automation table displays buy, blocked buy, and sell states correctly. It does not change real scores or place orders.")

        test_mode = st.selectbox(
            "Test scenario",
            ["Off", "Force Approved Buy", "Force Blocked Buy", "Force Sell"],
            index=0,
            key="automation_test_mode",
        )

        test_ticker = st.selectbox(
            "Test ticker",
            list(active_watchlist.keys()),
            index=0,
            key="automation_test_ticker",
        )

        regime_for_auto = get_market_regime_snapshot()
        plan_df = automation_trade_plan_table(db, active_watchlist, regime_for_auto, test_mode=test_mode, test_ticker=test_ticker)

        if test_mode != "Off":
            st.warning(f"Test mode is active for {test_ticker}: {test_mode}. This is only a display simulation.")

        if plan_df.empty:
            st.info("No automation decisions available yet. Refresh data first.")
        else:
            buy_candidates = int((plan_df["Decision"] == "BUY CANDIDATE").sum())
            blocked_buys = int((plan_df["Decision"] == "BLOCKED BUY").sum())
            sells = int((plan_df["Decision"] == "SELL").sum())
            holds = int(plan_df["Decision"].isin(["HOLD", "HOLD EXISTING"]).sum())

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Approved Buys", buy_candidates)
            d2.metric("Blocked Buys", blocked_buys)
            d3.metric("Sells", sells)
            d4.metric("Holds", holds)

            st.subheader("Trade Decision Table")
            st.dataframe(
                plan_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Allowed": st.column_config.CheckboxColumn("Allowed"),
                    "Suggested Dollars": st.column_config.NumberColumn("Suggested Dollars", format="$%.2f"),
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                    "Risk Adjusted Score": st.column_config.ProgressColumn("Risk Adjusted Score", min_value=0, max_value=100, format="%.1f"),
                    "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f"),
                },
            )

            approved = plan_df[plan_df["Decision"].isin(["BUY CANDIDATE", "SELL"])]
            blocked = plan_df[plan_df["Decision"] == "BLOCKED BUY"]

            st.subheader("Approved Actions")
            if approved.empty:
                st.info("No buys or sells are approved right now.")
            else:
                st.dataframe(approved, width="stretch", hide_index=True)

            st.subheader("Blocked Buy Reasons")
            if blocked.empty:
                st.info("No blocked buy signals right now.")
            else:
                st.dataframe(blocked[["Ticker", "Company", "Score", "Risk Adjusted Score", "Confidence", "Risk", "Main Reason"]], width="stretch", hide_index=True)

            st.divider()
            st.subheader("Alpaca Paper Trading Connection")
            st.caption("Connects the bot to Alpaca paper trading. This is for fake broker orders first, not real money.")

            alpaca_ready = alpaca_keys_ready()
            account_response = fetch_alpaca_account()

            ap1, ap2, ap3, ap4 = st.columns(4)
            ap1.metric("Alpaca Keys", "FOUND" if alpaca_ready else "MISSING")
            ap2.metric("Base URL", "PAPER" if "paper-api" in alpaca_base_url() else "LIVE URL")
            ap3.metric("Broker Orders", "ENABLED" if live_orders else "BLOCKED")
            ap4.metric("Connection", "OK" if account_response.get("ok") else "NOT CONNECTED")

            if account_response.get("ok"):
                acct = account_response.get("data", {})
                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("Equity", f"${float(acct.get('equity', 0)):,.2f}")
                ac2.metric("Buying Power", f"${float(acct.get('buying_power', 0)):,.2f}")
                ac3.metric("Cash", f"${float(acct.get('cash', 0)):,.2f}")
                ac4.metric("Pattern Day Trader", str(acct.get("pattern_day_trader", "N/A")))
                st.success("Alpaca paper account connected.")
            else:
                st.info(account_response.get("error", "Alpaca is not connected yet."))

            st.subheader("Paper Order Test")
            st.caption("This uses the approved automation plan. It will only send an Alpaca order if Alpaca keys exist and ENABLE_BROKER_ORDERS=true.")

            executable = plan_df[plan_df["Decision"].isin(["BUY CANDIDATE", "SELL"])].copy()

            if executable.empty:
                st.info("No approved buy or sell actions are available for paper order testing. Use Force Test Mode to test the UI.")
            else:
                order_labels = [
                    f"{row['Decision']} | {row['Ticker']} | ${float(row['Suggested Dollars']):,.2f}"
                    for _, row in executable.iterrows()
                ]

                selected_order_label = st.selectbox("Choose approved action", order_labels, key="alpaca_order_select")
                selected_index = order_labels.index(selected_order_label)
                selected_order = executable.iloc[selected_index]

                st.write("Selected order:")
                st.json({
                    "ticker": selected_order["Ticker"],
                    "decision": selected_order["Decision"],
                    "suggested_dollars": float(selected_order["Suggested Dollars"]),
                    "score": float(selected_order["Score"]),
                    "risk_adjusted_score": float(selected_order["Risk Adjusted Score"]),
                    "confidence": float(selected_order["Confidence"]),
                    "reason": selected_order["Main Reason"],
                })

                confirm_order = st.checkbox("I understand this is for Alpaca paper trading and should not be used with real money yet.", key="alpaca_confirm")

                if st.button("Submit Alpaca Paper Order", disabled=not confirm_order):
                    side = "buy" if selected_order["Decision"] == "BUY CANDIDATE" else "sell"
                    result = submit_alpaca_paper_order(
                        ticker=selected_order["Ticker"],
                        side=side,
                        dollars=float(selected_order["Suggested Dollars"]),
                    )

                    if result.get("ok"):
                        st.success("Paper order submitted to Alpaca.")
                    elif result.get("blocked"):
                        st.warning(result.get("error"))
                    else:
                        st.error(result.get("error", "Order failed."))

                    st.json(result)

        st.divider()
        st.subheader("Safety & Order Journal")
        st.caption("These controls help prevent the bot from taking too much risk when future order sending is enabled.")

        safety = automation_safety_snapshot(db, plan_df if "plan_df" in locals() else pd.DataFrame())

        sg1, sg2, sg3, sg4, sg5 = st.columns(5)
        sg1.metric("Safety Status", safety["status"])
        sg2.metric("Portfolio Value", f"${safety['portfolio_value']:,.2f}")
        sg3.metric("Cash", f"${safety['cash']:,.2f}")
        sg4.metric("Exposure", f"{safety['exposure_percent']:.1f}%")
        sg5.metric("Approved Buy $", f"${safety['approved_buy_dollars']:,.2f}")

        if safety["status"] == "BLOCK":
            st.error("Safety engine would block new buy orders.")
        elif safety["status"] == "CAUTION":
            st.warning("Safety engine is cautious: " + " ".join(safety["warnings"]))
        else:
            st.success("Safety engine status is OK.")

        emergency_stop = st.checkbox(
            "Emergency stop: block all future order sending from the UI",
            value=True,
            key="automation_emergency_stop",
        )

        if emergency_stop:
            st.info("Emergency stop is ON. The UI should not send orders even if broker settings are later enabled.")
        else:
            st.warning("Emergency stop is OFF. Only disable this during controlled paper-trading tests.")

        journal_df = get_recent_trade_journal(db)

        st.subheader("Recent Paper Trade Journal")
        if journal_df.empty:
            st.info("No paper trades have been logged yet.")
        else:
            st.dataframe(journal_df, width="stretch", hide_index=True)
            st.download_button(
                "Download Trade Journal CSV",
                data=journal_df.to_csv(index=False),
                file_name="paper_trade_journal.csv",
                mime="text/csv",
            )

        st.divider()
        st.subheader("Alpaca Order Reconciliation")
        st.caption("After a paper order is submitted, this section verifies broker orders and current paper positions directly from Alpaca.")

        if not alpaca_keys_ready():
            st.info("Add Alpaca paper API keys before order reconciliation can connect.")
        else:
            col_orders, col_positions = st.columns(2)

            with col_orders:
                st.write("Recent Alpaca Orders")
                order_status_filter = st.selectbox(
                    "Order status filter",
                    ["all", "open", "closed"],
                    index=0,
                    key="alpaca_order_status_filter",
                )
                orders_response = fetch_alpaca_orders(status=order_status_filter, limit=50)

                if orders_response.get("ok"):
                    orders_df = alpaca_orders_dataframe(orders_response.get("data", []))
                    if orders_df.empty:
                        st.info("No Alpaca orders found.")
                    else:
                        st.dataframe(orders_df, width="stretch", hide_index=True)
                        st.download_button(
                            "Download Alpaca Orders CSV",
                            data=orders_df.to_csv(index=False),
                            file_name="alpaca_orders.csv",
                            mime="text/csv",
                        )
                else:
                    st.error(orders_response.get("error", "Could not fetch Alpaca orders."))

            with col_positions:
                st.write("Current Alpaca Positions")
                positions_response = fetch_alpaca_positions()

                if positions_response.get("ok"):
                    positions_df = alpaca_positions_dataframe(positions_response.get("data", []))
                    if positions_df.empty:
                        st.info("No Alpaca paper positions found.")
                    else:
                        st.dataframe(positions_df, width="stretch", hide_index=True)
                        st.download_button(
                            "Download Alpaca Positions CSV",
                            data=positions_df.to_csv(index=False),
                            file_name="alpaca_positions.csv",
                            mime="text/csv",
                        )
                else:
                    st.error(positions_response.get("error", "Could not fetch Alpaca positions."))

        st.warning("Future live trading should only be connected after broker paper trading, logging, kill switches, and order reconciliation are tested.")


# =========================
# UTILITY COMMANDS
# =========================

def run_backtest():
    backtester = Backtester()
    results = []

    for ticker in active_watchlist:
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