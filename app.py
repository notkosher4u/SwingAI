import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import json
import xml.etree.ElementTree as ET

st.set_page_config(page_title="SwingAI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 14px; border-radius: 12px;}
.small-muted {color: #888; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

st.title("📈 SwingAI")
st.caption("US Swing Trading Research • 2–10 trading days • Educational paper-analysis prototype")

with st.sidebar:
    st.header("Analysis settings")
    ticker = st.text_input("Stock ticker", value="NVDA").strip().upper()
    period = st.selectbox("Chart history", ["6mo", "1y", "2y", "5y"], index=1)
    min_score = st.slider("Minimum setup score", 0, 100, 65, 5)
    st.markdown("---")
    st.caption("V1 uses publicly accessible market data. Availability, delay and coverage may vary.")

def fetch_data(symbol, period):
    obj = yf.Ticker(symbol)
    hist = obj.history(period=period, interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError("No price history returned. Check the ticker or try again later.")
    return obj, hist

def indicators(df):
    x = df.copy()
    close = x["Close"]
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = close.ewm(span=50, adjust=False).mean()
    x["SMA200"] = close.rolling(200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    tr = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - close.shift()).abs(),
        (x["Low"] - close.shift()).abs()
    ], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()
    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["RVOL"] = x["Volume"] / x["VOL20"]
    return x

def score_latest(x):
    r = x.iloc[-1]
    prev = x.iloc[-2] if len(x) > 1 else r
    long_parts = {
        "Trend": 100 if r["Close"] > r["EMA20"] > r["EMA50"] else 55 if r["Close"] > r["EMA50"] else 15,
        "Momentum": 100 if 50 <= r["RSI14"] <= 70 and r["MACD"] > r["MACD_SIGNAL"] else 55 if r["RSI14"] > 45 else 15,
        "Volume": 100 if r["RVOL"] >= 1.5 else 65 if r["RVOL"] >= 1 else 35,
        "Structure": 100 if r["Close"] > prev["High"] else 60 if r["Close"] >= r["EMA20"] else 25,
    }
    short_parts = {
        "Trend": 100 if r["Close"] < r["EMA20"] < r["EMA50"] else 55 if r["Close"] < r["EMA50"] else 15,
        "Momentum": 100 if 30 <= r["RSI14"] <= 50 and r["MACD"] < r["MACD_SIGNAL"] else 55 if r["RSI14"] < 55 else 15,
        "Volume": 100 if r["RVOL"] >= 1.5 else 65 if r["RVOL"] >= 1 else 35,
        "Structure": 100 if r["Close"] < prev["Low"] else 60 if r["Close"] <= r["EMA20"] else 25,
    }
    weights = {"Trend": .30, "Momentum": .25, "Volume": .15, "Structure": .30}
    ls = round(sum(long_parts[k] * weights[k] for k in weights))
    ss = round(sum(short_parts[k] * weights[k] for k in weights))
    return ls, ss, long_parts, short_parts

def fmt(v):
    return "—" if pd.isna(v) else f"{v:,.2f}"

def get_news(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 SwingAI educational prototype"})
    try:
        with urlopen(req, timeout=8) as response:
            root = ET.fromstring(response.read())
        items = []
        for item in root.findall(".//item")[:8]:
            items.append({
                "title": item.findtext("title", default=""),
                "published": item.findtext("pubDate", default=""),
                "link": item.findtext("link", default="")
            })
        return items
    except Exception:
        return []

if st.button("Analyze stock", type="primary", use_container_width=True):
    if not ticker:
        st.error("Enter a ticker symbol.")
        st.stop()
    with st.spinner(f"Loading {ticker} market data..."):
        try:
            stock, raw = fetch_data(ticker, period)
            data = indicators(raw)
            info = stock.get_info()
        except Exception as e:
            st.error(f"Could not load data: {e}")
            st.stop()

    latest = data.iloc[-1]
    long_score, short_score, long_parts, short_parts = score_latest(data)
    signal = "LONG setup" if long_score >= short_score else "SHORT setup"
    best_score = max(long_score, short_score)

    st.subheader(f"{ticker} — Technical overview")
    st.caption(f"Last available daily bar: {data.index[-1].strftime('%Y-%m-%d')} • Data source: Yahoo Finance via yfinance • Not guaranteed real-time")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Last close", f"${latest['Close']:,.2f}")
    c2.metric("RSI (14)", fmt(latest["RSI14"]))
    c3.metric("ATR (14)", fmt(latest["ATR14"]))
    c4.metric("Relative volume", f"{latest['RVOL']:.2f}x" if pd.notna(latest["RVOL"]) else "—")

    left, right = st.columns([1.7,1])
    with left:
        st.markdown("#### Price and moving averages")
        chart = data[["Close","EMA20","EMA50","SMA200"]].dropna(how="all")
        st.line_chart(chart, height=360)
        st.markdown("#### Volume")
        st.bar_chart(data[["Volume"]].tail(90), height=150)
    with right:
        st.markdown("#### Setup scores")
        st.metric("Long score", f"{long_score}/100")
        st.metric("Short score", f"{short_score}/100")
        if best_score < min_score:
            st.warning("NO TRADE — score is below your selected threshold.")
        else:
            st.info(f"Highest-scoring scenario: {signal}. This is a screening signal, not a trade instruction.")
        st.markdown("#### Long score components")
        st.json(long_parts)
        st.markdown("#### Short score components")
        st.json(short_parts)

    st.markdown("#### Illustrative risk framework")
    atr = latest["ATR14"]
    close = latest["Close"]
    if pd.notna(atr) and atr > 0:
        a,b = st.columns(2)
        with a:
            st.write("**Long scenario (illustrative)**")
            st.write(f"Reference price: ${close:,.2f}")
            st.write(f"ATR-based invalidation reference: ${close-1.5*atr:,.2f}")
            st.write(f"1R target reference: ${close+1.5*atr:,.2f}")
        with b:
            st.write("**Short scenario (illustrative)**")
            st.write(f"Reference price: ${close:,.2f}")
            st.write(f"ATR-based invalidation reference: ${close+1.5*atr:,.2f}")
            st.write(f"1R target reference: ${close-1.5*atr:,.2f}")
        st.caption("These are formula demonstrations, not validated entry/stop/target levels. Real setups require price structure, spread, event and execution checks.")
    else:
        st.info("Not enough data to calculate ATR-based references.")

    st.markdown("#### Company snapshot")
    if info:
        keys = ["longName","sector","industry","marketCap","trailingPE","forwardPE","profitMargins","revenueGrowth"]
        snap = {k: info.get(k) for k in keys if info.get(k) is not None}
        st.json(snap if snap else {"status":"Company metadata unavailable"})
    else:
        st.info("Company metadata unavailable.")

    st.markdown("#### Recent headlines")
    headlines = get_news(ticker)
    if headlines:
        for item in headlines:
            st.markdown(f"- [{item['title']}]({item['link']})  \n  <span class='small-muted'>{item['published']}</span>", unsafe_allow_html=True)
    else:
        st.info("Headlines could not be loaded. News coverage is not guaranteed.")

    st.markdown("#### Raw indicator data")
    st.dataframe(data.tail(15).round(3), use_container_width=True)

st.markdown("---")
st.caption("SwingAI is an educational research prototype. Scores are heuristic and unvalidated. Data can be delayed, incomplete or incorrect. No brokerage connection, trade execution or personalized financial advice.")
