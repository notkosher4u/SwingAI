import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

st.set_page_config(page_title="SwingAI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.25rem; max-width: 1450px;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 14px; border-radius: 12px;}
.small-muted {color: #888; font-size: .82rem;}
div[data-testid="stExpander"] {border: 1px solid rgba(128,128,128,.18); border-radius: 10px;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("📈 SwingAI")
st.caption("US stock research • Daily chart • Educational paper-analysis prototype")

with st.sidebar:
    st.header("Analysis settings")
    ticker = st.text_input("Stock ticker", value="AAPL", help="Enter a US stock symbol, for example AAPL or MSFT.").strip().upper()
    period = st.selectbox("Chart history", ["6mo", "1y", "2y", "5y"], index=1)
    min_score = st.slider(
        "Minimum setup score",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="The minimum heuristic score required for a setup to pass the screening threshold. This is not a probability of profit."
    )
    st.caption("A higher threshold is more selective. It does not improve or guarantee results.")
    st.divider()
    st.caption("Public market data may be delayed, incomplete or unavailable.")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol, selected_period):
    hist = yf.Ticker(symbol).history(period=selected_period, interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError("No price history returned. Check the ticker or try again later.")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(hist.columns)
    if missing:
        raise ValueError("Market data is missing required fields: " + ", ".join(sorted(missing)))
    return hist

def add_indicators(df):
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
    x.loc[(loss == 0) & (gain > 0), "RSI14"] = 100
    x.loc[(loss == 0) & (gain == 0), "RSI14"] = 50

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
    rsi = r["RSI14"]
    rvol = r["RVOL"]

    long_parts = {
        "Trend": 100 if r["Close"] > r["EMA20"] > r["EMA50"] else 55 if r["Close"] > r["EMA50"] else 15,
        "Momentum": 100 if pd.notna(rsi) and 50 <= rsi <= 70 and r["MACD"] > r["MACD_SIGNAL"] else 55 if pd.notna(rsi) and rsi > 45 else 15,
        "Volume": 100 if pd.notna(rvol) and rvol >= 1.5 else 65 if pd.notna(rvol) and rvol >= 1 else 35,
        "Structure": 100 if r["Close"] > prev["High"] else 60 if r["Close"] >= r["EMA20"] else 25,
    }
    short_parts = {
        "Trend": 100 if r["Close"] < r["EMA20"] < r["EMA50"] else 55 if r["Close"] < r["EMA50"] else 15,
        "Momentum": 100 if pd.notna(rsi) and 30 <= rsi <= 50 and r["MACD"] < r["MACD_SIGNAL"] else 55 if pd.notna(rsi) and rsi < 55 else 15,
        "Volume": 100 if pd.notna(rvol) and rvol >= 1.5 else 65 if pd.notna(rvol) and rvol >= 1 else 35,
        "Structure": 100 if r["Close"] < prev["Low"] else 60 if r["Close"] <= r["EMA20"] else 25,
    }
    weights = {"Trend": .30, "Momentum": .25, "Volume": .15, "Structure": .30}
    long_score = round(sum(long_parts[k] * weights[k] for k in weights))
    short_score = round(sum(short_parts[k] * weights[k] for k in weights))
    return long_score, short_score, long_parts, short_parts

def fmt(value):
    return "—" if pd.isna(value) else f"{value:,.2f}"

def build_candlestick_chart(data, symbol):
    chart_data = data.tail(260).copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.76, 0.24]
    )
    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        name="Price",
        increasing_line_color="#16a36a",
        decreasing_line_color="#e05252"
    ), row=1, col=1)

    for column, label, color in [
        ("EMA20", "EMA 20", "#268bd2"),
        ("EMA50", "EMA 50", "#e6a23c"),
        ("SMA200", "SMA 200", "#a78bfa")
    ]:
        fig.add_trace(go.Scatter(
            x=chart_data.index, y=chart_data[column],
            mode="lines", name=label,
            line=dict(width=1.5, color=color)
        ), row=1, col=1)

    volume_colors = np.where(
        chart_data["Close"] >= chart_data["Open"], "#16a36a", "#e05252"
    )
    fig.add_trace(go.Bar(
        x=chart_data.index, y=chart_data["Volume"],
        name="Volume", marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} · Daily candlestick chart",
        template="plotly_dark",
        height=590,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
        showlegend=True
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor="rgba(150,150,150,.15)")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="rgba(150,150,150,.15)")
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)
    return fig

def get_news(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 SwingAI educational prototype"})
    try:
        with urlopen(req, timeout=8) as response:
            root = ET.fromstring(response.read())
        items = []
        for item in root.findall(".//item")[:5]:
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
            raw = fetch_history(ticker, period)
            data = add_indicators(raw)
        except Exception as error:
            st.error(f"Could not load market data: {error}")
            st.stop()

    latest = data.iloc[-1]
    long_score, short_score, long_parts, short_parts = score_latest(data)
    best_score = max(long_score, short_score)
    signal = "LONG" if long_score >= short_score else "SHORT"

    st.subheader(f"{ticker} — Technical overview")
    st.caption(
        f"Last available daily bar: {data.index[-1].strftime('%Y-%m-%d')} · "
        "Yahoo Finance data via yfinance · Not guaranteed real-time"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"${latest['Close']:,.2f}")
    c2.metric("RSI (14)", fmt(latest["RSI14"]))
    c3.metric("ATR (14)", fmt(latest["ATR14"]))
    c4.metric("Relative volume", f"{latest['RVOL']:.2f}x" if pd.notna(latest["RVOL"]) else "—")

    left, right = st.columns([1.65, 1])
    with left:
        st.markdown("#### Price action")
        st.plotly_chart(build_candlestick_chart(data, ticker), use_container_width=True, config={"displaylogo": False})

    with right:
        st.markdown("#### Setup scores")
        st.metric("Long score", f"{long_score}/100")
        st.metric("Short score", f"{short_score}/100")
        if best_score < min_score:
            st.warning(f"Below your minimum score threshold ({min_score}). No setup passes the selected screen.")
        else:
            st.info(f"{signal} has the higher score and passes the selected screen. This is a research filter, not a trade instruction.")

        with st.expander("Score breakdown"):
            st.markdown("**Long components**")
            st.json(long_parts)
            st.markdown("**Short components**")
            st.json(short_parts)

    with st.expander("Company information"):
        try:
            info = yf.Ticker(ticker).get_info()
            keys = ["longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "profitMargins", "revenueGrowth"]
            snapshot = {key: info.get(key) for key in keys if info and info.get(key) is not None}
            if snapshot:
                st.json(snapshot)
            else:
                st.info("Company information is unavailable.")
        except Exception:
            st.info("Company information is unavailable right now.")

    with st.expander("Recent headlines"):
        headlines = get_news(ticker)
        if headlines:
            for item in headlines:
                st.markdown(f"- [{item['title']}]({item['link']})")
                if item["published"]:
                    st.caption(item["published"])
        else:
            st.info("Headlines could not be loaded. News coverage is not guaranteed.")

    with st.expander("Advanced data · Latest 15 daily rows"):
        st.dataframe(data.tail(15).round(3), use_container_width=True)

st.caption("Educational research prototype only. Scores are heuristic and unvalidated. Market data may be delayed, incomplete or incorrect. No trade execution or personalized financial advice.")
