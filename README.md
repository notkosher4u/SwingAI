# SwingAI MVP — Starter

This is a local educational prototype for US stock technical research.

## What it currently does
- Accepts a US stock ticker.
- Retrieves daily historical prices through `yfinance` (Yahoo Finance access via an unofficial library).
- Calculates EMA20, EMA50, SMA200, RSI14, MACD, ATR14 and relative volume.
- Displays simple heuristic Long/Short scores.
- Shows illustrative ATR reference levels.
- Attempts to retrieve recent headlines through Google News RSS.
- Shows a basic company metadata snapshot when available.

## Important limitations
- Market data may be delayed, incomplete, rate-limited or unavailable.
- This is not a professional-grade market data feed and does not guarantee real-time quotes.
- Scores are prototype heuristics, not a validated trading strategy or probability of success.
- ATR reference levels are demonstrations, not verified trade setups.
- No broker connection, order placement, leverage or real-money trading.
- Backtesting, event filters, spread/liquidity checks, robust news analysis and proper candle charts are not yet implemented.

## Run locally (one-time setup)
Requires Python 3.10 or newer.

1. Unzip the project.
2. Open a terminal in the `swingai_mvp` folder.
3. Install dependencies:
   `python -m pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`
5. Open the local address shown in the terminal (usually http://localhost:8501).

No paid data subscription or API key is required for this prototype. Public sources can change their access rules at any time.
