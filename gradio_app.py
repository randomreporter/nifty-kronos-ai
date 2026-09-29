import sys
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import gradio as gr
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

# Load Model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

def resolve_ticker(input_ticker: str) -> str:
    ticker = input_ticker.strip().upper()
    mappings = {
        "NIFTY 50": "^NSEI",
        "NIFTY": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "BANKNIFTY": "^NSEBANK",
        "BSE SENSEX": "^BSESN",
        "SENSEX": "^BSESN",
    }
    if ticker in mappings:
        return mappings[ticker]
    if not (ticker.startswith("^") or ticker.endswith(".NS") or ticker.endswith(".BO") or "-" in ticker):
        return f"{ticker}.NS"
    return ticker

def fetch_market_data(ticker_code: str, period: str, interval: str) -> pd.DataFrame:
    """Robust market data fetcher with multiple fallbacks for cloud servers."""
    try:
        t = yf.Ticker(ticker_code)
        raw = t.history(period=period, interval=interval)
        if not raw.empty and len(raw) > 10:
            return raw
    except Exception:
        pass

    try:
        raw = yf.download(ticker_code, period=period, interval=interval, progress=False, auto_adjust=False)
        if not raw.empty and len(raw) > 10:
            return raw
    except Exception:
        pass

    alt_map = {
        "^NSEI": "NIFTY.NS",
        "^NSEBANK": "BANKNIFTY.NS",
        "^BSESN": "SENSEX.BO"
    }
    if ticker_code in alt_map:
        try:
            t = yf.Ticker(alt_map[ticker_code])
            raw = t.history(period=period, interval=interval)
            if not raw.empty:
                return raw
        except Exception:
            pass

    return pd.DataFrame()

def generate_forecast(asset_name, custom_asset, timeframe_name):
    target = custom_asset if asset_name == "Custom" else asset_name
    ticker_code = resolve_ticker(target)
    
    if "Daily" in timeframe_name:
        interval, period, pred_len = "1d", "3y", 20
        step = pd.offsets.BDay(1)
        mode_str = "DAILY"
        lookback = 400
    elif "5-Minute" in timeframe_name:
        interval, period, pred_len = "5m", "5d", 6
        step = pd.Timedelta(minutes=5)
        mode_str = "5-MINUTE"
        lookback = 300
    else:
        interval, period, pred_len = "1h", "60d", 24
        step = pd.Timedelta(hours=1)
        mode_str = "HOURLY"
        lookback = 400

    raw = fetch_market_data(ticker_code, period=period, interval=interval)

    if raw.empty:
        return f"Error: Could not fetch market data for {ticker_code}", None

    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    df = raw[["open", "high", "low", "close", "volume"]].dropna().tail(lookback).reset_index()
    df = df.rename(columns={df.columns[0]: "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)

    x_ts = df["timestamps"]
    y_ts = pd.Series([x_ts.iloc[-1] + step * (i + 1) for i in range(pred_len)])

    paths = []
    for _ in range(20):
        pred = predictor.predict(
            df=df[["open", "high", "low", "close", "volume"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=1.0, top_p=0.9,
            sample_count=1,
            verbose=False
        )
        paths.append(pred["close"].values)

    closes = pd.DataFrame(paths).T
    band = pd.DataFrame({
        "low_5%": closes.quantile(0.05, axis=1),
        "likely_50%": closes.median(axis=1),
        "high_95%": closes.quantile(0.95, axis=1)
    })

    last_close = df["close"].iloc[-1]
    next_median = band["likely_50%"].iloc[0]
    pct_change = ((next_median - last_close) / last_close) * 100

    if pct_change >= 0.5:
        signal = f"🟢 BULLISH SIGNAL — Consider BULL CALL SPREAD / CALL OPTION (CE)\nExpected Gain: +{pct_change:.2f}%"
    elif pct_change <= -0.5:
        signal = f"🔴 BEARISH SIGNAL — Consider BEAR PUT SPREAD / PUT OPTION (PE)\nExpected Drop: {pct_change:.2f}%"
    else:
        signal = f"⚪ NEUTRAL / NO TRADE — Expected Move (+{pct_change:.2f}%) within +-0.5%\nAvoid buying options today due to theta decay."

    summary = (
        f"### {ticker_code} Forecast Summary\n"
        f"- **Current Price**: INR {last_close:,.2f}\n"
        f"- **Next Target (50% Median)**: INR {next_median:,.2f} ({pct_change:+.2f}%)\n"
        f"- **5% Bearish Risk Band**: INR {band['low_5%'].iloc[0]:,.2f}\n"
        f"- **95% Bullish Target Band**: INR {band['high_95%'].iloc[0]:,.2f}\n\n"
        f"**AI TRADING SIGNAL**:\n{signal}"
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_len = min(120, len(df))
    ax.plot(x_ts.tail(plot_len), df["close"].tail(plot_len), color="#1e293b", linewidth=1.5, label="Historical Price")
    ax.fill_between(y_ts, band["low_5%"], band["high_95%"], color="#3b82f6", alpha=0.25, label="90% Confidence Interval Range")
    ax.plot(y_ts, band["likely_50%"], color="#1d4ed8", linewidth=2.5, label="Median Path (50%)")
    ax.set_title(f"{ticker_code} — Kronos AI {mode_str} Price Forecast", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (INR)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    return summary, fig

demo = gr.Interface(
    fn=generate_forecast,
    inputs=[
        gr.Dropdown(["NIFTY 50", "BANK NIFTY", "BSE SENSEX", "RELIANCE", "TCS", "INFY", "HDFCBANK", "TATAMOTORS", "Custom"], value="NIFTY 50", label="Select Asset"),
        gr.Textbox(value="SBIN", label="Custom Ticker (If Custom Selected)"),
        gr.Radio(["Daily (Next-Day / BTST)", "Hourly", "5-Minute"], value="Daily (Next-Day / BTST)", label="Timeframe")
    ],
    outputs=[
        gr.Markdown(label="Forecast Analysis"),
        gr.Plot(label="Forecast Chart")
    ],
    title="📈 Kronos AI — Indian Market & Options Forecaster",
    description="Open-source AI time-series forecasting for NSE/BSE stocks & options."
)

if __name__ == "__main__":
    demo.launch()
