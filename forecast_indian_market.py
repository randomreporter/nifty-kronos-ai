import sys
import os

# Suppress HuggingFace hub symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch

# Ensure Kronos package imports work properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

def resolve_ticker(input_ticker: str) -> str:
    """Normalize input symbol for Indian Market Yahoo Finance tickers."""
    ticker = input_ticker.strip().upper()
    mappings = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "SENSEX": "^BSESN",
        "BANKNIFTY": "^NSEBANK",
        "NIFTYBANK": "^NSEBANK",
    }
    if ticker in mappings:
        return mappings[ticker]
    # If not starting with '^' or ending with '.NS' or '.BO' or '-USD'
    if not (ticker.startswith("^") or ticker.endswith(".NS") or ticker.endswith(".BO") or "-" in ticker):
        return f"{ticker}.NS"
    return ticker

def fetch_market_data(ticker_code: str, period: str, interval: str) -> pd.DataFrame:
    """Robust market data fetcher with multiple fallbacks for cloud servers."""
    # Method 1: yf.Ticker.history()
    try:
        t = yf.Ticker(ticker_code)
        raw = t.history(period=period, interval=interval)
        if not raw.empty and len(raw) > 10:
            return raw
    except Exception:
        pass

    # Method 2: yf.download()
    try:
        raw = yf.download(ticker_code, period=period, interval=interval, progress=False, auto_adjust=False)
        if not raw.empty and len(raw) > 10:
            return raw
    except Exception:
        pass

    # Method 3: Fallback for index symbols on Cloud IPs
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

def main():
    raw_ticker = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "daily"

    ticker = resolve_ticker(raw_ticker)
    print(f"\n=======================================================")
    print(f"       Kronos AI Forecast - Indian Market Module       ")
    print(f"=======================================================")
    print(f"Target Asset : {ticker} (Input: '{raw_ticker}')")
    print(f"Timeframe    : {mode.upper()}")

    # Set parameters based on timeframe
    if mode == "daily":
        interval, period, pred_len = "1d", "3y", 20
        step = pd.offsets.BDay(1)
        lookback = 400
    elif mode == "5m":
        interval, period, pred_len = "5m", "5d", 6
        step = pd.Timedelta(minutes=5)
        lookback = 300
    else:  # hourly default
        interval, period, pred_len = "1h", "60d", 24
        step = pd.Timedelta(hours=1)
        lookback = 400

    # 1. Download market data
    print(f"Fetching historical market data from Yahoo Finance...")
    raw = fetch_market_data(ticker, period=period, interval=interval)
    
    if raw.empty:
        print(f"ERROR: Could not fetch data for '{ticker}'. Please check the symbol.")
        return

    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    df = raw[["open", "high", "low", "close", "volume"]].dropna().tail(lookback).reset_index()
    df = df.rename(columns={df.columns[0]: "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)

    last_close = df["close"].iloc[-1]
    last_time = df["timestamps"].iloc[-1]
    print(f"Latest Market Price : INR {last_close:.2f} (Timestamp: {last_time})")

    # 2. Load Model & Tokenizer
    print("\nLoading Kronos Foundation Model weights...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device      : {device.upper()}")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    x_ts = df["timestamps"]
    y_ts = pd.Series([x_ts.iloc[-1] + step * (i + 1) for i in range(pred_len)])

    # 3. Generate sample prediction paths
    paths_count = 20
    print(f"Generating {paths_count} probabilistic future paths...")
    paths = []
    for i in range(paths_count):
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

    print("\n-------------------------------------------------------")
    print("                PROBABILISTIC FORECAST TABLE           ")
    print("-------------------------------------------------------")
    print(band.head(5).round(2).to_string())

    # 4. Analyze Next-Period Option Trade Skew
    next_close_median = band["likely_50%"].iloc[0]
    next_pct_change = ((next_close_median - last_close) / last_close) * 100

    print("\n=======================================================")
    print("           NEXT STEP / OVERNIGHT TRADING SIGNAL        ")
    print("=======================================================")
    print(f"Current Price    : INR {last_close:.2f}")
    print(f"Next Target (50%): INR {next_close_median:.2f} ({next_pct_change:+.2f}%)")
    print(f"5% Bearish Band  : INR {band['low_5%'].iloc[0]:.2f}")
    print(f"95% Bullish Band : INR {band['high_95%'].iloc[0]:.2f}")

    if next_pct_change >= 0.5:
        signal = "[BULLISH SIGNAL] - Consider BULL CALL SPREAD / CALL OPTION (CE)"
        note = "Model projects a clear upward skew (> +0.5%). Prefer spreads to offset Theta decay."
    elif next_pct_change <= -0.5:
        signal = "[BEARISH SIGNAL] - Consider BEAR PUT SPREAD / PUT OPTION (PE)"
        note = "Model projects a clear downward skew (< -0.5%). Prefer spreads to offset Theta decay."
    else:
        signal = "[NEUTRAL / NO TRADE] - Flat Expected Movement"
        note = "Expected movement is within +-0.5%. Avoid buying options today due to theta decay risk."

    print(f"\nAI TRADING SIGNAL: {signal}")
    print(f"Strategy Note    : {note}")

    # 5. Plot chart and save image
    plt.figure(figsize=(11, 5))
    plot_history_len = min(120, len(df))
    plt.plot(x_ts.tail(plot_history_len), df["close"].tail(plot_history_len), color="black", label="Historical Price")
    plt.fill_between(y_ts, band["low_5%"], band["high_95%"], color="#2563eb", alpha=0.25, label="90% Confidence Interval Range")
    plt.plot(y_ts, band["likely_50%"], color="#1d4ed8", linewidth=2, label="Median Expected Path (50%)")
    plt.title(f"{ticker} - Kronos AI {mode.upper()} Price Forecast ({paths_count} sampled paths)", fontsize=12, fontweight="bold")
    plt.xlabel("Date / Time")
    plt.ylabel("Price (INR / Currency)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_plot = "forecast_plot.png"
    plt.savefig(output_plot, dpi=150)
    print(f"\nSaved forecast visualization to: {output_plot}\n")

if __name__ == "__main__":
    main()
