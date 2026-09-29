import sys
import os
import urllib.parse
import requests

# Suppress HuggingFace hub symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch

# Ensure Kronos package imports work properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

st.set_page_config(
    page_title="Kronos AI — Indian Market Forecaster",
    page_icon="📈",
    layout="wide"
)

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
    """Bulletproof data fetcher using Yahoo Finance direct REST API with browser headers (Cloud-Safe)."""
    encoded_ticker = urllib.parse.quote(ticker_code)
    range_str = "3y" if period in ["3y", "max"] else period
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_ticker}?range={range_str}&interval={interval}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            chart_results = res_json.get("chart", {}).get("result")
            if chart_results and len(chart_results) > 0:
                chart = chart_results[0]
                timestamps = chart.get("timestamp", [])
                indicators = chart.get("indicators", {}).get("quote", [{}])[0]
                
                if timestamps and indicators:
                    df = pd.DataFrame({
                        "timestamps": pd.to_datetime(timestamps, unit="s"),
                        "open": indicators.get("open"),
                        "high": indicators.get("high"),
                        "low": indicators.get("low"),
                        "close": indicators.get("close"),
                        "volume": indicators.get("volume", [0] * len(timestamps))
                    })
                    df = df.dropna(subset=["close"]).reset_index(drop=True)
                    if not df.empty and len(df) > 10:
                        return df
    except Exception as e:
        print(f"Direct REST API Fetch error: {e}")
        
    # Fallback method: yfinance Ticker history
    try:
        t = yf.Ticker(ticker_code)
        raw = t.history(period=period, interval=interval)
        if not raw.empty and len(raw) > 10:
            raw.columns = [c.lower() for c in raw.columns]
            df = raw.reset_index()
            df = df.rename(columns={df.columns[0]: "timestamps"})
            df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
            df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna(subset=["close"]).reset_index(drop=True)
            return df
    except Exception:
        pass

    return pd.DataFrame()

@st.cache_resource
def load_kronos_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    return predictor, device

st.title("📈 Kronos AI — Indian Market & Options Forecaster")
st.markdown("Open-source probabilistic AI time-series forecasting for NSE/BSE stocks & indices.")

st.sidebar.header("⚙️ Forecast Controls")

preset_options = ["NIFTY 50", "BANK NIFTY", "BSE SENSEX", "RELIANCE", "TCS", "INFY", "HDFCBANK", "TATAMOTORS", "Custom"]
selected_preset = st.sidebar.selectbox("Select Asset / Index", preset_options)

if selected_preset == "Custom":
    custom_ticker = st.sidebar.text_input("Enter Ticker Symbol", value="SBIN")
    target_asset = custom_ticker
else:
    target_asset = selected_preset

timeframe = st.sidebar.radio("Select Timeframe", ["Daily (Next-Day / BTST)", "Hourly", "5-Minute"])
paths_count = st.sidebar.slider("Sample Paths Count", min_value=10, max_value=50, value=20, step=5)

run_btn = st.sidebar.button("🔮 Run AI Forecast", type="primary", use_container_width=True)

if run_btn or "forecast_data" not in st.session_state:
    ticker_code = resolve_ticker(target_asset)
    
    if "Daily" in timeframe:
        interval, period, pred_len = "1d", "3y", 20
        step = pd.offsets.BDay(1)
        mode_str = "DAILY"
        lookback = 400
    elif "5-Minute" in timeframe:
        interval, period, pred_len = "5m", "5d", 6
        step = pd.Timedelta(minutes=5)
        mode_str = "5-MINUTE"
        lookback = 300
    else:
        interval, period, pred_len = "1h", "60d", 24
        step = pd.Timedelta(hours=1)
        mode_str = "HOURLY"
        lookback = 400

    with st.spinner(f"Fetching market data for {ticker_code} and running Kronos AI inference..."):
        df_raw = fetch_market_data(ticker_code, period=period, interval=interval)

        if df_raw.empty:
            st.error(f"Could not fetch market data for ticker '{ticker_code}'. Please check the symbol or try again.")
        else:
            df = df_raw[["timestamps", "open", "high", "low", "close", "volume"]].dropna().tail(lookback).reset_index(drop=True)

            predictor, device = load_kronos_model()

            x_ts = df["timestamps"]
            y_ts = pd.Series([x_ts.iloc[-1] + step * (i + 1) for i in range(pred_len)])

            paths = []
            for _ in range(paths_count):
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

            st.session_state["forecast_data"] = {
                "df": df,
                "x_ts": x_ts,
                "y_ts": y_ts,
                "band": band,
                "ticker_code": ticker_code,
                "mode_str": mode_str,
                "last_close": last_close,
                "next_median": next_median,
                "pct_change": pct_change,
                "device": device,
                "paths_count": paths_count
            }

if "forecast_data" in st.session_state:
    data = st.session_state["forecast_data"]

    # Metric Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"₹{data['last_close']:,.2f}")
    col2.metric("Target (50% Median)", f"₹{data['next_median']:,.2f}", f"{data['pct_change']:+.2f}%")
    col3.metric("5% Downside Risk", f"₹{data['band']['low_5%'].iloc[0]:,.2f}")
    col4.metric("95% Upside Target", f"₹{data['band']['high_95%'].iloc[0]:,.2f}")
    col5.metric("Device", data["device"].upper())

    st.markdown("---")

    # Trading Signal Alert Banner
    pct = data["pct_change"]
    if pct >= 0.5:
        st.success(f"🟢 **BULLISH SIGNAL — Consider BULL CALL SPREAD / CALL OPTION (CE)**\n\nExpected upward skew: +{pct:.2f}%. Option Call Spreads recommended to offset theta decay.")
    elif pct <= -0.5:
        st.error(f"🔴 **BEARISH SIGNAL — Consider BEAR PUT SPREAD / PUT OPTION (PE)**\n\nExpected downward skew: {pct:.2f}%. Option Put Spreads recommended to offset theta decay.")
    else:
        st.warning(f"⚪ **NEUTRAL / NO TRADE — Expected Movement (+{pct:.2f}%) within ±0.5%**\n\nAvoid buying options today to prevent theta decay losses.")

    # Chart Plot
    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_len = min(120, len(data["df"]))
    ax.plot(data["x_ts"].tail(plot_len), data["df"]["close"].tail(plot_len), color="#1e293b", linewidth=1.5, label="Historical Price")
    ax.fill_between(data["y_ts"], data["band"]["low_5%"], data["band"]["high_95%"], color="#3b82f6", alpha=0.25, label="90% Confidence Interval Range")
    ax.plot(data["y_ts"], data["band"]["likely_50%"], color="#1d4ed8", linewidth=2.5, label="Median Path (50%)")
    ax.set_title(f"{data['ticker_code']} — Kronos AI {data['mode_str']} Price Forecast ({data['paths_count']} sampled paths)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (INR)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    st.pyplot(fig)

    # Forecast Data Table
    with st.expander("📊 View Detailed Probabilistic Forecast Table"):
        display_band = data["band"].copy()
        display_band.index = [f"Step +{i+1}" for i in range(len(display_band))]
        st.dataframe(display_band.round(2), use_container_width=True)
