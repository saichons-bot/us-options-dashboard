# =========================================================
# US STOCK + OPTIONS DASHBOARD (FINAL)
# Technical + ML (XGBoost)
# =========================================================

import math
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# =======================
# Streamlit config
# =======================
st.set_page_config(
    page_title="US Stock Options ML Dashboard",
    layout="wide"
)

# =======================
# Indicator functions
# =======================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd(series):
    macd_line = ema(series, 12) - ema(series, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =======================
# Data fetch (cached)
# =======================
@st.cache_data(ttl=600)
def load_price(ticker, interval, period):
    df = yf.Ticker(ticker).history(
        interval=interval,
        period=period,
        auto_adjust=False
    )
    return df.dropna()

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    t = yf.Ticker(ticker)
    return t.info, t.earnings_dates

@st.cache_data(ttl=900)
def load_news(ticker):
    return yf.Ticker(ticker).news[:8]

@st.cache_data(ttl=900)
def load_options(ticker):
    t = yf.Ticker(ticker)
    exps = t.options
    chains = {}
    for e in exps[:5]:
        try:
            oc = t.option_chain(e)
            calls = oc.calls.assign(type="Call")
            puts = oc.puts.assign(type="Put")
            chains[e] = pd.concat([calls, puts])
        except:
            pass
    return exps, chains

# =======================
# Feature Engineering
# =======================
def make_features(df):
    df = df.copy()
    df["ret1"] = df["Close"].pct_change()
    df["ret5"] = df["Close"].pct_change(5)
    df["ret10"] = df["Close"].pct_change(10)
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma20_ratio"] = df["Close"] / df["ma20"] - 1
    df["ma50_ratio"] = df["Close"] / df["ma50"] - 1
    df["rsi14"] = rsi(df["Close"])
    m, s, h = macd(df["Close"])
    df["macd"] = m
    df["macd_signal"] = s
    df["macd_hist"] = h
    df["atr14"] = atr(df)
    df["vol20"] = df["ret1"].rolling(20).std()
    return df.dropna()

def make_label(df, horizon):
    return (df["Close"].shift(-horizon) > df["Close"]).astype(int)

# =======================
# Train ML (XGBoost)
# =======================
@st.cache_data(ttl=3600)
def train_models(df):
    if len(df) < 300:
        return None

    feats = make_features(df)
    cols = [
        "ret1","ret5","ret10",
        "ma20_ratio","ma50_ratio",
        "rsi14","macd","macd_signal","macd_hist",
        "atr14","vol20"
    ]

    X_all = feats[cols].values
    last_row = X_all[-1:]

    results = {}
    for horizon in [1, 5, 15]:
        y = make_label(feats, horizon).iloc[:-horizon]
        X = X_all[:-horizon]

        model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X, y)
        results[horizon] = model

    return results, last_row

# =======================
# Sidebar
# =======================
st.sidebar.title("Settings")
ticker = st.sidebar.text_input(
    "US Stock Ticker",
    value="AMZN"
).upper()

# =======================
# Load data
# =======================
df_5m = load_price(ticker, "5m", "5d")
df_1h = load_price(ticker, "1h", "1mo")
df_1d = load_price(ticker, "1d", "2y")

info, earnings = load_fundamentals(ticker)
news = load_news(ticker)
exps, chains = load_options(ticker)

# =======================
# Header
# =======================
st.title(f"{ticker} — US Stock ML Options Dashboard")

last_price = df_1d["Close"].iloc[-1]
st.metric("Last Close", f"${last_price:,.2f}")

# =======================
# Charts
# =======================
c1, c2, c3 = st.columns(3)

def candle(df, title):
    fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))
    fig.update_layout(height=350, title=title, xaxis_rangeslider_visible=False)
    return fig

with c1:
    st.plotly_chart(candle(df_5m, "5 Minute"), use_container_width=True)
with c2:
    st.plotly_chart(candle(df_1h, "1 Hour"), use_container_width=True)
with c3:
    st.plotly_chart(candle(df_1d, "1 Day"), use_container_width=True)

# =======================
# ML Prediction
# =======================
st.subheader("ML Prediction (XGBoost)")

bundle = train_models(df_1d)
if bundle:
    models, last_row = bundle

    def show_prob(h):
        p = models[h].predict_proba(last_row)[0,1]
        st.metric(f"P(Up) {h}D", f"{p*100:.1f}%")
        if p >= 0.58:
            st.success("Call Bias")
        elif p <= 0.42:
            st.error("Put Bias")
        else:
            st.warning("Neutral")

    col1, col2, col3 = st.columns(3)
    with col1: show_prob(1)
    with col2: show_prob(5)
    with col3: show_prob(15)
else:
    st.warning("ข้อมูลไม่พอสำหรับ ML")

# =======================
# Options Chain
# =======================
st.subheader("Options Chain")

if exps:
    exp = st.selectbox("Expiry", exps)
    chain = chains.get(exp)
    if chain is not None:
        calls = chain[chain["type"]=="Call"].sort_values("openInterest", ascending=False).head(5)
        puts = chain[chain["type"]=="Put"].sort_values("openInterest", ascending=False).head(5)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Call OI**")
            st.dataframe(calls[["strike","lastPrice","openInterest","impliedVolatility"]])
        with c2:
            st.markdown("**Top Put OI**")
            st.dataframe(puts[["strike","lastPrice","openInterest","impliedVolatility"]])

# =======================
# News
# =======================
st.subheader("Latest News")
for n in news:
    t = dt.datetime.fromtimestamp(n["providerPublishTime"]).strftime("%Y-%m-%d")
    st.markdown(f"- **{n['title']}** ({t})")
    st.write(n["link"])
