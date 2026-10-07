import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from engine import fetch_data, build_features, train_model, make_signal
from news import get_news_risk

st.set_page_config(page_title="15M AI Market Analysis", layout="wide")
st.title("15M AI Market Analysis — BTC • ETH • XAUUSD")
st.caption("Research / paper-trading dashboard. No real-money order execution.")

asset = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "XAUUSD"])
bars = st.slider("Historical 15M candles", 500, 5000, 1500, 100)
min_conf = st.slider("Minimum model confidence", 0.55, 0.90, 0.70, 0.01)

if st.button("Refresh 15M analysis"):
    st.cache_data.clear()

try:
    df = fetch_data(asset, bars)
    df = build_features(df)
    model_info = train_model(df)
    signal = make_signal(df, model_info, min_conf=min_conf)
    news = get_news_risk(asset)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("15M Model", signal["direction"])
    c2.metric("Confidence", f'{signal["confidence"]*100:.1f}%')
    c3.metric("Confluence", f'{signal["confluence"]}/100')
    c4.metric("News risk", news["risk"])

    st.subheader("Paper-trade scenario")
    if signal["direction"] == "WAIT":
        st.info(signal["reason"])
    else:
        a,b,c,d,e = st.columns(5)
        a.metric("Entry", f'{signal["entry"]:.4f}')
        b.metric("Stop-loss", f'{signal["stop"]:.4f}')
        c.metric("Target 1", f'{signal["tp1"]:.4f}')
        d.metric("Target 2", f'{signal["tp2"]:.4f}')
        e.metric("Target 3", f'{signal["tp3"]:.4f}')
        st.write(f'Risk/Reward to T1: **{signal["rr1"]:.2f}** | T2: **{signal["rr2"]:.2f}** | T3: **{signal["rr3"]:.2f}**')
        st.write(f'Invalidation: **{signal["invalidation"]:.4f}**')

    st.subheader("15M chart")
    tail = df.tail(250)
    fig = go.Figure(data=[go.Candlestick(
        x=tail.index, open=tail["open"], high=tail["high"],
        low=tail["low"], close=tail["close"], name="15M"
    )])
    if signal["direction"] != "WAIT":
        fig.add_hline(y=signal["entry"], annotation_text="Entry")
        fig.add_hline(y=signal["stop"], annotation_text="SL")
        for i, tp in enumerate([signal["tp1"],signal["tp2"],signal["tp3"]],1):
            fig.add_hline(y=tp, annotation_text=f"TP{i}")
    fig.update_layout(height=650, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why the model reached this result")
    st.json({
        "structure": signal["structure"],
        "liquidity": signal["liquidity"],
        "fvg": signal["fvg"],
        "order_block_proxy": signal["order_block"],
        "displacement": signal["displacement"],
        "momentum": signal["momentum"],
        "volume": signal["volume"],
        "validation_accuracy": model_info["accuracy"],
        "validation_roc_auc": model_info["roc_auc"],
        "validation_coverage": model_info["coverage"],
        "note": "70–80% is a target to test on historical data, not a guaranteed live accuracy."
    })

    st.subheader("Recent 15M data")
    st.dataframe(df.tail(20)[["open","high","low","close","volume","rsi","atr","ema_fast","ema_slow","bos_up","bos_down","sweep_high","sweep_low","fvg_up","fvg_down"]])

except Exception as e:
    st.error(f"Data/model error: {e}")
    st.info("Check internet access and provider availability, then refresh.")
