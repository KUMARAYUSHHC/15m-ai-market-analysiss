# 15M AI Market Analysis Dashboard — BTC / ETH / XAUUSD

Educational research + paper-trading dashboard. It does NOT place real orders.

## Assets
- BTC/USDT
- ETH/USDT
- XAUUSD proxy: COMEX Gold futures (`GC=F`) through yfinance

## Timeframe
The signal engine is fixed to **15 minutes**.

Higher-timeframe context is optional and is NOT used to create separate trade signals.

## Features
- Market structure: HH/HL/LH/LL, BOS-style breaks, displacement
- Liquidity sweep proxies
- Equal-high/equal-low proxies
- Fair Value Gap (FVG) proxies
- Order-block proxies
- Premium/discount
- Previous-day high/low
- ATR, RSI, EMA, volume z-score
- BTC/ETH derivatives fields when a provider supplies them
- News-risk placeholder/feed layer
- Confluence score
- Probability model
- Paper-trade entry, stop, and 3 target levels
- Walk-forward-style validation and precision/coverage reporting
- No forced signal: weak setups return WAIT

## Important
A displayed probability is a model estimate, not a guarantee. The dashboard does NOT promise 70–80% live accuracy. The app reports validation metrics so you can see whether the current model actually reaches a chosen target on historical data.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run may need internet access for market-data providers.

## Data
Crypto: CCXT public OHLCV endpoints.
Gold: yfinance `GC=F` (COMEX gold futures proxy).

Provider limits, outages, delayed data, symbol differences, and missing derivatives/news data can affect results.

## Safety
This project is intentionally paper-trading/research only. It contains no broker/exchange order execution and should not be used as a promise of profit.
