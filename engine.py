import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

TIMEFRAME = "15m"

def fetch_data(asset, limit=1500):
    if asset in ["BTC/USDT","ETH/USDT"]:
        ex = ccxt.binance({"enableRateLimit": True})
        raw = ex.fetch_ohlcv(asset, timeframe=TIMEFRAME, limit=min(limit, 1000))
        # Fetch a second page when practical.
        if limit > 1000:
            since = raw[0][0] - 1000*15*60*1000
            try:
                older = ex.fetch_ohlcv(asset, timeframe=TIMEFRAME, since=since, limit=1000)
                raw = older + raw
            except Exception:
                pass
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    if asset == "XAUUSD":
        import yfinance as yf
        df = yf.download("GC=F", period="60d", interval="15m", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        return df[["open","high","low","close","volume"]].dropna()
    raise ValueError("Unsupported asset")

def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0,np.nan)
    return 100 - 100/(1+rs)

def build_features(df):
    x=df.copy()
    x["ret"]=x.close.pct_change()
    x["range"]=x.high-x.low
    x["body"]=(x.close-x.open).abs()
    x["upper_wick"]=x.high-x[["open","close"]].max(axis=1)
    x["lower_wick"]=x[["open","close"]].min(axis=1)-x.low
    tr=pd.concat([(x.high-x.low),(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
    x["atr"]=tr.rolling(14).mean()
    x["rsi"]=_rsi(x.close)
    x["ema_fast"]=x.close.ewm(span=20,adjust=False).mean()
    x["ema_slow"]=x.close.ewm(span=50,adjust=False).mean()
    x["vol_z"]=(x.volume-x.volume.rolling(30).mean())/x.volume.rolling(30).std()
    x["range_z"]=(x.range-x.range.rolling(30).mean())/x.range.rolling(30).std()
    x["swing_hi"]=x.high.rolling(10).max().shift(1)
    x["swing_lo"]=x.low.rolling(10).min().shift(1)
    x["bos_up"]=x.close>x.swing_hi
    x["bos_down"]=x.close<x.swing_lo
    x["sweep_high"]=(x.high>x.swing_hi)&(x.close<x.swing_hi)
    x["sweep_low"]=(x.low<x.swing_lo)&(x.close>x.swing_lo)
    # 3-candle FVG-style proxies
    x["fvg_up"]=x.low>x.high.shift(2)
    x["fvg_down"]=x.high<x.low.shift(2)
    x["displacement"]=(x.body > 1.5*x.atr) & (x.vol_z>0.5)
    x["ob_bull"]=x.close.shift(1)<x.open.shift(1)
    x["ob_bear"]=x.close.shift(1)>x.open.shift(1)
    x["pd_mid"]=(x.high.rolling(50).max()+x.low.rolling(50).min())/2
    x["premium"]=x.close>x.pd_mid
    # Previous-day levels from UTC date as a consistent research proxy.
    day=x.index.floor("D")
    dh=x.high.groupby(day).transform("max")
    dl=x.low.groupby(day).transform("min")
    x["day_high_prev"]=dh.groupby(day).transform("first").shift(1)
    x["day_low_prev"]=dl.groupby(day).transform("first").shift(1)
    # Clean booleans
    for c in ["bos_up","bos_down","sweep_high","sweep_low","fvg_up","fvg_down","displacement","ob_bull","ob_bear","premium"]:
        x[c]=x[c].fillna(False).astype(int)
    return x.replace([np.inf,-np.inf],np.nan).dropna()

FEATURES=["ret","range","body","upper_wick","lower_wick","atr","rsi","ema_fast","ema_slow","vol_z","range_z",
          "bos_up","bos_down","sweep_high","sweep_low","fvg_up","fvg_down","displacement","ob_bull","ob_bear","premium"]

def train_model(df):
    d=df.copy()
    # Next 4 x 15M candles (~1 hour) direction.
    future=d.close.shift(-4)/d.close-1
    # Volatility-aware target; not a fixed "profit guarantee".
    threshold=(d.atr/d.close)*0.20
    y=(future>threshold).astype(int)
    data=pd.concat([d[FEATURES],y.rename("y")],axis=1).dropna()
    split=int(len(data)*0.8)
    tr,te=data.iloc[:split],data.iloc[split:]
    model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=0.06,l2_regularization=1.0,random_state=7)
    model.fit(tr[FEATURES],tr.y)
    p=model.predict_proba(te[FEATURES])[:,1]
    pred=(p>=0.5).astype(int)
    acc=accuracy_score(te.y,pred)
    try: auc=roc_auc_score(te.y,p)
    except Exception: auc=float("nan")
    # Coverage above 70% confidence; shows whether high-confidence calls are selective.
    conf=np.maximum(p,1-p)
    mask=conf>=0.70
    coverage=float(mask.mean())
    high_conf_acc=float(accuracy_score(te.y[mask],pred[mask])) if mask.any() else float("nan")
    return {"model":model,"accuracy":float(acc),"roc_auc":float(auc),"coverage":coverage,"high_conf_accuracy":high_conf_acc}

def make_signal(df, info, min_conf=0.70):
    row=df.iloc[-1]
    p=float(info["model"].predict_proba(df[FEATURES].tail(1))[0,1])
    direction="BULLISH" if p>=0.5 else "BEARISH"
    conf=max(p,1-p)
    confluence=0
    confluence += 15 if ((row["bos_up"] and direction=="BULLISH") or (row["bos_down"] and direction=="BEARISH")) else 0
    confluence += 15 if ((row["sweep_low"] and direction=="BULLISH") or (row["sweep_high"] and direction=="BEARISH")) else 0
    confluence += 15 if ((row["fvg_up"] and direction=="BULLISH") or (row["fvg_down"] and direction=="BEARISH")) else 0
    confluence += 15 if row["displacement"] else 0
    confluence += 15 if ((row["ema_fast"]>row["ema_slow"] and direction=="BULLISH") or (row["ema_fast"]<row["ema_slow"] and direction=="BEARISH")) else 0
    confluence += 10 if row["vol_z"]>0.5 else 0
    confluence += 15 if (row["rsi"]<70 if direction=="BULLISH" else row["rsi"]>30) else 0
    confluence=int(min(100,confluence))
    price=float(row.close); atr=float(row.atr)
    if conf < min_conf or confluence < 55:
        return {"direction":"WAIT","confidence":conf,"confluence":confluence,
                "reason":"No sufficiently strong 15M setup. Waiting is preferable to forcing a prediction."}
    if direction=="BULLISH":
        entry=price
        stop=min(float(row.low)-0.15*atr, price-1.0*atr)
        risk=entry-stop
        tp1=entry+1.0*risk; tp2=entry+1.8*risk; tp3=entry+2.6*risk
        invalidation=stop
        structure="Bullish structure / momentum"
        liquidity="Low-side sweep proxy" if row.sweep_low else "No strong sweep"
        fvg="Bullish FVG proxy" if row.fvg_up else "None"
        ob="Bullish OB proxy" if row.ob_bull else "None"
    else:
        entry=price
        stop=max(float(row.high)+0.15*atr, price+1.0*atr)
        risk=stop-entry
        tp1=entry-1.0*risk; tp2=entry-1.8*risk; tp3=entry-2.6*risk
        invalidation=stop
        structure="Bearish structure / momentum"
        liquidity="High-side sweep proxy" if row.sweep_high else "No strong sweep"
        fvg="Bearish FVG proxy" if row.fvg_down else "None"
        ob="Bearish OB proxy" if row.ob_bear else "None"
    return {"direction":direction,"confidence":conf,"confluence":confluence,
            "entry":entry,"stop":stop,"tp1":tp1,"tp2":tp2,"tp3":tp3,
            "rr1":1.0,"rr2":1.8,"rr3":2.6,"invalidation":invalidation,
            "structure":structure,"liquidity":liquidity,"fvg":fvg,
            "order_block":ob,"displacement":"Strong" if row.displacement else "Normal",
            "momentum":"Bullish" if row.ema_fast>row.ema_slow else "Bearish",
            "volume":"High" if row.vol_z>0.5 else "Normal"}
