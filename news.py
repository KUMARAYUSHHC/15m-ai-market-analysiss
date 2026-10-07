import feedparser
from datetime import datetime, timezone

FEEDS = [
    "https://feeds.feedburner.com/CoinDesk",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://feeds.reuters.com/reuters/businessNews"
]

def get_news_risk(asset):
    terms = {
        "BTC/USDT":["bitcoin","crypto","btc","etf","fed","rates","inflation","regulation"],
        "ETH/USDT":["ethereum","eth","crypto","etf","fed","rates","inflation","regulation"],
        "XAUUSD":["gold","fed","federal reserve","treasury","yield","inflation","jobs","dollar"]
    }[asset]
    headlines=[]
    for url in FEEDS:
        try:
            feed=feedparser.parse(url)
            for e in feed.entries[:15]:
                title=e.get("title","")
                low=title.lower()
                if any(t in low for t in terms):
                    headlines.append(title)
        except Exception:
            pass
    risk="HIGH" if len(headlines)>=8 else "MEDIUM" if len(headlines)>=3 else "LOW"
    return {"risk":risk,"headlines":headlines[:8]}
