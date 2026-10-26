import os
from datetime import time, timedelta
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

HORIZONS = [5, 15, 30]   # minutes après la news

# 1. Titres sûrs, publiés pendant la séance (9h30–15h30 à New York)
df = pd.read_csv("news_scored.csv")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
df = df[df["confiance"] >= 0.8]
df["score"] = df["sentiment_finbert"].map({"positive": 1, "negative": -1, "neutral": 0}) * df["confiance"]

ny = df["created_at"].dt.tz_convert("America/New_York")
en_seance = (ny.dt.weekday < 5) & (ny.dt.time >= time(9, 30)) & (ny.dt.time <= time(15, 30))
df = df[en_seance].sort_values("created_at")
print(f"{len(df)} titres publiés pendant la séance")

# 2. Prix minute par minute du QQQ (flux IEX, gratuit)
client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
req = StockBarsRequest(
    symbol_or_symbols="QQQ",
    timeframe=TimeFrame.Minute,
    start=df["created_at"].min() - timedelta(hours=1),
    end=df["created_at"].max() + timedelta(hours=1),
    feed=DataFeed.IEX,
)
prix = client.get_stock_bars(req).df.xs("QQQ")["close"]
print(f"{len(prix)} bougies minute téléchargées\n")

# 3. Prix juste avant la news, puis X minutes après
def prix_a(instants):
    pos = prix.index.searchsorted(instants, side="right") - 1
    return np.where(pos >= 0, prix.values[pos.clip(0)], np.nan)

p0 = prix_a(df["created_at"] - timedelta(minutes=1))
for h in HORIZONS:
    df[f"ret_{h}min"] = prix_a(df["created_at"] + timedelta(minutes=h)) / p0 - 1

# 4. Résultats
cols = [f"ret_{h}min" for h in HORIZONS]
print("Rendement moyen après la news (en points de base, 1 pb = 0,01 %) :")
print((df.groupby("sentiment_finbert")[cols].mean() * 10000).round(1))
print(df["sentiment_finbert"].value_counts().to_string(), "\n")

for c in cols:
    d = df[["score", c]].dropna()
    r, p = pearsonr(d["score"], d[c])
    verdict = "significatif" if p < 0.05 else "peut être du hasard"
    print(f"score vs {c:<9} r = {r:+.2f}   p = {p:.3f}   → {verdict}")