import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

client = NewsClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

end = datetime.now(timezone.utc)
start = end - timedelta(days=180)
frames = []

while end > start:                      # ← la condition d'arrêt manquante
    req = NewsRequest(start=start, end=end, limit=50)
    batch = client.get_news(req).df
    if batch.empty:
        break
    frames.append(batch)
    print(f"  {len(batch)} titres jusqu'au {batch['created_at'].min():%Y-%m-%d %H:%M}")
    end = batch["created_at"].min().to_pydatetime() - timedelta(seconds=1)
    if len(batch) < 50:
        break

df = pd.concat(frames).drop_duplicates(subset="headline")
print(f"{len(df)} titres au total")

# 2. Garder uniquement la macro
MACRO = ["fed", "fomc", "powell", "rate cut", "rate hike", "interest rate",
         "inflation", "cpi", "pce", "jobs report", "payrolls", "unemployment",
         "jobless", "gdp", "recession", "treasury", "yields", "tariff"]

pattern = "|".join(MACRO)
macro = df[df["headline"].str.lower().str.contains(pattern)]

print(f"{len(macro)} titres macro\n")
print(macro[["created_at", "headline"]].head(20).to_string())
macro.to_csv("news_macro.csv")