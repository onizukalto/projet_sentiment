import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

SEUIL_CONFIANCE = 0.8
MIN_TITRES = 5
FENETRE = 10   # nb de jours pour le "sentiment habituel"

# 1. Titres scorés, en heure de New York, prédictions sûres uniquement
df = pd.read_csv("news_scored.csv")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("America/New_York")
df = df[df["confiance"] >= SEUIL_CONFIANCE]

# Score pondéré par la confiance : un -1 sûr à 0.99 pèse plus qu'un -1 à 0.81
df["score"] = df["sentiment_finbert"].map({"positive": 1, "negative": -1, "neutral": 0}) * df["confiance"]

# 2. Rendements du QQQ sur 1 an
qqq = yf.download("QQQ", period="1y", auto_adjust=True, progress=False)["Close"].squeeze()
rendements = qqq.pct_change().dropna()
jours_bourse = rendements.index

# 3. Rattacher chaque news au jour de bourse qu'elle peut influencer
date_effet = (df["created_at"] + pd.Timedelta(hours=8)).dt.tz_localize(None).dt.normalize()
position = jours_bourse.searchsorted(date_effet)
garde = position < len(jours_bourse)
df = df[garde]
df["jour"] = jours_bourse[position[garde]]

# 4. Sentiment par jour + sentiment relatif à la normale récente
par_jour = df.groupby("jour").agg(sentiment=("score", "mean"), nb_titres=("score", "size"))
par_jour["habituel"] = par_jour["sentiment"].rolling(FENETRE, min_periods=3).mean().shift(1)
par_jour["relatif"] = par_jour["sentiment"] - par_jour["habituel"]

# 5. Ne garder que les jours fiables, puis ajouter les rendements
par_jour = par_jour[par_jour["nb_titres"] >= MIN_TITRES]
data = par_jour.join(rendements.rename("rendement"))
data["rendement_lendemain"] = rendements.shift(-1).reindex(data.index)
data = data.dropna()

print(data.round(4).tail(15).to_string(), "\n")
print(f"{len(data)} jours analysés (≥ {MIN_TITRES} titres)\n")

# 6. Corrélations + p-value
for signal in ["sentiment", "relatif"]:
    for cible in ["rendement", "rendement_lendemain"]:
        r, p = pearsonr(data[signal], data[cible])
        verdict = "significatif" if p < 0.05 else "peut être du hasard"
        print(f"{signal:>9} vs {cible:<20} r = {r:+.2f}   p = {p:.3f}   → {verdict}")