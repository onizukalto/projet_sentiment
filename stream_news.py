import os
import re
import csv
from datetime import datetime
from transformers import pipeline
from alpaca.data.live.news import NewsDataStream

SEUIL_CONFIANCE = 0.8
FICHIER = "news_live.csv"

MACRO = ["fed", "fomc", "powell", "rate cut", "rate hike", "interest rate",
         "inflation", "cpi", "pce", "jobs report", "payrolls", "unemployment",
         "jobless", "gdp", "recession", "treasury", "yields", "tariff"]
PATTERN = re.compile(r"\b(" + "|".join(MACRO) + r")\b", re.IGNORECASE)

# 1. Chargement une seule fois, au démarrage
print("Chargement de FinBERT...")
finbert = pipeline("text-classification", model="ProsusAI/finbert")

stream = NewsDataStream(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

scores_du_jour = []
jour_courant = datetime.now().date()

# 2. Appelé automatiquement à chaque nouveau titre
async def on_news(news):
    global scores_du_jour, jour_courant

    titre = news.headline
    if not PATTERN.search(titre):          # pas macro → on ignore
        return

    resultat = finbert(titre, truncation=True)[0]
    label, confiance = resultat["label"], resultat["score"]

    # Sauvegarde (même colonnes que news_scored.csv)
    with open(FICHIER, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([news.created_at, titre, label, round(confiance, 2)])

    # Sentiment du jour, remis à zéro à minuit
    if datetime.now().date() != jour_courant:
        scores_du_jour, jour_courant = [], datetime.now().date()

    if confiance >= SEUIL_CONFIANCE:
        valeur = {"positive": 1, "negative": -1, "neutral": 0}[label] * confiance
        scores_du_jour.append(valeur)

    moyenne = sum(scores_du_jour) / len(scores_du_jour) if scores_du_jour else 0
    icone = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[label]
    print(f"{icone} [{news.created_at:%H:%M:%S}] {label:<8} {confiance:.2f} | {titre}")
    print(f"   → sentiment du jour : {moyenne:+.2f} ({len(scores_du_jour)} titres sûrs)\n")

print("Connecté, en attente des news macro... (Ctrl+C pour arrêter)")
stream.subscribe_news(on_news, "*")
stream.run()