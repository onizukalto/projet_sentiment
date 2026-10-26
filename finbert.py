import pandas as pd
from transformers import pipeline
from dictionnaire import score_dictionnaire

# Charge le modèle (≈400 Mo, téléchargé une seule fois)
finbert = pipeline("text-classification", model="ProsusAI/finbert")

df = pd.read_csv("news_macro.csv")
titres = df["headline"].tolist()

# FinBERT traite tous les titres d'un coup
resultats = finbert(titres, batch_size=16, truncation=True)

df["sentiment_finbert"] = [r["label"] for r in resultats]
df["confiance"] = [round(r["score"], 2) for r in resultats]
df["sentiment_dico"] = df["headline"].apply(score_dictionnaire)

print(df["sentiment_finbert"].value_counts(), "\n")

# Les titres où les deux méthodes ne sont pas d'accord
desaccords = df[df["sentiment_dico"] != df["sentiment_finbert"]]
print(f"{len(desaccords)} désaccords sur {len(df)} titres\n")
pd.set_option("display.max_colwidth", 100)
print(desaccords[["headline", "sentiment_dico", "sentiment_finbert", "confiance"]]
      .head(20).to_string())

df.to_csv("news_scored.csv", index=False)