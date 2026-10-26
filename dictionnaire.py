import re
import pandas as pd

POSITIFS = ["rise", "rises", "gain", "gains", "growth", "strong", "beat", "beats",
            "surge", "rally", "boost", "optimism", "recovery", "cut", "cuts", "eases"]

NEGATIFS = ["fall", "falls", "drop", "drops", "decline", "weak", "miss", "misses",
            "recession", "crisis", "fear", "fears", "slump", "risk", "concern",
            "hike", "hikes", "inflation", "layoffs", "tariff", "tariffs"]

def score_dictionnaire(titre):
    mots = re.findall(r"[a-z]+", titre.lower())   # découpe en mots
    pos = sum(m in POSITIFS for m in mots)
    neg = sum(m in NEGATIFS for m in mots)
    score = pos - neg
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"

df = pd.read_csv("news_macro.csv")
df["sentiment_dico"] = df["headline"].apply(score_dictionnaire)

print(df["sentiment_dico"].value_counts(), "\n")
pd.set_option("display.max_colwidth", 120)
print(df[["headline", "sentiment_dico"]].head(25).to_string())

if __name__ == "__main__":
    df = pd.read_csv("news_macro.csv")
    # ... (le reste, indenté)