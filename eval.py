import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dictionnaire import score_dictionnaire

# 1. Le dataset annoté
ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
df = ds.to_pandas()
df["vrai"] = df["label"].map({0: "negative", 1: "positive", 2: "neutral"})
print(f"{len(df)} phrases annotées\n")

# 2. Les deux méthodes
df["dico"] = df["text"].apply(score_dictionnaire)

finbert = pipeline("text-classification", model="ProsusAI/finbert")
res = finbert(df["text"].tolist(), batch_size=32, truncation=True)
df["finbert"] = [r["label"] for r in res]

# 3. Les notes
labels = ["negative", "neutral", "positive"]
for methode in ["dico", "finbert"]:
    print(f"===== {methode.upper()} =====")
    print(f"Précision globale : {accuracy_score(df['vrai'], df[methode]):.1%}\n")
    print(classification_report(df["vrai"], df[methode], labels=labels, zero_division=0))
    cm = confusion_matrix(df["vrai"], df[methode], labels=labels)
    print(pd.DataFrame(cm,
          index=[f"vrai {l}" for l in labels],
          columns=[f"prédit {l}" for l in labels]), "\n")