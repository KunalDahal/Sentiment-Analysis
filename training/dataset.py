import pandas as pd
from model.preprocess import combine_text

def load_dataset(path):
    df = pd.read_csv(path)

    df["final_text"] = df.apply(
        lambda r: combine_text(r["Summary"], r["Review"]),
        axis=1
    )

    df["label"] = df["Sentiment"].map({"positive": 1, "negative": 0})
    df = df.dropna(subset=["final_text", "label"])

    return df