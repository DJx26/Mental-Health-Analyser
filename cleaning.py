import pandas as pd
import re

GO_PATH = "data/goemotions.csv"
REDDIT_PATH = "data/reddit_mental_health.csv"
OUT_PATH = "data/merged_dataset.csv"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def main():
    # Expect CSVs with at least columns: text, emotion/label
    
    go = pd.read_csv(GO_PATH)
    rd = pd.read_csv(REDDIT_PATH)

    go = go.rename(columns={"label":"emotion"})
    rd = rd.rename(columns={"label":"emotion"})

    go["text"] = go["text"].astype(str).apply(clean_text)
    rd["text"] = rd["text"].astype(str).apply(clean_text)

    go = go[["text","emotion"]].dropna()
    rd = rd[["text","emotion"]].dropna()

    merged = pd.concat([go, rd], ignore_index=True)
    merged = merged.dropna().drop_duplicates()
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved merged dataset → {OUT_PATH} | rows={len(merged)}")

if __name__ == "__main__":
    main()
