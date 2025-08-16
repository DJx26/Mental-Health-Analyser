import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

DATA_PATH = "data/merged_dataset.csv"
MODEL_PATH = "models/emotion_model.pkl"
VEC_PATH = "models/tokenizer.pkl"

# Map granular emotions → coarse mental health condition
EMOTION_TO_CONDITION = {
    "sad": "Depressed",
    "sadness": "Depressed",
    "grief": "Depressed",
    "hopeless": "Depressed",
    "hopelessness": "Depressed",
    "down": "Depressed",
    "joy": "Normal",
    "happy": "Normal",
    "neutral": "Normal",
    "calm": "Normal",
    "anger": "Stressed",
    "angry": "Stressed",
    "frustration": "Stressed",
    "frustrated": "Stressed",
    "disgust": "Stressed",
    "overwhelmed": "Stressed",
    "fear": "Anxious",
    "afraid": "Anxious",
    "worry": "Anxious",
    "worried": "Anxious",
    "nervous": "Anxious",
    "anxiety": "Anxious",
    "anxious": "Anxious",
    "surprise": "Normal",
    "love": "Normal"
}

def emotion_to_condition_label(emotion: str) -> str:
    if not isinstance(emotion, str):
        return "Normal"
    e = emotion.strip().lower()
    # direct
    if e in EMOTION_TO_CONDITION:
        return EMOTION_TO_CONDITION[e]
    # keyword contains
    for k, v in EMOTION_TO_CONDITION.items():
        if k in e:
            return v
    return "Normal"

def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["text"])
    # If dataset already has "condition", prefer it. Otherwise map from "emotion".
    if "condition" not in df.columns:
        if "emotion" not in df.columns:
            raise ValueError("Dataset must have either 'condition' or 'emotion' column.")
        df["condition"] = df["emotion"].apply(emotion_to_condition_label)

    X = df["text"].astype(str)
    y = df["condition"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(max_iter=400, n_jobs=None)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("Model Performance:\n")
    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vec, f)
    print(f"✅ Saved model → {MODEL_PATH}")
    print(f"✅ Saved vectorizer → {VEC_PATH}")

if __name__ == "__main__":
    main()