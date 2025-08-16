import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import re

DATA_PATH = "data/merged_dataset.csv"
MODEL_PATH = "models/emotion_model.pkl"
VEC_PATH = "models/tokenizer.pkl"

def preprocess_text(text):
    """Enhanced text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:\-\(\)]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_enhanced_features(df):
    """Create additional features for better classification"""
    df = df.copy()
    
    # Text length features
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    # Emotion keywords features
    depressed_keywords = ['sad', 'depressed', 'hopeless', 'empty', 'lonely', 'tired', 'exhausted', 'worthless', 'suicide', 'kill myself', 'want to die']
    anxious_keywords = ['anxious', 'anxiety', 'worried', 'worry', 'nervous', 'fear', 'afraid', 'panic', 'scared', 'terrified', 'stress']
    stressed_keywords = ['stressed', 'stress', 'overwhelmed', 'frustrated', 'frustration', 'angry', 'anger', 'mad', 'irritated', 'annoyed', 'hate']
    normal_keywords = ['happy', 'good', 'great', 'excited', 'love', 'calm', 'relaxed', 'fine', 'okay', 'wonderful', 'amazing', 'joy']
    
    # Count keyword occurrences
    df['depressed_keyword_count'] = df['text'].str.lower().apply(lambda x: sum(1 for word in depressed_keywords if word in x))
    df['anxious_keyword_count'] = df['text'].str.lower().apply(lambda x: sum(1 for word in anxious_keywords if word in x))
    df['stressed_keyword_count'] = df['text'].str.lower().apply(lambda x: sum(1 for word in stressed_keywords if word in x))
    df['normal_keyword_count'] = df['text'].str.lower().apply(lambda x: sum(1 for word in normal_keywords if word in x))
    
    # Sentiment indicators
    df['has_negative_words'] = (df['depressed_keyword_count'] + df['anxious_keyword_count'] + df['stressed_keyword_count']) > 0
    df['has_positive_words'] = df['normal_keyword_count'] > 0
    
    return df

def main():
    print("🚀 Training Enhanced Mental Health Model...")
    
    # Load and preprocess data
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH).dropna(subset=["text"])
    
    # Preprocess text
    print("Preprocessing text...")
    df['text_processed'] = df['text'].apply(preprocess_text)
    df = df[df['text_processed'].str.len() > 5]  # Remove very short texts
    
    # Create enhanced features
    print("Creating enhanced features...")
    df = create_enhanced_features(df)
    
    # Prepare features - ensure proper data types
    X_text = df['text_processed']
    X_features = df[['text_length', 'word_count', 'depressed_keyword_count', 'anxious_keyword_count', 
                     'stressed_keyword_count', 'normal_keyword_count', 'has_negative_words', 'has_positive_words']].astype(float)
    y = df['condition']
    
    print(f"Dataset size: {len(df)} examples")
    print("Condition distribution:")
    print(y.value_counts())
    
    # Split data
    X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Fit and transform text features
    X_text_train_vec = vectorizer.fit_transform(X_text_train)
    X_text_test_vec = vectorizer.transform(X_text_test)
    
    # Convert feature DataFrames to numpy arrays and ensure float type
    X_features_train_array = X_features_train.values.astype(float)
    X_features_test_array = X_features_test.values.astype(float)
    
    # Combine text and feature vectors
    from scipy.sparse import hstack, csr_matrix
    X_train_combined = hstack([X_text_train_vec, csr_matrix(X_features_train_array)])
    X_test_combined = hstack([X_text_test_vec, csr_matrix(X_features_test_array)])
    
    # Train multiple models and compare
    print("Training models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_combined, y_train)
        score = model.score(X_test_combined, y_test)
        print(f"{name} accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
    
    print(f"\n🏆 Best model: {best_score:.4f} accuracy")
    
    # Final evaluation with best model
    y_pred = best_model.predict(X_test_combined)
    print("\nFinal Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save the best model and vectorizer
    print("Saving model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"✅ Saved enhanced model → {MODEL_PATH}")
    print(f"✅ Saved vectorizer → {VEC_PATH}")
    
    # Test some specific cases
    print("\n🧪 Testing specific cases:")
    test_cases = [
        "I feel very sad and hopeless today",
        "I am so anxious about my presentation tomorrow",
        "I am feeling great and happy today",
        "Work is stressing me out so much",
        "I feel lonely and empty inside",
        "I'm worried about everything"
    ]
    
    for test_text in test_cases:
        processed_text = preprocess_text(test_text)
        text_vec = vectorizer.transform([processed_text])
        
        # Create feature vector for test case
        test_features = create_enhanced_features(pd.DataFrame({'text': [test_text]}))
        feature_vec = test_features[['text_length', 'word_count', 'depressed_keyword_count', 
                                   'anxious_keyword_count', 'stressed_keyword_count', 
                                   'normal_keyword_count', 'has_negative_words', 'has_positive_words']].values.astype(float)
        
        # Combine vectors
        test_combined = hstack([text_vec, csr_matrix(feature_vec)])
        
        # Predict
        prediction = best_model.predict(test_combined)[0]
        confidence = max(best_model.predict_proba(test_combined)[0])
        
        print(f"'{test_text}' → {prediction} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()
