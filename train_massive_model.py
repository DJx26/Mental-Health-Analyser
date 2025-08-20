import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
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
    
    # Enhanced emotion keywords with more comprehensive lists
    depressed_keywords = ['sad', 'depressed', 'hopeless', 'empty', 'lonely', 'tired', 'exhausted', 'worthless', 
                         'suicide', 'kill myself', 'want to die', 'sadness', 'grief', 'disappointment', 'remorse',
                         'miserable', 'unhappy', 'down', 'blue', 'melancholy', 'despair', 'desperate']
    
    anxious_keywords = ['anxious', 'anxiety', 'worried', 'worry', 'nervous', 'fear', 'afraid', 'panic', 
                       'scared', 'terrified', 'nervousness', 'stress', 'tense', 'jittery', 'uneasy',
                       'apprehensive', 'frightened', 'alarmed', 'distressed']
    
    stressed_keywords = ['stressed', 'stress', 'overwhelmed', 'frustrated', 'frustration', 'angry', 'anger', 
                        'mad', 'irritated', 'annoyed', 'hate', 'disgust', 'annoyance', 'rage', 'fury',
                        'outraged', 'furious', 'livid', 'enraged', 'hostile', 'aggressive']
    
    normal_keywords = ['happy', 'good', 'great', 'excited', 'love', 'calm', 'relaxed', 'fine', 'okay', 
                      'wonderful', 'amazing', 'joy', 'gratitude', 'optimism', 'pride', 'admiration',
                      'amusement', 'approval', 'caring', 'curiosity', 'desire', 'surprise', 'neutral',
                      'pleased', 'content', 'satisfied', 'cheerful', 'delighted', 'thrilled']
    
    # Count keyword occurrences
    df['depressed_keyword_count'] = df['text'].str.lower().apply(lambda x: sum(1 for word in depressed_keywords if word in x))
    df['anxious_keyword_count'] = df['text'].str.lower().apply(lambda x: sum(1 for word in anxious_keywords if word in x))
    df['stressed_keyword_count'] = df['text'].str.lower().apply(lambda x: sum(1 for word in stressed_keywords if word in x))
    df['normal_keyword_count'] = df['text'].str.lower().apply(lambda x: sum(1 for word in normal_keywords if word in x))
    
    # Sentiment indicators
    df['has_negative_words'] = (df['depressed_keyword_count'] + df['anxious_keyword_count'] + df['stressed_keyword_count']) > 0
    df['has_positive_words'] = df['normal_keyword_count'] > 0
    
    # Additional features
    df['exclamation_count'] = df['text'].str.count('!')
    df['question_count'] = df['text'].str.count('\?')
    df['capital_letter_ratio'] = df['text'].str.count('[A-Z]') / df['text_length'].replace(0, 1)
    
    return df

def create_massive_dataset():
    """Create a massive dataset using all available data"""
    print("🚀 Creating Massive Dataset...")
    
    # Load all datasets
    print("Loading all datasets...")
    
    # Load existing merged dataset
    existing_df = pd.read_csv("data/merged_dataset.csv")
    print(f"Existing dataset: {len(existing_df)} rows")
    
    # Load suicide detection dataset
    suicide_df = pd.read_csv("data/Suicide_Detection.csv")
    print(f"Suicide detection dataset: {len(suicide_df)} rows")
    
    # Load go emotions dataset
    emotions_df = pd.read_csv("data/go_emotions_dataset.csv")
    print(f"Go emotions dataset: {len(emotions_df)} rows")
    
    # Process suicide detection data
    suicide_df = suicide_df.dropna(subset=['text', 'class'])
    suicide_df = suicide_df[suicide_df['text'].str.len() > 10]
    
    def map_suicide_class_to_condition(row):
        if row['class'] == 'suicide':
            return 'Depressed'
        else:
            text = row['text'].lower()
            anxious_keywords = ['anxious', 'anxiety', 'worried', 'worry', 'nervous', 'fear', 'afraid', 'panic']
            stressed_keywords = ['stressed', 'stress', 'overwhelmed', 'frustrated', 'frustration', 'angry', 'anger']
            depressed_keywords = ['sad', 'depressed', 'hopeless', 'empty', 'tired', 'exhausted', 'lonely']
            normal_keywords = ['happy', 'good', 'great', 'excited', 'love', 'calm', 'relaxed', 'fine', 'okay']
            
            anxious_count = sum(1 for word in anxious_keywords if word in text)
            stressed_count = sum(1 for word in stressed_keywords if word in text)
            depressed_count = sum(1 for word in depressed_keywords if word in text)
            normal_count = sum(1 for word in normal_keywords if word in text)
            
            if depressed_count > 0:
                return 'Depressed'
            elif anxious_count > 0:
                return 'Anxious'
            elif stressed_count > 0:
                return 'Stressed'
            elif normal_count > 0:
                return 'Normal'
            else:
                return 'Normal'
    
    suicide_df['condition'] = suicide_df.apply(map_suicide_class_to_condition, axis=1)
    suicide_df_clean = suicide_df[['text', 'condition']].copy()
    
    # Process go emotions dataset with more aggressive sampling
    print("Processing go emotions dataset...")
    
    emotion_to_condition = {
        'sadness': 'Depressed',
        'grief': 'Depressed',
        'disappointment': 'Depressed',
        'remorse': 'Depressed',
        'fear': 'Anxious',
        'nervousness': 'Anxious',
        'anger': 'Stressed',
        'annoyance': 'Stressed',
        'disgust': 'Stressed',
        'disapproval': 'Stressed',
        'joy': 'Normal',
        'love': 'Normal',
        'excitement': 'Normal',
        'gratitude': 'Normal',
        'optimism': 'Normal',
        'pride': 'Normal',
        'relief': 'Normal',
        'admiration': 'Normal',
        'amusement': 'Normal',
        'approval': 'Normal',
        'caring': 'Normal',
        'curiosity': 'Normal',
        'desire': 'Normal',
        'surprise': 'Normal',
        'neutral': 'Normal',
        'confusion': 'Normal',
        'embarrassment': 'Normal',
        'realization': 'Normal'
    }
    
    def map_emotions_to_condition(row):
        emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                          'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                          'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                          'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                          'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        
        for emotion in emotion_columns:
            if row[emotion] == 1:
                return emotion_to_condition.get(emotion, 'Normal')
        return 'Normal'
    
    emotions_df['condition'] = emotions_df.apply(map_emotions_to_condition, axis=1)
    emotions_df_clean = emotions_df[['text', 'condition']].copy()
    emotions_df_clean = emotions_df_clean[emotions_df_clean['text'].str.len() > 10]
    emotions_df_clean = emotions_df_clean.dropna(subset=['text'])
    
    print(f"Go emotions processed: {len(emotions_df_clean)} rows")
    print("Go emotions condition distribution:")
    print(emotions_df_clean['condition'].value_counts())
    
    # Create massive balanced dataset - much larger this time
    print("\nCreating massive balanced dataset...")
    
    # Target: 5000 examples per condition (20000 total)
    target_per_condition = 5000
    
    conditions = ['Depressed', 'Anxious', 'Stressed', 'Normal']
    all_sampled_dfs = []
    
    for condition in conditions:
        print(f"\nProcessing {condition} condition:")
        
        # Get data from each source for this condition
        existing_condition = existing_df[existing_df['condition'] == condition]
        suicide_condition = suicide_df_clean[suicide_df_clean['condition'] == condition]
        emotions_condition = emotions_df_clean[emotions_df_clean['condition'] == condition]
        
        print(f"  - Existing: {len(existing_condition)} examples")
        print(f"  - Suicide detection: {len(suicide_condition)} examples")
        print(f"  - Go emotions: {len(emotions_condition)} examples")
        
        # Combine all sources for this condition
        all_condition_data = pd.concat([existing_condition, suicide_condition, emotions_condition], ignore_index=True)
        
        # Remove duplicates based on text
        all_condition_data = all_condition_data.drop_duplicates(subset=['text'])
        
        print(f"  - Total unique: {len(all_condition_data)} examples")
        
        # Sample to target size
        if len(all_condition_data) > target_per_condition:
            sampled = all_condition_data.sample(n=target_per_condition, random_state=42)
        else:
            sampled = all_condition_data
        
        all_sampled_dfs.append(sampled)
        print(f"  - Final sampled: {len(sampled)} examples")
    
    # Combine all sampled data
    final_df = pd.concat(all_sampled_dfs, ignore_index=True)
    
    # Shuffle the data
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n🎉 Final massive dataset: {len(final_df)} rows")
    print("\nFinal condition distribution:")
    print(final_df['condition'].value_counts())
    
    # Save the massive dataset
    output_path = "data/massive_dataset.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved massive dataset to {output_path}")
    
    return final_df

def main():
    print("🚀 Training MASSIVE Mental Health Model...")
    
    # Create massive dataset
    df = create_massive_dataset()
    
    # Load and preprocess data
    print("Preprocessing text...")
    df['text_processed'] = df['text'].apply(preprocess_text)
    df = df[df['text_processed'].str.len() > 5]  # Remove very short texts
    
    # Create enhanced features
    print("Creating enhanced features...")
    df = create_enhanced_features(df)
    
    # Prepare features - ensure proper data types
    X_text = df['text_processed']
    X_features = df[['text_length', 'word_count', 'depressed_keyword_count', 'anxious_keyword_count', 
                     'stressed_keyword_count', 'normal_keyword_count', 'has_negative_words', 'has_positive_words',
                     'exclamation_count', 'question_count', 'capital_letter_ratio']].astype(float)
    y = df['condition']
    
    print(f"Massive dataset size: {len(df)} examples")
    print("Condition distribution:")
    print(y.value_counts())
    
    # Split data
    X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create enhanced TF-IDF vectorizer
    print("Creating enhanced TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True
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
    print("Training multiple models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, C=1.0, solver='lbfgs'),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, min_samples_split=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True, C=1.0, gamma='scale')
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
    
    print(f"✅ Saved massive model → {MODEL_PATH}")
    print(f"✅ Saved vectorizer → {VEC_PATH}")
    
    # Test specific cases
    print("\n🧪 Testing specific cases:")
    test_cases = [
        "I feel very sad and hopeless today",
        "I am so anxious about my presentation tomorrow",
        "I am feeling great and happy today",
        "Work is stressing me out so much",
        "I feel lonely and empty inside",
        "I'm worried about everything",
        "I'm so frustrated with my job",
        "Today was amazing and I feel wonderful"
    ]
    
    for test_text in test_cases:
        processed_text = preprocess_text(test_text)
        text_vec = vectorizer.transform([processed_text])
        
        # Create feature vector for test case
        test_features = create_enhanced_features(pd.DataFrame({'text': [test_text]}))
        feature_vec = test_features[['text_length', 'word_count', 'depressed_keyword_count', 
                                   'anxious_keyword_count', 'stressed_keyword_count', 
                                   'normal_keyword_count', 'has_negative_words', 'has_positive_words',
                                   'exclamation_count', 'question_count', 'capital_letter_ratio']].values.astype(float)
        
        # Combine vectors
        test_combined = hstack([text_vec, csr_matrix(feature_vec)])
        
        # Predict
        prediction = best_model.predict(test_combined)[0]
        confidence = max(best_model.predict_proba(test_combined)[0])
        
        print(f"'{test_text}' → {prediction} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()
