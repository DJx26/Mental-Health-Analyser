import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import pickle
import re
import warnings
warnings.filterwarnings('ignore')


MODEL_PATH = "models/emotion_model.pkl"

VEC_PATH = "models/tokenizer.pkl"

def advanced_preprocess_text(text):
    """Ultra-advanced text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters but keep important punctuation and emojis
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:\-\(\)\'\"]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very short texts
    if len(text) < 3:
        return ""
    
    return text

def create_ultra_features(df):
    """Create ultra-advanced features for maximum accuracy"""
    df = df.copy()
    
    # Basic text features
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_length'] = df['text'].str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
    
    # Punctuation features
    df['exclamation_count'] = df['text'].str.count('!')
    df['question_count'] = df['text'].str.count(r'\?')
    df['period_count'] = df['text'].str.count(r'\.')
    df['comma_count'] = df['text'].str.count(',')
    df['capital_letter_count'] = df['text'].str.count('[A-Z]')
    df['capital_letter_ratio'] = df['capital_letter_count'] / df['text_length'].replace(0, 1)
    
    # Advanced emotion keywords with weighted scoring
    depressed_keywords = {
        'sad': 2, 'depressed': 3, 'hopeless': 3, 'empty': 2, 'lonely': 2, 'tired': 1, 'exhausted': 2,
        'worthless': 3, 'suicide': 4, 'kill myself': 4, 'want to die': 4, 'sadness': 2, 'grief': 3,
        'disappointment': 2, 'remorse': 2, 'miserable': 3, 'unhappy': 2, 'down': 1, 'blue': 1,
        'melancholy': 2, 'despair': 3, 'desperate': 3, 'hopelessness': 3, 'sorrow': 2, 'anguish': 3
    }
    
    anxious_keywords = {
        'anxious': 3, 'anxiety': 3, 'worried': 2, 'worry': 2, 'nervous': 2, 'fear': 2, 'afraid': 2,
        'panic': 3, 'scared': 2, 'terrified': 3, 'nervousness': 2, 'stress': 1, 'tense': 2, 'jittery': 2,
        'uneasy': 2, 'apprehensive': 2, 'frightened': 2, 'alarmed': 2, 'distressed': 2, 'overwhelmed': 2,
        'paranoid': 3, 'panic attack': 4, 'heart racing': 2, 'sweating': 1
    }
    
    stressed_keywords = {
        'stressed': 3, 'stress': 2, 'overwhelmed': 2, 'frustrated': 2, 'frustration': 2, 'angry': 2,
        'anger': 2, 'mad': 1, 'irritated': 2, 'annoyed': 1, 'hate': 2, 'disgust': 2, 'annoyance': 1,
        'rage': 3, 'fury': 3, 'outraged': 3, 'furious': 3, 'livid': 3, 'enraged': 3, 'hostile': 2,
        'aggressive': 2, 'violent': 3, 'explosive': 2, 'temper': 2, 'outburst': 2
    }
    
    normal_keywords = {
        'happy': 2, 'good': 1, 'great': 1, 'excited': 2, 'love': 2, 'calm': 1, 'relaxed': 1, 'fine': 1,
        'okay': 1, 'wonderful': 2, 'amazing': 2, 'joy': 2, 'gratitude': 2, 'optimism': 2, 'pride': 2,
        'admiration': 1, 'amusement': 1, 'approval': 1, 'caring': 1, 'curiosity': 1, 'desire': 1,
        'surprise': 1, 'neutral': 1, 'pleased': 1, 'content': 1, 'satisfied': 1, 'cheerful': 2,
        'delighted': 2, 'thrilled': 2, 'blessed': 2, 'grateful': 2, 'peaceful': 1, 'serene': 1
    }
    
    # Calculate weighted keyword scores
    def calculate_weighted_score(text, keyword_dict):
        text_lower = text.lower()
        score = 0
        for keyword, weight in keyword_dict.items():
            if keyword in text_lower:
                score += weight
        return score
    
    df['depressed_score'] = df['text'].apply(lambda x: calculate_weighted_score(x, depressed_keywords))
    df['anxious_score'] = df['text'].apply(lambda x: calculate_weighted_score(x, anxious_keywords))
    df['stressed_score'] = df['text'].apply(lambda x: calculate_weighted_score(x, stressed_keywords))
    df['normal_score'] = df['text'].apply(lambda x: calculate_weighted_score(x, normal_keywords))
    
    # Sentiment indicators
    df['has_negative_words'] = (df['depressed_score'] + df['anxious_score'] + df['stressed_score']) > 0
    df['has_positive_words'] = df['normal_score'] > 0
    
    # Text complexity features
    df['unique_words'] = df['text'].str.split().apply(lambda x: len(set(x)) if x else 0)
    df['lexical_diversity'] = df['unique_words'] / df['word_count'].replace(0, 1)
    
    # Emotional intensity features
    df['emotional_intensity'] = df['exclamation_count'] + df['capital_letter_count'] * 0.5
    df['question_intensity'] = df['question_count'] * 2
    
    # Context features
    df['is_question'] = df['question_count'] > 0
    df['is_exclamation'] = df['exclamation_count'] > 0
    df['is_long_text'] = df['text_length'] > 100
    df['is_short_text'] = df['text_length'] < 20
    
    return df

def create_ultra_dataset():
    """Create ultra-high-quality dataset"""
    print("🚀 Creating Ultra-High-Quality Dataset...")
    
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
    
    # Process suicide detection data with better mapping
    suicide_df = suicide_df.dropna(subset=['text', 'class'])
    suicide_df = suicide_df[suicide_df['text'].str.len() > 15]  # Longer texts for better context
    
    def map_suicide_class_to_condition(row):
        if row['class'] == 'suicide':
            return 'Depressed'
        else:
            text = row['text'].lower()
            
            # More sophisticated keyword detection
            anxious_keywords = ['anxious', 'anxiety', 'worried', 'worry', 'nervous', 'fear', 'afraid', 'panic', 'scared', 'terrified']
            stressed_keywords = ['stressed', 'stress', 'overwhelmed', 'frustrated', 'frustration', 'angry', 'anger', 'mad', 'irritated', 'annoyed', 'hate']
            depressed_keywords = ['sad', 'depressed', 'hopeless', 'empty', 'tired', 'exhausted', 'lonely', 'worthless', 'grief']
            normal_keywords = ['happy', 'good', 'great', 'excited', 'love', 'calm', 'relaxed', 'fine', 'okay', 'wonderful', 'amazing']
            
            # Count with weights
            anxious_count = sum(2 if word in text else 0 for word in anxious_keywords)
            stressed_count = sum(2 if word in text else 0 for word in stressed_keywords)
            depressed_count = sum(2 if word in text else 0 for word in depressed_keywords)
            normal_count = sum(1 if word in text else 0 for word in normal_keywords)
            
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
    
    # Process go emotions dataset with better mapping
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
    emotions_df_clean = emotions_df_clean[emotions_df_clean['text'].str.len() > 15]
    emotions_df_clean = emotions_df_clean.dropna(subset=['text'])
    
    print(f"Go emotions processed: {len(emotions_df_clean)} rows")
    
    # Create ultra-balanced dataset
    print("\nCreating ultra-balanced dataset...")
    
    # Target: 8000 examples per condition (32000 total)
    target_per_condition = 8000
    
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
        
        # Remove duplicates and low-quality texts
        all_condition_data = all_condition_data.drop_duplicates(subset=['text'])
        all_condition_data = all_condition_data[all_condition_data['text'].str.len() > 15]
        
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
    
    print(f"\n🎉 Final ultra dataset: {len(final_df)} rows")
    print("\nFinal condition distribution:")
    print(final_df['condition'].value_counts())
    
    # Save the ultra dataset
    output_path = "data/ultra_dataset.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved ultra dataset to {output_path}")
    
    return final_df

def main():
    print("🚀 Training ULTRA Mental Health Model for 90%+ Accuracy...")
    
    # Create ultra dataset
    df = create_ultra_dataset()
    
    # Advanced preprocessing
    print("Advanced text preprocessing...")
    df['text_processed'] = df['text'].apply(advanced_preprocess_text)
    df = df[df['text_processed'].str.len() > 10]  # Remove very short texts
    
    # Create ultra features
    print("Creating ultra-advanced features...")
    df = create_ultra_features(df)
    
    # Prepare features
    X_text = df['text_processed']
    feature_columns = ['text_length', 'word_count', 'avg_word_length', 'exclamation_count', 'question_count',
                      'period_count', 'comma_count', 'capital_letter_count', 'capital_letter_ratio',
                      'depressed_score', 'anxious_score', 'stressed_score', 'normal_score',
                      'has_negative_words', 'has_positive_words', 'unique_words', 'lexical_diversity',
                      'emotional_intensity', 'question_intensity', 'is_question', 'is_exclamation',
                      'is_long_text', 'is_short_text']
    
    X_features = df[feature_columns].astype(float)
    y = df['condition']
    
    print(f"Ultra dataset size: {len(df)} examples")
    print("Condition distribution:")
    print(y.value_counts())
    
    # Split data
    X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Create ultra TF-IDF vectorizer
    print("Creating ultra TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 5),
        min_df=3,
        max_df=0.9,
        stop_words='english',
        sublinear_tf=True,
        analyzer='word'
    )
    
    # Fit and transform text features
    X_text_train_vec = vectorizer.fit_transform(X_text_train)
    X_text_test_vec = vectorizer.transform(X_text_test)
    
    # Scale features
    scaler = StandardScaler()
    X_features_train_scaled = scaler.fit_transform(X_features_train)
    X_features_test_scaled = scaler.transform(X_features_test)
    
    # Combine text and feature vectors
    from scipy.sparse import hstack, csr_matrix
    X_train_combined = hstack([X_text_train_vec, csr_matrix(X_features_train_scaled)])
    X_test_combined = hstack([X_text_test_vec, csr_matrix(X_features_test_scaled)])
    
    # Train multiple models with hyperparameter tuning
    print("Training ultra models with hyperparameter optimization...")
    
    # Define models with optimized parameters
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=3000, random_state=42, C=0.1, solver='lbfgs', class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, random_state=42, max_depth=20, min_samples_split=3,
            min_samples_leaf=2, class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=42, max_depth=8, learning_rate=0.05,
            subsample=0.8, min_samples_split=5, min_samples_leaf=2
        ),
        'SVM': SVC(
            kernel='rbf', random_state=42, probability=True, C=10, gamma='scale',
            class_weight='balanced'
        )
    }
    
    # Train and evaluate each model
    model_scores = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_combined, y_train)
        score = model.score(X_test_combined, y_test)
        model_scores[name] = score
        trained_models[name] = model
        print(f"{name} accuracy: {score:.4f}")
    
    # Create ensemble model
    print("\nCreating ensemble model...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', trained_models['Logistic Regression']),
            ('rf', trained_models['Random Forest']),
            ('gb', trained_models['Gradient Boosting']),
            ('svm', trained_models['SVM'])
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train_combined, y_train)
    ensemble_score = ensemble.score(X_test_combined, y_test)
    print(f"Ensemble accuracy: {ensemble_score:.4f}")
    
    # Find best model
    best_model_name = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model_name]
    best_model = trained_models[best_model_name]  # Default to best individual model
    
    if ensemble_score > best_score:
        best_model = ensemble
        best_score = ensemble_score
        best_model_name = "Ensemble"
    
    print(f"\n🏆 Best model: {best_model_name} with {best_score:.4f} accuracy")
    
    # Final evaluation
    y_pred = best_model.predict(X_test_combined)
    print("\nFinal Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save the best model, vectorizer, and scaler
    print("Saving ultra model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save scaler for prediction
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Saved ultra model → {MODEL_PATH}")
    print(f"✅ Saved vectorizer → {VEC_PATH}")
    print(f"✅ Saved scaler → models/scaler.pkl")
    
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
        processed_text = advanced_preprocess_text(test_text)
        text_vec = vectorizer.transform([processed_text])
        
        # Create feature vector for test case
        test_features = create_ultra_features(pd.DataFrame({'text': [test_text]}))
        feature_vec = test_features[feature_columns].values.astype(float)
        feature_vec_scaled = scaler.transform(feature_vec)
        
        # Combine vectors
        test_combined = hstack([text_vec, csr_matrix(feature_vec_scaled)])
        
        # Predict
        prediction = best_model.predict(test_combined)[0]
        confidence = max(best_model.predict_proba(test_combined)[0])
        
        print(f"'{test_text}' → {prediction} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()

