import os, pickle
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

MODEL_PATH = os.path.join("models", "emotion_model.pkl")
VEC_PATH = os.path.join("models", "tokenizer.pkl")

def load_model():
    """Load model and vectorizer fresh each time"""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)
    return model, vec

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

def detect_suicidal_content(text):
    """CRITICAL: Detect suicidal content with high priority"""
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # High-risk suicidal keywords and phrases
    suicidal_indicators = [
        'kill myself', 'want to die', 'want to kill myself', 'suicide', 'suicidal',
        'end my life', 'take my life', 'end it all', 'no reason to live',
        'better off dead', 'world would be better without me', 'no point in living',
        'want to end it', 'can\'t go on', 'give up', 'tired of living',
        'life is not worth living', 'death would be better', 'want to disappear',
        'hurt myself', 'self harm', 'cut myself', 'overdose', 'overdosing'
    ]
    
    # Check for suicidal content
    for indicator in suicidal_indicators:
        if indicator in text_lower:
            return True
    
    # Check for individual high-risk words
    high_risk_words = ['suicide', 'kill', 'die', 'death', 'end', 'harm']
    for word in high_risk_words:
        if word in text_lower:
            # Additional context check to avoid false positives
            if any(context in text_lower for context in ['myself', 'my life', 'want to', 'going to']):
                return True
    
    return False

def detect_depressed_content(text):
    """Detect depressed content that might be misclassified as Normal"""
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Depressed keywords and phrases
    depressed_indicators = [
        'sad', 'depressed', 'hopeless', 'hopelessness', 'empty', 'lonely', 'worthless',
        'miserable', 'unhappy', 'down', 'blue', 'melancholy', 'despair', 'desperate',
        'feel sad', 'very sad', 'so sad', 'extremely sad', 'really sad',
        'feel hopeless', 'feel empty', 'feel lonely', 'feel worthless',
        'no hope', 'lost hope', 'no motivation', 'no energy', 'tired of everything',
        'don\'t care', 'don\'t want to', 'can\'t be bothered', 'what\'s the point',
        'nothing matters', 'everything is pointless', 'life is meaningless',
        'feel like crying', 'want to cry', 'always crying', 'tears',
        'dark thoughts', 'negative thoughts', 'bad thoughts', 'intrusive thoughts',
        # Additional depressed expressions
        'want to get better', 'don\'t know how to start', 'don\'t know what to do',
        'everything bad', 'bad things happen', 'always bad', 'nothing good',
        'can\'t get better', 'stuck', 'trapped', 'no way out', 'no solution',
        'overwhelmed', 'exhausted', 'tired', 'fatigued', 'drained', 'burned out',
        'no purpose', 'no meaning', 'no direction', 'lost', 'confused',
        'feel like giving up', 'want to give up', 'ready to give up'
    ]
    
    # Check for depressed content
    for indicator in depressed_indicators:
        if indicator in text_lower:
            return True
    
    # Check for combinations of negative words
    negative_words = ['sad', 'hopeless', 'empty', 'lonely', 'worthless', 'miserable', 'unhappy', 'down', 'bad', 'terrible', 'awful', 'horrible']
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if negative_count >= 2:  # Multiple negative words indicate depression
        return True
    
    return False

def detect_stressed_content(text):
    """Detect stressed content that might be misclassified as Normal"""
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Stressed keywords and phrases
    stressed_indicators = [
        'stressed', 'stress', 'overwhelmed', 'frustrated', 'frustration', 'angry', 'anger',
        'mad', 'irritated', 'annoyed', 'hate', 'disgust', 'annoyance', 'rage', 'fury',
        'outraged', 'furious', 'livid', 'enraged', 'hostile', 'aggressive',
        # Additional stressed expressions
        'irritated', 'irritating', 'nothing is working', 'not working', 'doesn\'t work',
        'can\'t handle', 'can\'t deal with', 'too much', 'too overwhelming',
        'fed up', 'sick of', 'tired of', 'had enough', 'can\'t take it anymore',
        'everything is wrong', 'everything is bad', 'nothing goes right',
        'always problems', 'constant problems', 'never works', 'always fails',
        'pissed off', 'pissed', 'mad at', 'angry at', 'frustrated with',
        'want to scream', 'want to yell', 'want to break something',
        'can\'t stand', 'can\'t tolerate', 'unbearable', 'intolerable'
    ]
    
    # Check for stressed content
    for indicator in stressed_indicators:
        if indicator in text_lower:
            return True
    
    # Check for frustration patterns
    frustration_patterns = [
        'nothing is working', 'not working', 'doesn\'t work', 'can\'t get',
        'always bad', 'everything bad', 'never good', 'always wrong'
    ]
    
    for pattern in frustration_patterns:
        if pattern in text_lower:
            return True
    
    return False

def create_enhanced_features(text):
    """Create additional features for better classification"""
    # Text length features
    text_length = len(text)
    word_count = len(text.split())
    
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
    depressed_keyword_count = sum(1 for word in depressed_keywords if word in text.lower())
    anxious_keyword_count = sum(1 for word in anxious_keywords if word in text.lower())
    stressed_keyword_count = sum(1 for word in stressed_keywords if word in text.lower())
    normal_keyword_count = sum(1 for word in normal_keywords if word in text.lower())
    
    # Sentiment indicators
    has_negative_words = (depressed_keyword_count + anxious_keyword_count + stressed_keyword_count) > 0
    has_positive_words = normal_keyword_count > 0
    
    # Additional features
    exclamation_count = text.count('!')
    question_count = text.count('?')
    capital_letter_ratio = sum(1 for c in text if c.isupper()) / max(text_length, 1)
    
    return np.array([[
        text_length, word_count, depressed_keyword_count, anxious_keyword_count,
        stressed_keyword_count, normal_keyword_count, has_negative_words, has_positive_words,
        exclamation_count, question_count, capital_letter_ratio
    ]], dtype=float)

CONDITION_MESSAGES = {
    "Depressed": "I hear you, and it's okay to feel this way. Want me to suggest a couple tiny steps for today?",
    "Anxious": "Sounds like you've got a lot on your mind. Let's slow down—want to try a 60‑second breathing tip?",
    "Stressed": "That does sound intense. A short break and some water might help—want a quick reset idea?",
    "Normal": "You seem in a good spot right now—love to see it! Anything nice planned today?"
}

# CRITICAL: Emergency message for suicidal content
SUICIDAL_EMERGENCY_MESSAGE = """🚨 CRITICAL: I'm very concerned about what you're saying. 

If you're having thoughts of suicide, please know that:
• You are not alone
• Your life has value
• Help is available 24/7

**IMMEDIATE HELP - INDIA:**
• **Kiran Mental Health Helpline:** 1800-599-0019
• **AASRA Suicide Prevention:** 91-22-27546669
• **Vandrevala Foundation:** 1860-266-2345
• **Emergency Services:** 100 (Police) / 108 (Ambulance)

Please reach out to someone you trust or call one of these numbers right now. Your life matters."""

def predict_condition(user_text: str):
    # CRITICAL: Check for suicidal content FIRST
    if detect_suicidal_content(user_text):
        return "Depressed", 0.999  # High confidence for suicidal content
    
    # Check for depressed content that might be misclassified
    if detect_depressed_content(user_text):
        return "Depressed", 0.850  # High confidence for depressed content
    
    # Check for stressed content that might be misclassified
    if detect_stressed_content(user_text):
        return "Stressed", 0.850  # High confidence for stressed content
    
    # Load model fresh each time to avoid caching issues
    model, vec = load_model()
    
    # Preprocess text
    processed_text = preprocess_text(user_text)
    
    # Create text vector (only TF-IDF, no enhanced features for compatibility)
    text_vec = vec.transform([processed_text])
    
    # Predict
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_vec)[0]
        labels = list(model.classes_)
        # confidence is prob of predicted class
        pred_idx = proba.argmax()
        confidence = float(proba[pred_idx])
        pred_label = labels[pred_idx]
    else:
        pred_label = model.predict(text_vec)[0]
        confidence = None
    
    return pred_label, confidence

def friendly_message_for(condition: str):
    return CONDITION_MESSAGES.get(condition, CONDITION_MESSAGES["Normal"])

def get_emergency_message(user_text: str):
    """Get emergency message for suicidal content"""
    if detect_suicidal_content(user_text):
        return SUICIDAL_EMERGENCY_MESSAGE
    else:
        condition, confidence = predict_condition(user_text)
        return friendly_message_for(condition)