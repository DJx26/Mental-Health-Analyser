import pandas as pd
import numpy as np

def merge_all_datasets():
    print("🚀 Starting comprehensive dataset merge...")
    
    # Load existing merged dataset
    print("Loading existing merged dataset...")
    existing_df = pd.read_csv("data/merged_dataset.csv")
    print(f"Existing dataset: {len(existing_df)} rows")
    
    # Load suicide detection dataset
    print("Loading suicide detection dataset...")
    suicide_df = pd.read_csv("data/Suicide_Detection.csv")
    print(f"Suicide detection dataset: {len(suicide_df)} rows")
    
    # Load go emotions dataset
    print("Loading go emotions dataset...")
    emotions_df = pd.read_csv("data/go_emotions_dataset.csv")
    print(f"Go emotions dataset: {len(emotions_df)} rows")
    
    # Clean suicide detection data
    suicide_df = suicide_df.dropna(subset=['text', 'class'])
    suicide_df = suicide_df[suicide_df['text'].str.len() > 10]
    
    # Map suicide detection classes
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
    
    # Process go emotions dataset
    print("Processing go emotions dataset...")
    
    # Define emotion to condition mapping
    emotion_to_condition = {
        # Depressed emotions
        'sadness': 'Depressed',
        'grief': 'Depressed',
        'disappointment': 'Depressed',
        'remorse': 'Depressed',
        
        # Anxious emotions
        'fear': 'Anxious',
        'nervousness': 'Anxious',
        
        # Stressed emotions
        'anger': 'Stressed',
        'annoyance': 'Stressed',
        'disgust': 'Stressed',
        'disapproval': 'Stressed',
        
        # Normal/Positive emotions
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
        
        # Mixed/Unclear emotions
        'confusion': 'Normal',
        'embarrassment': 'Normal',
        'realization': 'Normal'
    }
    
    def map_emotions_to_condition(row):
        # Get the emotion with highest value (1) for this row
        emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                          'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                          'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                          'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                          'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        
        # Find the emotion with value 1
        for emotion in emotion_columns:
            if row[emotion] == 1:
                return emotion_to_condition.get(emotion, 'Normal')
        
        # If no emotion is 1, return Normal
        return 'Normal'
    
    # Apply mapping to go emotions dataset
    emotions_df['condition'] = emotions_df.apply(map_emotions_to_condition, axis=1)
    emotions_df_clean = emotions_df[['text', 'condition']].copy()
    
    # Remove very short texts and clean data
    emotions_df_clean = emotions_df_clean[emotions_df_clean['text'].str.len() > 10]
    emotions_df_clean = emotions_df_clean.dropna(subset=['text'])
    
    print(f"Go emotions processed: {len(emotions_df_clean)} rows")
    print("Go emotions condition distribution:")
    print(emotions_df_clean['condition'].value_counts())
    
    # Sample from each dataset to create a balanced, large dataset
    print("\nSampling data to create balanced dataset...")
    
    # Target: 2000 examples per condition (8000 total)
    target_per_condition = 2000
    
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
    
    print(f"\n🎉 Final merged dataset: {len(final_df)} rows")
    print("\nFinal condition distribution:")
    print(final_df['condition'].value_counts())
    
    # Save the comprehensive dataset
    output_path = "data/merged_dataset.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved comprehensive dataset to {output_path}")
    
    return final_df

if __name__ == "__main__":
    merge_all_datasets()

