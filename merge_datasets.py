import pandas as pd
import numpy as np

def merge_datasets():
    # Load existing dataset
    print("Loading existing dataset...")
    existing_df = pd.read_csv("data/merged_dataset.csv")
    print(f"Existing dataset: {len(existing_df)} rows")
    
    # Load suicide detection dataset
    print("Loading suicide detection dataset...")
    suicide_df = pd.read_csv("data/Suicide_Detection.csv")
    print(f"Suicide detection dataset: {len(suicide_df)} rows")
    
    # Clean suicide detection data
    suicide_df = suicide_df.dropna(subset=['text', 'class'])
    suicide_df = suicide_df[suicide_df['text'].str.len() > 10]  # Remove very short texts
    
    # Map suicide detection classes to mental health conditions
    def map_suicide_class_to_condition(row):
        if row['class'] == 'suicide':
            # Suicide-related texts are likely depressed
            return 'Depressed'
        else:
            # Non-suicide texts could be various conditions
            # Let's use some heuristics to categorize them
            text = row['text'].lower()
            
            # Keywords for different conditions
            anxious_keywords = ['anxious', 'anxiety', 'worried', 'worry', 'nervous', 'fear', 'afraid', 'panic']
            stressed_keywords = ['stressed', 'stress', 'overwhelmed', 'frustrated', 'frustration', 'angry', 'anger']
            depressed_keywords = ['sad', 'depressed', 'hopeless', 'empty', 'tired', 'exhausted', 'lonely']
            normal_keywords = ['happy', 'good', 'great', 'excited', 'love', 'calm', 'relaxed', 'fine', 'okay']
            
            # Count keyword matches
            anxious_count = sum(1 for word in anxious_keywords if word in text)
            stressed_count = sum(1 for word in stressed_keywords if word in text)
            depressed_count = sum(1 for word in depressed_keywords if word in text)
            normal_count = sum(1 for word in normal_keywords if word in text)
            
            # Determine condition based on keyword frequency
            if depressed_count > 0:
                return 'Depressed'
            elif anxious_count > 0:
                return 'Anxious'
            elif stressed_count > 0:
                return 'Stressed'
            elif normal_count > 0:
                return 'Normal'
            else:
                # Default to Normal if no clear indicators
                return 'Normal'
    
    # Apply mapping to suicide detection dataset
    print("Mapping suicide detection classes to mental health conditions...")
    suicide_df['condition'] = suicide_df.apply(map_suicide_class_to_condition, axis=1)
    
    # Select relevant columns and sample to balance the dataset
    suicide_df_clean = suicide_df[['text', 'condition']].copy()
    
    # Sample to get a reasonable size (let's get 1000 examples per condition)
    print("Sampling data to balance conditions...")
    conditions = ['Depressed', 'Anxious', 'Stressed', 'Normal']
    sampled_dfs = []
    
    for condition in conditions:
        condition_data = suicide_df_clean[suicide_df_clean['condition'] == condition]
        if len(condition_data) > 1000:
            sampled = condition_data.sample(n=1000, random_state=42)
        else:
            sampled = condition_data
        sampled_dfs.append(sampled)
        print(f"{condition}: {len(sampled)} examples")
    
    # Combine sampled data
    suicide_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # Merge with existing dataset
    print("Merging datasets...")
    merged_df = pd.concat([existing_df, suicide_sampled], ignore_index=True)
    
    # Shuffle the data
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final merged dataset: {len(merged_df)} rows")
    print("\nCondition distribution:")
    print(merged_df['condition'].value_counts())
    
    # Save merged dataset
    output_path = "data/merged_dataset.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved merged dataset to {output_path}")
    
    return merged_df

if __name__ == "__main__":
    merge_datasets()
