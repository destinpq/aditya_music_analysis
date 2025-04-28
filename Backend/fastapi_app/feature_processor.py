"""
Process raw YouTube video metadata into machine learning features.
Handles text processing, date/time features, and numerical transformations.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import re
import isodate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
from pathlib import Path

# Create models directory if it doesn't exist
MODELS_DIR = Path(__file__).parent / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class YouTubeFeatureProcessor:
    """Process YouTube video metadata into ML features"""
    
    def __init__(self):
        self.title_vectorizer = None
        self.tags_vectorizer = None
        self.category_encoder = None
        self.is_fitted = False
        
        # Feature configuration
        self.title_max_features = 50
        self.tags_max_features = 50
    
    def preprocess_text(self, text):
        """Basic text preprocessing for titles and tags"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_title(self, df):
        """Extract features from video title"""
        df_processed = df.copy()
        
        if 'title' in df_processed.columns:
            # Clean title
            df_processed['title_cleaned'] = df_processed['title'].apply(self.preprocess_text)
            
            # Get title length
            df_processed['title_length'] = df_processed['title'].fillna('').apply(len)
            
            # Count uppercase characters (shouting/emphasis)
            df_processed['title_uppercase_ratio'] = df_processed['title'].fillna('').apply(
                lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1) if x else 0
            )
            
            # Count exclamation marks (enthusiasm)
            df_processed['title_exclamation_count'] = df_processed['title'].fillna('').apply(
                lambda x: x.count('!')
            )
            
            # Count question marks (curiosity)
            df_processed['title_question_count'] = df_processed['title'].fillna('').apply(
                lambda x: x.count('?')
            )
        
        return df_processed
    
    def process_duration(self, df):
        """Convert ISO8601 duration to seconds and length buckets"""
        df_processed = df.copy()
        
        if 'duration' in df_processed.columns:
            # Convert ISO8601 duration to seconds
            df_processed['duration_seconds'] = df_processed['duration'].apply(
                lambda x: int(isodate.parse_duration(x).total_seconds()) if pd.notnull(x) and x else 0
            )
            
            # Create duration categories
            def categorize_duration(seconds):
                if seconds < 60:  # < 1 min
                    return 'very_short'
                elif seconds < 300:  # < 5 min
                    return 'short'
                elif seconds < 1200:  # < 20 min
                    return 'medium'
                elif seconds < 3600:  # < 1 hour
                    return 'long'
                else:  # >= 1 hour
                    return 'very_long'
            
            df_processed['duration_category'] = df_processed['duration_seconds'].apply(categorize_duration)
        
        return df_processed
    
    def process_datetime(self, df):
        """Extract features from published_at datetime"""
        df_processed = df.copy()
        
        if 'published_at' in df_processed.columns:
            # Convert to datetime
            df_processed['published_at'] = pd.to_datetime(df_processed['published_at'], errors='coerce')
            
            # Extract components
            df_processed['publish_year'] = df_processed['published_at'].dt.year
            df_processed['publish_month'] = df_processed['published_at'].dt.month
            df_processed['publish_day'] = df_processed['published_at'].dt.day
            df_processed['publish_hour'] = df_processed['published_at'].dt.hour
            df_processed['publish_day_of_week'] = df_processed['published_at'].dt.dayofweek
            df_processed['publish_is_weekend'] = df_processed['publish_day_of_week'].apply(
                lambda x: 1 if x >= 5 else 0  # 5 = Saturday, 6 = Sunday
            )
            
            # Create time-of-day categories
            def categorize_time(hour):
                if pd.isna(hour):
                    return 'unknown'
                elif 5 <= hour < 12:
                    return 'morning'
                elif 12 <= hour < 17:
                    return 'afternoon'
                elif 17 <= hour < 21:
                    return 'evening'
                else:
                    return 'night'
            
            df_processed['publish_time_of_day'] = df_processed['publish_hour'].apply(categorize_time)
        
        return df_processed
    
    def process_tags(self, df):
        """Process video tags"""
        df_processed = df.copy()
        
        if 'tags' in df_processed.columns:
            # Count number of tags
            df_processed['tag_count'] = df_processed['tags'].fillna('').apply(
                lambda x: len(str(x).split(',')) if x else 0
            )
            
            # Prepare tags for TF-IDF
            df_processed['tags_processed'] = df_processed['tags'].fillna('').apply(
                lambda x: ' '.join(self.preprocess_text(tag) for tag in str(x).split(',')) if x else ''
            )
        
        return df_processed
    
    def process_category(self, df):
        """Process video category"""
        df_processed = df.copy()
        
        if 'category_id' in df_processed.columns:
            # Ensure category_id is string
            df_processed['category_id'] = df_processed['category_id'].fillna('unknown').astype(str)
            
            # Map common categories
            category_map = {
                '1': 'film_animation',
                '2': 'autos_vehicles',
                '10': 'music',
                '15': 'pets_animals',
                '17': 'sports',
                '18': 'short_movies',
                '19': 'travel_events',
                '20': 'gaming',
                '21': 'videoblogging',
                '22': 'people_blogs',
                '23': 'comedy',
                '24': 'entertainment',
                '25': 'news_politics',
                '26': 'howto_style',
                '27': 'education',
                '28': 'science_technology',
                '29': 'nonprofit_activism',
                '30': 'movies',
                '31': 'anime_animation',
                '32': 'action_adventure',
                '33': 'classics',
                '34': 'comedy',
                '35': 'documentary',
                '36': 'drama',
                '37': 'family',
                '38': 'foreign',
                '39': 'horror',
                '40': 'sci_fi_fantasy',
                '41': 'thriller',
                '42': 'shorts',
                '43': 'shows',
                '44': 'trailers'
            }
            
            df_processed['category_name'] = df_processed['category_id'].map(
                lambda x: category_map.get(x, 'other')
            )
        
        return df_processed
    
    def create_tfidf_features(self, df, fit=True):
        """Create TF-IDF features from title and tags"""
        df_processed = df.copy()
        
        # TF-IDF for title
        if 'title_cleaned' in df_processed.columns:
            if fit or self.title_vectorizer is None:
                self.title_vectorizer = TfidfVectorizer(
                    max_features=self.title_max_features,
                    stop_words='english',
                    min_df=2
                )
                title_tfidf = self.title_vectorizer.fit_transform(df_processed['title_cleaned'].fillna(''))
            else:
                title_tfidf = self.title_vectorizer.transform(df_processed['title_cleaned'].fillna(''))
            
            # Convert to DataFrame
            title_cols = [f'title_tfidf_{i}' for i in range(title_tfidf.shape[1])]
            title_tfidf_df = pd.DataFrame(
                title_tfidf.toarray(),
                columns=title_cols,
                index=df_processed.index
            )
            
            # Add to processed DataFrame
            df_processed = pd.concat([df_processed, title_tfidf_df], axis=1)
        
        # TF-IDF for tags
        if 'tags_processed' in df_processed.columns:
            if fit or self.tags_vectorizer is None:
                self.tags_vectorizer = TfidfVectorizer(
                    max_features=self.tags_max_features,
                    stop_words='english',
                    min_df=2
                )
                tags_tfidf = self.tags_vectorizer.fit_transform(df_processed['tags_processed'].fillna(''))
            else:
                tags_tfidf = self.tags_vectorizer.transform(df_processed['tags_processed'].fillna(''))
            
            # Convert to DataFrame
            tags_cols = [f'tags_tfidf_{i}' for i in range(tags_tfidf.shape[1])]
            tags_tfidf_df = pd.DataFrame(
                tags_tfidf.toarray(),
                columns=tags_cols,
                index=df_processed.index
            )
            
            # Add to processed DataFrame
            df_processed = pd.concat([df_processed, tags_tfidf_df], axis=1)
        
        return df_processed
    
    def one_hot_encode_categories(self, df, fit=True):
        """One-hot encode categorical features"""
        df_processed = df.copy()
        
        # One-hot encode category_id
        if 'category_id' in df_processed.columns:
            if fit or self.category_encoder is None:
                self.category_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                category_encoded = self.category_encoder.fit_transform(df_processed[['category_id']])
            else:
                category_encoded = self.category_encoder.transform(df_processed[['category_id']])
            
            # Create column names
            if hasattr(self.category_encoder, 'categories_'):
                category_cols = [f'category_{cat}' for cat in self.category_encoder.categories_[0]]
            else:
                category_cols = [f'category_{i}' for i in range(category_encoded.shape[1])]
            
            # Convert to DataFrame
            category_df = pd.DataFrame(
                category_encoded,
                columns=category_cols,
                index=df_processed.index
            )
            
            # Add to processed DataFrame
            df_processed = pd.concat([df_processed, category_df], axis=1)
        
        # One-hot encode time-of-day
        if 'publish_time_of_day' in df_processed.columns:
            time_dummies = pd.get_dummies(
                df_processed['publish_time_of_day'], 
                prefix='time',
                dummy_na=False
            )
            df_processed = pd.concat([df_processed, time_dummies], axis=1)
        
        # One-hot encode duration category
        if 'duration_category' in df_processed.columns:
            duration_dummies = pd.get_dummies(
                df_processed['duration_category'], 
                prefix='duration',
                dummy_na=False
            )
            df_processed = pd.concat([df_processed, duration_dummies], axis=1)
        
        return df_processed
    
    def clean_numerical_columns(self, df):
        """Clean numerical columns like view_count, like_count, etc."""
        df_processed = df.copy()
        
        numeric_cols = ['view_count', 'like_count', 'dislike_count', 
                       'favorite_count', 'comment_count']
        
        for col in numeric_cols:
            if col in df_processed.columns:
                # Handle comma-separated numbers and convert to integer
                df_processed[col] = df_processed[col].apply(
                    lambda x: int(str(x).replace(',', '').split('.')[0]) if pd.notnull(x) else 0
                )
        
        return df_processed
    
    def process_features(self, data, fit=True):
        """Process all features from raw YouTube metadata"""
        # Convert input to DataFrame if it's a dict or series
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data.to_dict()])
        else:
            df = data.copy()
        
        # Apply all feature processing steps
        df = self.process_title(df)
        df = self.process_duration(df)
        df = self.process_datetime(df)
        df = self.process_tags(df)
        df = self.process_category(df)
        df = self.clean_numerical_columns(df)
        
        # Apply feature extraction methods
        df = self.create_tfidf_features(df, fit)
        df = self.one_hot_encode_categories(df, fit)
        
        if fit:
            self.is_fitted = True
        
        return df
    
    def get_feature_names(self):
        """Get list of all feature names produced by this processor"""
        return [
            # Basic features
            'title_length', 'title_uppercase_ratio', 'title_exclamation_count', 
            'title_question_count', 'duration_seconds', 'publish_year', 
            'publish_month', 'publish_day', 'publish_hour', 'publish_day_of_week',
            'publish_is_weekend', 'tag_count',
            
            # TF-IDF features are dynamic
            # Category features are dynamic
            # One-hot encoded features are dynamic
        ]
    
    def save_processor(self, filename="feature_processor"):
        """Save the fitted feature processor to disk"""
        processor_data = {
            'title_vectorizer': self.title_vectorizer,
            'tags_vectorizer': self.tags_vectorizer,
            'category_encoder': self.category_encoder,
            'is_fitted': self.is_fitted,
            'title_max_features': self.title_max_features,
            'tags_max_features': self.tags_max_features
        }
        
        filepath = MODELS_DIR / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
            
        return str(filepath)
    
    def load_processor(self, filename="feature_processor"):
        """Load a fitted feature processor from disk"""
        filepath = MODELS_DIR / f"{filename}.pkl"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature processor not found at {filepath}")
            
        with open(filepath, 'rb') as f:
            processor_data = pickle.load(f)
            
        self.title_vectorizer = processor_data['title_vectorizer']
        self.tags_vectorizer = processor_data['tags_vectorizer']
        self.category_encoder = processor_data['category_encoder']
        self.is_fitted = processor_data['is_fitted']
        self.title_max_features = processor_data['title_max_features']
        self.tags_max_features = processor_data['tags_max_features']
        
        return True

# Create a singleton instance
feature_processor = YouTubeFeatureProcessor() 