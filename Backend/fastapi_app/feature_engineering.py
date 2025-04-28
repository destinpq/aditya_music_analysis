import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import isodate
import json

class FeatureEngineering:
    """
    Feature engineering for YouTube video metadata.
    Processes raw video data into model-ready features.
    """
    
    def __init__(self):
        self.title_vectorizer = None
        self.tags_vectorizer = None
        self.category_encoder = None
        self.is_fitted = False
    
    def clean_numeric_columns(self, df):
        """Clean view count, like count, and comment count columns"""
        numeric_cols = ['view_count', 'like_count', 'comment_count']
        df_cleaned = df.copy()
        
        for col in numeric_cols:
            if col in df_cleaned.columns:
                # Remove commas and convert to integer
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(',', '').str.split('.').str[0]
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0).astype(int)
        
        return df_cleaned
    
    def process_duration(self, df):
        """Convert ISO8601 duration to seconds"""
        df_processed = df.copy()
        
        if 'duration' in df_processed.columns:
            df_processed['duration_seconds'] = df_processed['duration'].apply(
                lambda x: int(isodate.parse_duration(x).total_seconds()) if pd.notnull(x) else 0
            )
        
        return df_processed
    
    def extract_datetime_features(self, df):
        """Extract features from published_at datetime"""
        df_processed = df.copy()
        
        if 'published_at' in df_processed.columns:
            # Convert to datetime if not already
            df_processed['published_at'] = pd.to_datetime(df_processed['published_at'])
            
            # Extract temporal features
            df_processed['publish_month'] = df_processed['published_at'].dt.month
            df_processed['publish_day_of_week'] = df_processed['published_at'].dt.dayofweek
            df_processed['publish_hour'] = df_processed['published_at'].dt.hour
        
        return df_processed
    
    def process_tags(self, df):
        """Process tags and count them"""
        df_processed = df.copy()
        
        if 'tags' in df_processed.columns:
            # Count number of tags
            df_processed['tag_count'] = df_processed['tags'].fillna('').apply(
                lambda x: len(x.split(',')) if x else 0
            )
            
            # Clean tags for TF-IDF
            df_processed['tags_processed'] = df_processed['tags'].fillna('').apply(
                lambda x: ' '.join(tag.strip() for tag in x.split(',')) if x else ''
            )
        
        return df_processed
    
    def create_tfidf_features(self, df, fit=True):
        """Create TF-IDF features for title and tags"""
        df_processed = df.copy()
        
        # Process title
        if 'title' in df_processed.columns:
            if fit or self.title_vectorizer is None:
                self.title_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                title_tfidf = self.title_vectorizer.fit_transform(df_processed['title'].fillna(''))
            else:
                title_tfidf = self.title_vectorizer.transform(df_processed['title'].fillna(''))
            
            title_tfidf_df = pd.DataFrame(
                title_tfidf.toarray(),
                columns=[f'title_tfidf_{i}' for i in range(title_tfidf.shape[1])],
                index=df_processed.index
            )
            df_processed = pd.concat([df_processed, title_tfidf_df], axis=1)
        
        # Process tags
        if 'tags_processed' in df_processed.columns:
            if fit or self.tags_vectorizer is None:
                self.tags_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                tags_tfidf = self.tags_vectorizer.fit_transform(df_processed['tags_processed'])
            else:
                tags_tfidf = self.tags_vectorizer.transform(df_processed['tags_processed'])
            
            tags_tfidf_df = pd.DataFrame(
                tags_tfidf.toarray(),
                columns=[f'tags_tfidf_{i}' for i in range(tags_tfidf.shape[1])],
                index=df_processed.index
            )
            df_processed = pd.concat([df_processed, tags_tfidf_df], axis=1)
        
        return df_processed
    
    def one_hot_encode_category(self, df, fit=True):
        """One-hot encode category_id"""
        df_processed = df.copy()
        
        if 'category_id' in df_processed.columns:
            # Fill NaN values and convert to string
            df_processed['category_id'] = df_processed['category_id'].fillna('unknown').astype(str)
            
            if fit or self.category_encoder is None:
                self.category_encoder = OneHotEncoder(sparse=False)
                category_encoded = self.category_encoder.fit_transform(df_processed[['category_id']])
            else:
                category_encoded = self.category_encoder.transform(df_processed[['category_id']])
            
            # Create DataFrame with encoded values
            category_columns = [f'category_{cat}' for cat in self.category_encoder.categories_[0]]
            category_df = pd.DataFrame(
                category_encoded,
                columns=category_columns,
                index=df_processed.index
            )
            
            # Concatenate with processed DataFrame
            df_processed = pd.concat([df_processed, category_df], axis=1)
        
        return df_processed
    
    def process(self, data, fit=True):
        """Process the input data with all feature engineering steps"""
        # Convert to DataFrame if JSON or dict
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                raise ValueError("Invalid JSON string")
                
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
            
        # Apply all feature engineering steps
        df = self.clean_numeric_columns(df)
        df = self.process_duration(df)
        df = self.extract_datetime_features(df)
        df = self.process_tags(df)
        df = self.create_tfidf_features(df, fit)
        df = self.one_hot_encode_category(df, fit)
        
        if fit:
            self.is_fitted = True
            
        return df
    
    def get_features_list(self):
        """Get list of all features created by this processor"""
        features = [
            'duration_seconds',
            'publish_month', 
            'publish_day_of_week',
            'publish_hour',
            'tag_count'
        ]
        
        # Add TF-IDF features
        if self.title_vectorizer is not None:
            features.extend([f'title_tfidf_{i}' for i in range(len(self.title_vectorizer.get_feature_names_out()))])
            
        if self.tags_vectorizer is not None:
            features.extend([f'tags_tfidf_{i}' for i in range(len(self.tags_vectorizer.get_feature_names_out()))])
            
        # Add category features
        if self.category_encoder is not None:
            features.extend([f'category_{cat}' for cat in self.category_encoder.categories_[0]])
            
        return features 