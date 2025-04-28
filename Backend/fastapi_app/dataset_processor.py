"""
Dataset processor for YouTube video data.
Handles preprocessing, feature engineering, and train/test splitting.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import isodate
from datetime import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Make sure the data directory exists
DATA_DIR = Path(__file__).parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Make sure the models directory exists
MODELS_DIR = Path(__file__).parent / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class YouTubeDatasetProcessor:
    """Process YouTube datasets for ML training and prediction"""
    
    def __init__(self):
        self.feature_columns = []
        self.target_columns = ['view_count', 'like_count', 'comment_count']
        self.category_encoder = None
        self.preprocessing_state = {}
    
    def process_dataset(self, 
                        df: pd.DataFrame, 
                        test_size: float = 0.2, 
                        random_state: int = 42,
                        dataset_name: str = "youtube_dataset") -> Dict[str, Any]:
        """
        Process a YouTube dataset and split into train/test sets
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing YouTube video data
        test_size : float
            Proportion of data to include in test set
        random_state : int
            Random seed for reproducibility
        dataset_name : str
            Name for saving the processed data
            
        Returns:
        --------
        dict
            Processing results and dataset stats
        """
        print(f"Processing dataset with {df.shape[0]} rows")
        
        # Process features
        df_processed = self._process_features(df)
        
        # Ensure target columns exist
        for col in self.target_columns:
            if col not in df_processed.columns:
                print(f"Warning: Target column {col} not found, creating with random values")
                if col == 'view_count':
                    df_processed[col] = np.random.randint(1000, 1000000, size=df_processed.shape[0])
                elif col == 'like_count':
                    df_processed[col] = df_processed.get('view_count', 10000) * np.random.uniform(0.01, 0.1, size=df_processed.shape[0])
                elif col == 'comment_count':
                    df_processed[col] = df_processed.get('view_count', 10000) * np.random.uniform(0.001, 0.01, size=df_processed.shape[0])
                else:
                    df_processed[col] = np.random.randint(0, 1000, size=df_processed.shape[0])
        
        # Get features and targets
        X = df_processed[self.feature_columns]
        y = {target: df_processed[target].values for target in self.target_columns}
        
        # Split data for each target
        train_test_data = {}
        
        for target in self.target_columns:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y[target], test_size=test_size, random_state=random_state
            )
            
            train_test_data[target] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        
        # Save the preprocessing state
        self.preprocessing_state = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'category_encoder': self.category_encoder
        }
        
        # Save to disk
        self._save_processed_data(train_test_data, dataset_name)
        
        # Return summary
        result = {
            'dataset_name': dataset_name,
            'num_samples': df.shape[0],
            'num_features': len(self.feature_columns),
            'features': self.feature_columns,
            'targets': self.target_columns,
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'saved_path': str(DATA_DIR / dataset_name)
        }
        
        return result
    
    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and extract features from YouTube data"""
        df_processed = df.copy()
        
        # 1. Process Duration - convert to seconds
        self._process_duration(df_processed)
        
        # 2. Process publication date
        self._process_datetime(df_processed)
        
        # 3. Process tags
        self._process_tags(df_processed)
        
        # 4. Process category ID - one-hot encode
        self._process_category(df_processed)
        
        # Define core feature columns
        core_features = ['DurationSeconds', 'PublishMonth', 'PublishDay', 
                        'PublishHour', 'TagCount']
        
        # Add category features if available
        category_features = [col for col in df_processed.columns if col.startswith('Category_')]
        
        # Combine all features
        self.feature_columns = core_features + category_features
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df_processed.columns:
                print(f"Warning: Feature column {col} not found, creating with zeros")
                df_processed[col] = 0
        
        return df_processed
    
    def _process_duration(self, df: pd.DataFrame) -> None:
        """Convert ISO8601 duration to seconds"""
        duration_col = next((col for col in ['duration', 'Duration'] if col in df.columns), None)
        
        if duration_col:
            df['DurationSeconds'] = df[duration_col].apply(
                lambda x: int(isodate.parse_duration(x).total_seconds()) 
                if pd.notnull(x) and isinstance(x, str) else 0
            )
        else:
            print("Warning: No duration column found")
            df['DurationSeconds'] = 0
    
    def _process_datetime(self, df: pd.DataFrame) -> None:
        """Extract date/time features from published date"""
        date_col = next((col for col in ['published_at', 'PublishedAt', 
                                        'published_date', 'PublishedDate'] 
                        if col in df.columns), None)
        
        if date_col:
            df['PublishedAt'] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Extract features
            df['PublishMonth'] = df['PublishedAt'].dt.month
            df['PublishDay'] = df['PublishedAt'].dt.day
            df['PublishHour'] = df['PublishedAt'].dt.hour
            df['PublishDayOfWeek'] = df['PublishedAt'].dt.dayofweek
            
            # Is weekend feature (0 = weekday, 1 = weekend)
            df['IsWeekend'] = df['PublishDayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        else:
            print("Warning: No publish date column found")
            df['PublishMonth'] = 1
            df['PublishDay'] = 1
            df['PublishHour'] = 12
            df['PublishDayOfWeek'] = 0
            df['IsWeekend'] = 0
    
    def _process_tags(self, df: pd.DataFrame) -> None:
        """Process video tags"""
        tags_col = next((col for col in ['tags', 'Tags'] if col in df.columns), None)
        
        if tags_col:
            df['TagCount'] = df[tags_col].fillna('').apply(
                lambda x: len(str(x).split(',')) if x else 0
            )
        else:
            print("Warning: No tags column found")
            df['TagCount'] = 0
    
    def _process_category(self, df: pd.DataFrame) -> None:
        """One-hot encode video category"""
        category_col = next((col for col in ['category_id', 'CategoryID', 'category'] 
                           if col in df.columns), None)
        
        if category_col:
            # Ensure categorical values are strings
            df['CategoryID'] = df[category_col].fillna('unknown').astype(str)
            
            # Create or use existing encoder
            if self.category_encoder is None:
                self.category_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                category_encoded = self.category_encoder.fit_transform(df[['CategoryID']])
            else:
                category_encoded = self.category_encoder.transform(df[['CategoryID']])
            
            # Get category names
            if hasattr(self.category_encoder, 'categories_'):
                category_names = self.category_encoder.categories_[0]
            else:
                category_names = [f"cat_{i}" for i in range(category_encoded.shape[1])]
            
            # Create DataFrame with encoded categories
            category_cols = [f'Category_{cat}' for cat in category_names]
            category_df = pd.DataFrame(
                category_encoded,
                columns=category_cols,
                index=df.index
            )
            
            # Add encoded categories to main DataFrame
            for col in category_cols:
                df[col] = category_df[col]
        else:
            print("Warning: No category column found")
            # Add a dummy category column
            df['Category_unknown'] = 1
    
    def _save_processed_data(self, train_test_data: Dict, dataset_name: str) -> None:
        """Save processed data and preprocessing state"""
        # Create dataset directory
        dataset_dir = DATA_DIR / dataset_name
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save each train/test split
        for target, data in train_test_data.items():
            data['X_train'].to_csv(dataset_dir / f"X_train_{target}.csv", index=False)
            data['X_test'].to_csv(dataset_dir / f"X_test_{target}.csv", index=False)
            
            pd.DataFrame(data['y_train'], columns=[target]).to_csv(
                dataset_dir / f"y_train_{target}.csv", index=False
            )
            pd.DataFrame(data['y_test'], columns=[target]).to_csv(
                dataset_dir / f"y_test_{target}.csv", index=False
            )
        
        # Save preprocessing state
        with open(dataset_dir / "preprocessing.pkl", 'wb') as f:
            pickle.dump(self.preprocessing_state, f)
    
    def load_preprocessing_state(self, dataset_name: str) -> bool:
        """Load preprocessing state from a saved dataset"""
        preprocessing_path = DATA_DIR / dataset_name / "preprocessing.pkl"
        
        if not os.path.exists(preprocessing_path):
            return False
        
        with open(preprocessing_path, 'rb') as f:
            self.preprocessing_state = pickle.load(f)
            
        self.feature_columns = self.preprocessing_state.get('feature_columns', [])
        self.target_columns = self.preprocessing_state.get('target_columns', [])
        self.category_encoder = self.preprocessing_state.get('category_encoder', None)
        
        return True
    
    def process_new_data(self, 
                        df: pd.DataFrame, 
                        dataset_name: str = "youtube_dataset") -> pd.DataFrame:
        """
        Process new data using existing preprocessing state
        
        Parameters:
        -----------
        df : pandas.DataFrame
            New data to process
        dataset_name : str
            Name of dataset with saved preprocessing state
            
        Returns:
        --------
        pandas.DataFrame
            Processed features ready for prediction
        """
        # Load preprocessing state if not already loaded
        if not self.preprocessing_state:
            success = self.load_preprocessing_state(dataset_name)
            if not success:
                raise ValueError(f"No preprocessing state found for {dataset_name}")
        
        # Process features
        df_processed = self._process_features(df)
        
        # Return only the required features
        return df_processed[self.feature_columns]
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available processed datasets"""
        return [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a processed dataset"""
        dataset_dir = DATA_DIR / dataset_name
        
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            return {"error": f"Dataset {dataset_name} not found"}
        
        # Load preprocessing state
        preprocessing_path = dataset_dir / "preprocessing.pkl"
        if not preprocessing_path.exists():
            return {"error": f"Preprocessing information missing for {dataset_name}"}
        
        with open(preprocessing_path, 'rb') as f:
            preprocessing = pickle.load(f)
        
        # Get file sizes and counts
        train_files = [f for f in dataset_dir.iterdir() if f.name.startswith('X_train_')]
        test_files = [f for f in dataset_dir.iterdir() if f.name.startswith('X_test_')]
        
        # Get sample counts from first train/test file
        train_size = 0
        test_size = 0
        
        if train_files:
            train_df = pd.read_csv(train_files[0])
            train_size = train_df.shape[0]
            
        if test_files:
            test_df = pd.read_csv(test_files[0])
            test_size = test_df.shape[0]
        
        return {
            "dataset_name": dataset_name,
            "feature_columns": preprocessing.get('feature_columns', []),
            "target_columns": preprocessing.get('target_columns', []),
            "train_size": train_size,
            "test_size": test_size,
            "total_size": train_size + test_size,
            "train_files": [f.name for f in train_files],
            "test_files": [f.name for f in test_files]
        }

# Create singleton instance
dataset_processor = YouTubeDatasetProcessor() 