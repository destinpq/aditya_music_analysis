import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Model storage location
MODEL_DIR = Path(__file__).parent / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class YouTubePredictionModels:
    """ML models for predicting YouTube video performance metrics"""
    
    def __init__(self):
        self.view_model = None
        self.like_model = None
        self.comment_model = None
        self.features = None
        self.is_trained = False
    
    def create_pipeline(self, model_type="random_forest"):
        """Create a scikit-learn pipeline with preprocessing and model"""
        if model_type == "random_forest":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=10,
                    random_state=42
                ))
            ])
        elif model_type == "gradient_boosting":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, data, feature_cols=None, test_size=0.2):
        """
        Train models to predict view, like, and comment counts
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Training data with features and target variables
        feature_cols : list, optional
            List of feature column names to use
        test_size : float, optional
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Model performance metrics
        """
        # Validate input data
        required_cols = ["view_count", "like_count", "comment_count"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Determine features to use
        if feature_cols:
            self.features = feature_cols
        else:
            # Use all columns except target variables
            self.features = [col for col in data.columns if col not in required_cols]
            
        if not self.features:
            raise ValueError("No feature columns available for training")
            
        # Extract features and targets
        X = data[self.features]
        y_views = data["view_count"]
        y_likes = data["like_count"]
        y_comments = data["comment_count"]
        
        # Split data
        X_train, X_test, y_views_train, y_views_test = train_test_split(
            X, y_views, test_size=test_size, random_state=42
        )
        _, _, y_likes_train, y_likes_test = train_test_split(
            X, y_likes, test_size=test_size, random_state=42
        )
        _, _, y_comments_train, y_comments_test = train_test_split(
            X, y_comments, test_size=test_size, random_state=42
        )
        
        # Train view count model
        self.view_model = self.create_pipeline("random_forest")
        self.view_model.fit(X_train, y_views_train)
        view_preds = self.view_model.predict(X_test)
        view_metrics = {
            "mae": mean_absolute_error(y_views_test, view_preds),
            "r2": r2_score(y_views_test, view_preds)
        }
        
        # Train like count model
        self.like_model = self.create_pipeline("gradient_boosting")
        self.like_model.fit(X_train, y_likes_train)
        like_preds = self.like_model.predict(X_test)
        like_metrics = {
            "mae": mean_absolute_error(y_likes_test, like_preds),
            "r2": r2_score(y_likes_test, like_preds)
        }
        
        # Train comment count model
        self.comment_model = self.create_pipeline("gradient_boosting")
        self.comment_model.fit(X_train, y_comments_train)
        comment_preds = self.comment_model.predict(X_test)
        comment_metrics = {
            "mae": mean_absolute_error(y_comments_test, comment_preds),
            "r2": r2_score(y_comments_test, comment_preds)
        }
        
        self.is_trained = True
        
        return {
            "view_model": view_metrics,
            "like_model": like_metrics,
            "comment_model": comment_metrics,
            "num_features": len(self.features),
            "num_samples": len(data)
        }
    
    def predict(self, features_df):
        """
        Make predictions using trained models
        
        Parameters:
        -----------
        features_df : pandas.DataFrame
            DataFrame containing feature columns
            
        Returns:
        --------
        dict
            Dictionary with predicted values
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
            
        # Ensure all required features are present
        missing_features = [f for f in self.features if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing features for prediction: {missing_features}")
            
        # Extract features in the correct order
        X = features_df[self.features]
        
        # Make predictions
        view_pred = max(0, float(self.view_model.predict(X)[0]))
        like_pred = max(0, float(self.like_model.predict(X)[0]))
        comment_pred = max(0, float(self.comment_model.predict(X)[0]))
        
        return {
            "predicted_views": int(view_pred),
            "predicted_likes": int(like_pred),
            "predicted_comments": int(comment_pred)
        }
    
    def save_models(self, filename="youtube_models"):
        """Save trained models to a file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained models")
            
        model_data = {
            "view_model": self.view_model,
            "like_model": self.like_model,
            "comment_model": self.comment_model,
            "features": self.features,
            "is_trained": True
        }
        
        filepath = MODEL_DIR / f"{filename}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
            
        return str(filepath)
    
    def load_models(self, filename="youtube_models"):
        """Load trained models from a file"""
        filepath = MODEL_DIR / f"{filename}.pkl"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, "rb") as f:
            model_data = pickle.dump(f)
            
        self.view_model = model_data["view_model"]
        self.like_model = model_data["like_model"]
        self.comment_model = model_data["comment_model"]
        self.features = model_data["features"]
        self.is_trained = model_data["is_trained"]
        
        return True

# Create a singleton instance
prediction_models = YouTubePredictionModels() 