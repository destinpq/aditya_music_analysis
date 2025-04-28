"""
Prediction service for YouTube view count prediction.
Provides functionality to train models, make predictions, and find optimal posting times.
"""

import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from database import SessionLocal, Video, Dataset
from advanced_ml_models import XGBoostRegressor, LightGBMRegressor, CatBoostRegressor, AdvancedRegressionModels
from hyperparameter_tuning import get_optimal_params

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

class PredictionService:
    """
    Service for making predictions using trained models.
    """
    def __init__(self, model_dir: str = "models"):
        """
        Initialize prediction service.
        
        Parameters:
        -----------
        model_dir : str
            Directory where models are stored
        """
        self.model_dir = model_dir
        self.models = None
        self.feature_columns = None
        self.target_column = None
        self.model_metadata = None
        
        # Try to load models on initialization
        try:
            self.load_models()
        except Exception as e:
            logger.warning(f"Could not load models on initialization: {e}")
            logger.info("Models will need to be trained before predictions can be made.")
    
    def load_models(self) -> None:
        """
        Load trained models from disk.
        """
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory {self.model_dir} not found")
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata file {metadata_path} not found")
        
        with open(metadata_path, 'r') as f:
            self.model_metadata = json.load(f)
        
        # Get feature and target columns
        self.feature_columns = self.model_metadata.get('feature_columns')
        self.target_column = self.model_metadata.get('target_column')
        
        # Initialize model instance
        self.models = AdvancedRegressionModels()
        self.models.load_models(self.model_dir)
        
        logger.info(f"Loaded models from {self.model_dir}")
        logger.info(f"Available models: {list(self.model_metadata['models'].keys())}")
    
    def predict(self, 
                features: Dict[str, Any], 
                model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a prediction using the specified model.
        
        Parameters:
        -----------
        features : dict
            Dictionary of feature values
        model_name : str, optional
            Name of model to use for prediction.
            If None, uses the best model according to metadata.
        
        Returns:
        --------
        dict
            Dictionary with prediction results
        """
        if self.models is None:
            raise ValueError("Models not loaded. Please load or train models first.")
        
        # Check if we have all required features
        missing_features = set(self.feature_columns) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract only the features we need and in the correct order
        feature_values = [features[col] for col in self.feature_columns]
        
        # Convert to numpy array for prediction
        X = np.array([feature_values])
        
        # Determine which model to use
        if model_name is None:
            model_name = self.model_metadata.get('best_model', 'xgboost')
        
        if model_name not in self.model_metadata['models']:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_metadata['models'].keys())}")
        
        # Make prediction
        prediction = self.models.predict(X, model_name)
        
        # Get model metrics
        model_metrics = self.model_metadata['models'][model_name]
        
        # Prepare response
        result = {
            'prediction': float(prediction[0]),
            'model_used': model_name,
            'model_metrics': {
                'rmse': model_metrics['rmse'],
                'r2': model_metrics['r2']
            },
            'confidence': self._calculate_confidence(model_metrics['r2'])
        }
        
        return result
    
    def predict_all_models(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions using all available models.
        
        Parameters:
        -----------
        features : dict
            Dictionary of feature values
        
        Returns:
        --------
        dict
            Dictionary with predictions from all models
        """
        if self.models is None:
            raise ValueError("Models not loaded. Please load or train models first.")
        
        results = {}
        for model_name in self.model_metadata['models'].keys():
            results[model_name] = self.predict(features, model_name)
        
        # Add ensemble prediction (average of all models)
        ensemble_prediction = np.mean([results[model]['prediction'] for model in results])
        results['ensemble'] = {
            'prediction': float(ensemble_prediction),
            'model_used': 'ensemble',
            'model_metrics': {
                'rmse': np.mean([results[model]['model_metrics']['rmse'] for model in results if model != 'ensemble']),
                'r2': np.mean([results[model]['model_metrics']['r2'] for model in results if model != 'ensemble'])
            }
        }
        
        return results
    
    def _calculate_confidence(self, r2: float) -> float:
        """
        Calculate a confidence score based on the R² value.
        
        Parameters:
        -----------
        r2 : float
            R² value of the model
        
        Returns:
        --------
        float
            Confidence score between 0 and 1
        """
        # Map R² to a confidence score between 0 and 1
        # R² can be negative in worst case, or 1 in best case
        # We clamp it to 0-1 range and map it to a percentage
        return max(0, min(1, r2)) * 100
    
    def find_optimal_posting_time(self, 
                                 content_features: Dict[str, Any],
                                 time_range_days: int = 7,
                                 time_step_hours: int = 1) -> Dict[str, Any]:
        """
        Find the optimal posting time based on predicted view counts.
        
        Parameters:
        -----------
        content_features : dict
            Dictionary of content features (excluding time features)
        time_range_days : int
            Number of days to look ahead
        time_step_hours : int
            Step size for time predictions in hours
        
        Returns:
        --------
        dict
            Dictionary with optimal posting time and predicted view count
        """
        if self.models is None:
            raise ValueError("Models not loaded. Please load or train models first.")
        
        # Generate time range
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=time_range_days)
        time_steps = []
        
        current_time = start_time
        while current_time < end_time:
            time_steps.append(current_time)
            current_time += timedelta(hours=time_step_hours)
        
        # Make predictions for each time step
        predictions = []
        for timestamp in time_steps:
            # Extract time features from timestamp
            time_features = {
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
                'month': timestamp.month,
                'day': timestamp.day
            }
            
            # Combine with content features
            features = {**content_features, **time_features}
            
            # Make prediction
            result = self.predict(features)
            predictions.append({
                'timestamp': timestamp.isoformat(),
                'prediction': result['prediction'],
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'day_name': timestamp.strftime('%A')
            })
        
        # Sort predictions by predicted view count
        sorted_predictions = sorted(predictions, key=lambda x: x['prediction'], reverse=True)
        
        # Return top 5 posting times
        return {
            'optimal_posting_time': sorted_predictions[0]['timestamp'],
            'top_posting_times': sorted_predictions[:5],
            'all_predictions': sorted_predictions
        }
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get feature importance from the specified model.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of model to get feature importance from.
            If None, uses the best model according to metadata.
        
        Returns:
        --------
        dict
            Dictionary with feature names and their importance scores
        """
        if self.models is None:
            raise ValueError("Models not loaded. Please load or train models first.")
        
        # Determine which model to use
        if model_name is None:
            model_name = self.model_metadata.get('best_model', 'xgboost')
        
        if model_name not in self.model_metadata['models']:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_metadata['models'].keys())}")
        
        # Get feature importance
        # Note: Not all models support feature importance
        try:
            importance = self.models.get_feature_importance(model_name)
            
            # Map importance to feature names
            feature_importance = {
                feature: float(importance[i]) 
                for i, feature in enumerate(self.feature_columns)
            }
            
            # Sort by importance
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda item: item[1], 
                reverse=True
            ))
            
            return sorted_importance
            
        except AttributeError:
            logger.warning(f"Model {model_name} does not support feature importance")
            return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

# Singleton instance
prediction_service = PredictionService()

def get_prediction_service() -> PredictionService:
    """
    Get prediction service instance
    
    Returns:
        PredictionService instance
    """
    return prediction_service 