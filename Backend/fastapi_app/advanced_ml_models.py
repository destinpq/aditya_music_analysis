"""
Advanced ML models for YouTube view count prediction.
Implements XGBoost, LightGBM, and CatBoost regressors with common interfaces.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import time
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import json
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory to store trained models
MODEL_DIR = Path(__file__).parent / "models" / "advanced"
os.makedirs(MODEL_DIR, exist_ok=True)

# Try importing required libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not available. Install with: pip install catboost")
    CATBOOST_AVAILABLE = False

# Check if TensorFlow is installed
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    HAVE_TF = True
except ImportError:
    HAVE_TF = False
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")

# Define base regressor class
class BaseRegressor(ABC):
    """Base abstract class for all regressors"""
    
    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the regressor
        
        Args:
            model_name: Name of the model
            params: Model hyperparameters
        """
        self.model_name = model_name
        self.params = params or {}
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self.is_trained = False
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create and return the model instance"""
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {self.model_name}...")
        self.feature_names = list(X_train.columns)
        self.model = self._create_model()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Get training metrics
        train_preds = self.predict(X_train)
        rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
        r2 = float(r2_score(y_train, train_preds))
        
        self.metrics = {
            "train_rmse": rmse,
            "train_r2": r2
        }
        
        self.is_trained = True
        logger.info(f"{self.model_name} training complete. Train RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return self.metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained before evaluation")
        
        logger.info(f"Evaluating {self.model_name}...")
        test_preds = self.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
        r2 = float(r2_score(y_test, test_preds))
        
        self.metrics.update({
            "test_rmse": rmse,
            "test_r2": r2
        })
        
        logger.info(f"{self.model_name} evaluation complete. Test RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained before prediction")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Ensure X has all the required features
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            # Add missing columns with zeros
            for col in missing_cols:
                X[col] = 0
                
        # Ensure X only has the features the model was trained on
        X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame
        
        Returns:
            DataFrame with feature names and importance values
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained to get feature importance")
        
        # Implementation depends on the specific model
        # Subclasses should override this method
        return pd.DataFrame()
    
    def save(self, model_dir: str) -> str:
        """
        Save model to disk
        
        Args:
            model_dir: Directory to save model
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'params': self.params,
                'is_trained': self.is_trained
            }, f)
        
        logger.info(f"{self.model_name} saved to {model_path}")
        return model_path
    
    def load(self, model_path: str) -> bool:
        """
        Load model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if loaded successfully
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.metrics = data['metrics']
            self.params = data['params']
            self.is_trained = data['is_trained']
            
            logger.info(f"{self.model_name} loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


# XGBoost Regressor
class XGBoostRegressor(BaseRegressor):
    """XGBoost implementation for view count prediction"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost regressor
        
        Args:
            params: XGBoost hyperparameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        default_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
            
        super().__init__('xgboost', default_params)
    
    def _create_model(self) -> Any:
        """Create and return the XGBoost model"""
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(**self.params)
        except ImportError:
            logger.error("XGBoost not installed. Please install it with: pip install xgboost")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for XGBoost model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        df.sort_values('importance', ascending=False, inplace=True)
        
        return df


# LightGBM Regressor
class LightGBMRegressor(BaseRegressor):
    """LightGBM implementation for view count prediction"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM regressor
        
        Args:
            params: LightGBM hyperparameters
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        default_params = {
            'objective': 'regression',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
            
        super().__init__('lightgbm', default_params)
    
    def _create_model(self) -> Any:
        """Create and return the LightGBM model"""
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self.params)
        except ImportError:
            logger.error("LightGBM not installed. Please install it with: pip install lightgbm")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for LightGBM model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        df.sort_values('importance', ascending=False, inplace=True)
        
        return df


# CatBoost Regressor
class CatBoostRegressor(BaseRegressor):
    """CatBoost implementation for view count prediction"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize CatBoost regressor
        
        Args:
            params: CatBoost hyperparameters
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available. Install with: pip install catboost")
        
        default_params = {
            'loss_function': 'RMSE',
            'learning_rate': 0.1,
            'depth': 6,
            'iterations': 100,
            'random_seed': 42,
            'verbose': False
        }
        
        if params:
            default_params.update(params)
            
        super().__init__('catboost', default_params)
    
    def _create_model(self) -> Any:
        """Create and return the CatBoost model"""
        try:
            from catboost import CatBoost
            return CatBoost(self.params)
        except ImportError:
            logger.error("CatBoost not installed. Please install it with: pip install catboost")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for CatBoost model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        df.sort_values('importance', ascending=False, inplace=True)
        
        return df


# TensorFlow Regressor
class TensorFlowRegressor(BaseRegressor):
    """TensorFlow Neural Network implementation for regression with standardized interface"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize TensorFlow regressor
        
        Args:
            params: TensorFlow model parameters
        """
        if not HAVE_TF:
            logger.warning("TensorFlow is not installed. This regressor will not work until TensorFlow is installed.")
        
        default_params = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32,
            'early_stopping_patience': 10,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
            
        super().__init__('tensorflow', default_params)
        
        # TF-specific attributes
        self.tfidf_vectorizer = None
        self.numerical_features = None
        self.text_features = None
        self.tfidf_max_features = 1000  # Default
        
    def _create_model(self, input_dim: int) -> Any:
        """
        Create TensorFlow model with the specified architecture
        
        Args:
            input_dim: Input dimension
            
        Returns:
            Sequential model
        """
        if not HAVE_TF:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
            
        # Set random seed for reproducibility
        tf.random.set_seed(self.params.get('random_state', 42))
        
        # Get architecture parameters
        hidden_layers = self.params.get('hidden_layers', [128, 64, 32])
        dropout_rate = self.params.get('dropout_rate', 0.3)
        learning_rate = self.params.get('learning_rate', 0.001)
        
        # Build model
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in hidden_layers:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        # Output layer (single neuron for regression)
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def _preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Preprocess features for the model
        
        Args:
            X: Features DataFrame
            fit: Whether to fit or just transform
            
        Returns:
            Preprocessed features as numpy array
        """
        # First call - identify numerical and text features
        if fit:
            self.numerical_features = []
            self.text_features = []
            
            for col in self.feature_names:
                if X[col].dtype == 'object' or X[col].dtype == 'string':
                    self.text_features.append(col)
                else:
                    self.numerical_features.append(col)
        
        # Ensure feature_names are set
        if not self.feature_names:
            self.feature_names = X.columns.tolist()
        
        # Extract numerical features
        numerical_data = X[self.numerical_features].values if self.numerical_features else np.array([])
        
        # Scale numerical features
        if len(numerical_data) > 0:
            if fit:
                numerical_scaled = self.scaler.fit_transform(numerical_data)
            else:
                numerical_scaled = self.scaler.transform(numerical_data)
        else:
            numerical_scaled = np.array([])
        
        # Process text features if available
        if self.text_features and len(self.text_features) > 0:
            # Combine text features
            text_data = X[self.text_features[0]].fillna('').astype(str)
            for feature in self.text_features[1:]:
                text_data = text_data + ' ' + X[feature].fillna('').astype(str)
            
            # TF-IDF vectorization
            if fit:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.params.get('tfidf_max_features', self.tfidf_max_features)
                )
                text_vectors = self.tfidf_vectorizer.fit_transform(text_data).toarray()
            else:
                if self.tfidf_vectorizer is None:
                    raise ValueError("TF-IDF vectorizer not fitted yet")
                text_vectors = self.tfidf_vectorizer.transform(text_data).toarray()
            
            # Combine with numerical features
            if len(numerical_scaled) > 0:
                X_processed = np.hstack([numerical_scaled, text_vectors])
            else:
                X_processed = text_vectors
        else:
            X_processed = numerical_scaled
        
        return X_processed
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str] = None) -> 'TensorFlowRegressor':
        """
        Train the TensorFlow model
        
        Args:
            X: Feature DataFrame
            y: Target values
            feature_names: Names of features
            
        Returns:
            Trained model
        """
        # Store feature names
        self.feature_names = feature_names or X.columns.tolist()
        
        # Preprocess features
        X_processed = self._preprocess_features(X, fit=True)
        
        # Create and compile model
        self.model = self._create_model(X_processed.shape[1])
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.params.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        )
        
        # Split data for validation if needed
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, 
            test_size=0.2, 
            random_state=self.params.get('random_state', 42)
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.params.get('epochs', 50),
            batch_size=self.params.get('batch_size', 32),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Preprocess features
        X_processed = self._preprocess_features(X, fit=False)
        
        # Make predictions
        return self.model.predict(X_processed).flatten()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (not directly available in TensorFlow)
        
        Returns:
            Dictionary with features and their importance
        """
        logger.warning("Feature importance is not directly available for TensorFlow neural networks")
        return {feature: 0.0 for feature in self.feature_names}
    
    def save(self, path: str) -> None:
        """
        Save model to file
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Create directory if needed
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save TensorFlow model
        tf_model_path = f"{path}_tf_model"
        self.model.save(tf_model_path)
        
        # Save metadata
        metadata = {
            'class': self.__class__.__name__,
            'params': self.params,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'text_features': self.text_features,
            'tfidf_max_features': self.tfidf_max_features
        }
        
        with open(f"{path}.meta", 'w') as f:
            json.dump(metadata, f)
        
        # Save preprocessors
        preprocessors = {
            'scaler': self.scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        
        with open(f"{path}.prep", 'wb') as f:
            pickle.dump(preprocessors, f)
        
        logger.info(f"TensorFlow model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TensorFlowRegressor':
        """
        Load model from file
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded model
        """
        # Load metadata
        with open(f"{path}.meta", 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(params=metadata.get('params', {}))
        instance.feature_names = metadata.get('feature_names')
        instance.numerical_features = metadata.get('numerical_features')
        instance.text_features = metadata.get('text_features')
        instance.tfidf_max_features = metadata.get('tfidf_max_features', 1000)
        
        # Load preprocessors
        with open(f"{path}.prep", 'rb') as f:
            preprocessors = pickle.load(f)
        
        instance.scaler = preprocessors.get('scaler')
        instance.tfidf_vectorizer = preprocessors.get('tfidf_vectorizer')
        
        # Load TensorFlow model
        tf_model_path = f"{path}_tf_model"
        instance.model = load_model(tf_model_path)
        
        logger.info(f"TensorFlow model loaded from {path}")
        return instance


# Ensemble Prediction
def create_ensemble_prediction(
    models: Dict[str, BaseRegressor],
    X: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Create an ensemble prediction by averaging predictions from multiple models
    
    Args:
        models: Dictionary of model name to model instance
        X: Features to predict on
        weights: Dictionary of model name to weight (optional)
        
    Returns:
        Array of predictions
    """
    if not models:
        raise ValueError("No models provided for ensemble prediction")
    
    # Get predictions from each model
    predictions = {}
    for name, model in models.items():
        if not model.is_trained:
            continue
        
        try:
            predictions[name] = model.predict(X)
        except Exception as e:
            logger.error(f"Error getting predictions from {name}: {str(e)}")
    
    if not predictions:
        raise RuntimeError("No valid predictions available for ensemble")
    
    # If weights not provided, use equal weights
    if not weights:
        weights = {name: 1.0 / len(predictions) for name in predictions}
    else:
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {name: weight / total for name, weight in weights.items()}
    
    # Create weighted average
    ensemble_pred = np.zeros(X.shape[0])
    for name, pred in predictions.items():
        if name in weights:
            ensemble_pred += pred * weights[name]
    
    return ensemble_pred


# Function to train and evaluate all models
def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[Dict[str, BaseRegressor], Dict[str, Dict[str, float]]]:
    """
    Train and evaluate all models
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data to use for testing
        random_state: Random state for train/test split
        params: Dictionary of model name to parameters
        
    Returns:
        Tuple of (models, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training data: {X_train.shape[0]} samples")
    logger.info(f"Testing data: {X_test.shape[0]} samples")
    
    # Default parameters
    if not params:
        params = {
            'xgboost': {},
            'lightgbm': {},
            'catboost': {},
            'tensorflow': {}
        }
    
    # Create models
    models = {
        'xgboost': XGBoostRegressor(params.get('xgboost', {})),
        'lightgbm': LightGBMRegressor(params.get('lightgbm', {})),
        'catboost': CatBoostRegressor(params.get('catboost', {})),
        'tensorflow': TensorFlowRegressor(params.get('tensorflow', {}))
    }
    
    # Train and evaluate models
    metrics = {}
    for name, model in models.items():
        try:
            # Train
            train_metrics = model.train(X_train, y_train)
            
            # Evaluate
            test_metrics = model.evaluate(X_test, y_test)
            
            # Store metrics
            metrics[name] = test_metrics
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
    
    # Return models and metrics
    return models, metrics


# Save all models
def save_models(models: Dict[str, BaseRegressor], model_dir: str) -> Dict[str, str]:
    """
    Save all models
    
    Args:
        models: Dictionary of model name to model instance
        model_dir: Directory to save models
        
    Returns:
        Dictionary of model name to model path
    """
    model_paths = {}
    for name, model in models.items():
        try:
            if model.is_trained:
                path = model.save(model_dir)
                model_paths[name] = path
        except Exception as e:
            logger.error(f"Error saving {name}: {str(e)}")
    
    return model_paths


# Load all models
def load_models(model_dir: str) -> Dict[str, BaseRegressor]:
    """
    Load all models from directory
    
    Args:
        model_dir: Directory with saved models
        
    Returns:
        Dictionary of model name to model instance
    """
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        return {}
    
    models = {}
    
    # Check for XGBoost
    xgb_path = os.path.join(model_dir, 'xgboost.pkl')
    if os.path.exists(xgb_path):
        xgb = XGBoostRegressor()
        if xgb.load(xgb_path):
            models['xgboost'] = xgb
    
    # Check for LightGBM
    lgb_path = os.path.join(model_dir, 'lightgbm.pkl')
    if os.path.exists(lgb_path):
        lgb = LightGBMRegressor()
        if lgb.load(lgb_path):
            models['lightgbm'] = lgb
    
    # Check for CatBoost
    cat_path = os.path.join(model_dir, 'catboost.pkl')
    if os.path.exists(cat_path):
        cat = CatBoostRegressor()
        if cat.load(cat_path):
            models['catboost'] = cat
    
    # Check for TensorFlow
    tf_path = os.path.join(model_dir, 'tensorflow.pkl')
    if os.path.exists(tf_path):
        tf = TensorFlowRegressor()
        if tf.load(tf_path):
            models['tensorflow'] = tf
    
    return models


class AdvancedRegressionModels:
    """
    Advanced ML models for predicting YouTube video view counts 
    using XGBoost, LightGBM, and CatBoost
    """
    
    def __init__(self):
        """Initialize the manager"""
        self.available_models = []
        
        if XGBOOST_AVAILABLE:
            self.available_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            self.available_models.append('lightgbm')
        if CATBOOST_AVAILABLE:
            self.available_models.append('catboost')
        if HAVE_TF:
            self.available_models.append('tensorflow')
        
        logger.info(f"Available models: {', '.join(self.available_models) if self.available_models else 'None'}")
        
        self.models = {}
    
    def create_model(self, model_type: str, params: Dict[str, Any] = None) -> BaseRegressor:
        """
        Create a model of the specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('xgboost', 'lightgbm', 'catboost', or 'tensorflow')
        params : dict, optional
            Model parameters
            
        Returns:
        --------
        BaseRegressor
            Created model
        """
        model_type = model_type.lower()
        
        if model_type == 'xgboost':
            return XGBoostRegressor(params)
        elif model_type == 'lightgbm':
            return LightGBMRegressor(params)
        elif model_type == 'catboost':
            return CatBoostRegressor(params)
        elif model_type == 'tensorflow':
            return TensorFlowRegressor(params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate random features
    X = np.random.rand(n_samples, n_features)
    
    # Add hour of day feature (0-23)
    hour_feature = np.random.randint(0, 24, size=n_samples)
    X = np.column_stack([X, hour_feature])
    
    # Generate target variable with noise
    # Make certain hours better for posting
    hour_coefficients = np.zeros(24)
    hour_coefficients[8:12] = 0.5  # Morning boost
    hour_coefficients[16:20] = 0.8  # Evening boost
    
    # Create formula for view count
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] - X[:, 4]
    
    # Add hour effect
    for i in range(n_samples):
        hour = int(X[i, -1])
        y[i] += hour_coefficients[hour] * 5
    
    # Add noise
    y += np.random.normal(0, 0.2, n_samples)
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)] + ['hour']
    df = pd.DataFrame(X, columns=feature_names)
    
    # Initialize models
    advanced_models = AdvancedRegressionModels()
    
    # Train models
    results = advanced_models.train(df, y, test_size=0.2, verbose=True)
    
    # Print results
    print("\nTraining Results:")
    print(f"Best model: {results['best_model']}")
    for model_name, metrics in results['metrics'].items():
        print(f"{model_name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    
    # Make predictions with each model
    test_data = pd.DataFrame(np.random.rand(5, n_features + 1), columns=feature_names)
    for model_name in advanced_models.models.keys():
        predictions = advanced_models.predict(test_data, model_name)
        print(f"\n{model_name} predictions:", predictions)
    
    # Get optimal posting time
    optimal_time = advanced_models.get_optimal_posting_time(df, time_col='hour')
    print("\nOptimal posting time:", optimal_time['optimal_time'])
    print("Predicted value:", optimal_time['predicted_value'])
    
    # Top 3 posting times
    sorted_times = list(optimal_time['all_predictions'].items())
    print("\nTop 3 posting times:")
    for time, value in sorted_times[:3]:
        print(f"Hour {time}: {value:.4f}")
    
    # Save models
    save_results = advanced_models.save_models(directory='../models')
    
    # Load models
    new_models = AdvancedRegressionModels()
    new_models.load_models(save_results['metadata_path'])
    
    # Verify loaded models
    loaded_perf = new_models.get_model_performance()
    print("\nLoaded model performance:")
    print(f"Best model: {loaded_perf['best_model']}")
    for model_name, metrics in loaded_perf['metrics'].items():
        print(f"{model_name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

# Create a singleton instance
advanced_models = AdvancedRegressionModels() 