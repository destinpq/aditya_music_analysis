"""
Model Trainer Module for YouTube prediction models.
This module handles training, evaluation, and hyperparameter tuning of multiple models.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
from pathlib import Path

# Import models
from advanced_ml_models import XGBoostModel, LightGBMModel, CatBoostModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up results directory
RESULTS_DIR = Path(__file__).parent / "models" / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class ModelTrainer:
    """
    Model trainer for YouTube video view prediction models.
    Handles training, evaluation, and hyperparameter tuning for multiple models.
    """
    
    def __init__(self):
        """Initialize the model trainer with default models"""
        # Initialize models
        self.models = {
            "xgboost": XGBoostModel(name="yt_views_xgboost"),
            "lightgbm": LightGBMModel(name="yt_views_lightgbm"),
            "catboost": CatBoostModel(name="yt_views_catboost")
        }
        
        # Training results
        self.results = {}
        
        # Best model and its parameters
        self.best_model_name = None
        self.best_model_params = None
        self.best_model_score = float('inf')  # Lower RMSE is better
        
        # Model rankings
        self.model_rankings = []
    
    def load_data(self, data_dir: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load training and testing datasets from the specified directory
        
        Parameters:
        -----------
        data_dir : str
            Directory containing dataset files
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple containing X_train, X_test, y_train, y_test arrays
        """
        try:
            # Load data
            data_path = Path(data_dir) / "processed_youtube_data.csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Define features and target
            target_col = "view_count"
            
            # Drop columns that shouldn't be used as features
            drop_cols = ["video_id", "title", "description", target_col]
            feature_cols = [col for col in df.columns if col not in drop_cols]
            
            # Split features and target
            X = df[feature_cols]
            y = df[target_col].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            logger.info(f"Data loaded successfully. Training set: {X_train.shape}, Test set: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train_models(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                    X_val: Optional[pd.DataFrame] = None, 
                    y_val: Optional[np.ndarray] = None,
                    params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all models with given data and parameters
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature data
        y_train : np.ndarray
            Training target values
        X_val : pd.DataFrame, optional
            Validation feature data
        y_val : np.ndarray, optional
            Validation target values
        params : dict, optional
            Dictionary of model parameters (key: model_name)
            
        Returns:
        --------
        dict
            Training results for all models
        """
        results = {}
        model_metrics = []
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model...")
            
            # Get model-specific parameters (if provided)
            model_params = None
            if params and model_name in params:
                model_params = params[model_name]
            
            # Train the model
            try:
                metrics = model.train(
                    X_train, y_train, 
                    X_val=X_val, 
                    y_val=y_val,
                    params=model_params
                )
                
                # Save model
                model_path = model.save_model()
                logger.info(f"Saved {model_name} model to {model_path}")
                
                # Store results
                results[model_name] = {
                    "metrics": metrics,
                    "model_path": model_path
                }
                
                # Track for ranking
                val_rmse = metrics.get("rmse_val", metrics.get("rmse_train"))
                model_metrics.append({
                    "model_name": model_name,
                    "rmse": val_rmse,
                    "r2": metrics.get("r2_val", metrics.get("r2_train"))
                })
                
                # Update best model
                if val_rmse < self.best_model_score:
                    self.best_model_name = model_name
                    self.best_model_score = val_rmse
                    self.best_model_params = model_params
                
            except Exception as e:
                logger.error(f"Error training {model_name} model: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        # Rank models by RMSE (lower is better)
        self.model_rankings = sorted(model_metrics, key=lambda x: x["rmse"])
        logger.info(f"Best model: {self.best_model_name} with RMSE: {self.best_model_score:.4f}")
        
        # Save rankings
        self._save_results(results)
        
        return results
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                           X_val: pd.DataFrame, y_val: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Perform hyperparameter tuning for each model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature data
        y_train : np.ndarray
            Training target values
        X_val : pd.DataFrame
            Validation feature data
        y_val : np.ndarray
            Validation target values
            
        Returns:
        --------
        dict
            Best parameters for each model
        """
        # Hyperparameter grids for each model
        param_grids = {
            "xgboost": {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "n_estimators": [50, 100, 200]
            },
            "lightgbm": {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "num_leaves": [15, 31, 63]
            },
            "catboost": {
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6, 8],
                "iterations": [50, 100, 200]
            }
        }
        
        best_params = {}
        
        # Tune each model
        for model_name, model in self.models.items():
            logger.info(f"Tuning hyperparameters for {model_name} model...")
            
            # Get parameter grid
            param_grid = param_grids.get(model_name, {})
            if not param_grid:
                logger.warning(f"No parameter grid defined for {model_name}. Skipping tuning.")
                continue
            
            # Train and evaluate with each parameter combination
            best_rmse = float('inf')
            best_params_model = None
            
            # Simple grid search implementation
            # For production, use more sophisticated methods
            for learning_rate in param_grid.get("learning_rate", [0.1]):
                for max_depth in param_grid.get("max_depth", [6]):
                    for n_est in param_grid.get("n_estimators", [100]):
                        # Common parameters
                        params = {
                            "learning_rate": learning_rate,
                        }
                        
                        # Model-specific parameters
                        if model_name == "xgboost":
                            params["max_depth"] = max_depth
                            params["n_estimators"] = n_est
                        elif model_name == "lightgbm":
                            params["max_depth"] = max_depth
                            params["n_estimators"] = n_est
                            params["num_leaves"] = param_grid.get("num_leaves", [31])[0]
                        elif model_name == "catboost":
                            params["depth"] = max_depth
                            params["iterations"] = n_est
                        
                        # Train with these parameters
                        try:
                            metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val, params=params)
                            val_rmse = metrics.get("rmse_val", float('inf'))
                            
                            if val_rmse < best_rmse:
                                best_rmse = val_rmse
                                best_params_model = params.copy()
                                
                            logger.info(f"  {model_name} with params {params}: RMSE = {val_rmse:.4f}")
                        except Exception as e:
                            logger.error(f"Error tuning {model_name} with params {params}: {str(e)}")
            
            # Save best parameters for this model
            if best_params_model:
                best_params[model_name] = best_params_model
                logger.info(f"Best parameters for {model_name}: {best_params_model} (RMSE: {best_rmse:.4f})")
            else:
                logger.warning(f"No valid parameters found for {model_name}")
        
        return best_params
    
    def predict_with_all_models(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with all trained models
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data for prediction
            
        Returns:
        --------
        dict
            Predictions from each model and ensemble
        """
        predictions = {}
        
        # Predict with each model
        for model_name, model in self.models.items():
            try:
                if not model.is_trained:
                    model.load_model()  # Load from saved file
                
                pred = model.predict(X)
                predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
                predictions[model_name] = np.zeros(len(X))
        
        # Create ensemble prediction (simple average)
        if predictions:
            ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
            predictions["ensemble"] = ensemble_pred
        
        return predictions
    
    def get_optimal_posting_time(self, X: pd.DataFrame, 
                               time_feature: str = "hour_published", 
                               day_feature: str = "day_of_week",
                               tag_feature: str = "tags") -> Dict[str, Any]:
        """
        Determine best time to post videos for maximizing views
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data to use for optimization
        time_feature : str
            Column name for the hour of day feature
        day_feature : str
            Column name for the day of week feature
        tag_feature : str
            Column name for video tags
            
        Returns:
        --------
        dict
            Optimal posting times by day and tag category
        """
        # Check if best model exists
        if not self.best_model_name or self.best_model_name not in self.models:
            if "xgboost" in self.models:
                self.best_model_name = "xgboost"
            else:
                model_names = list(self.models.keys())
                if model_names:
                    self.best_model_name = model_names[0]
                else:
                    raise ValueError("No models available for prediction")
        
        best_model = self.models[self.best_model_name]
        
        # Ensure model is trained
        if not best_model.is_trained:
            try:
                best_model.load_model()
            except Exception as e:
                raise ValueError(f"Could not load model: {str(e)}")
        
        # Get unique days and hours
        days = sorted(X[day_feature].unique())
        hours = sorted(X[time_feature].unique())
        
        # Results container
        optimal_times = {
            "overall": {},
            "by_day": {},
            "by_tag": {}
        }
        
        # Find overall best time
        best_views = -1
        best_day = None
        best_hour = None
        
        # Test each day and hour combination
        for day in days:
            day_results = {}
            
            for hour in hours:
                # Create a copy of features with modified time
                X_temp = X.copy()
                X_temp[day_feature] = day
                X_temp[time_feature] = hour
                
                # Predict views
                pred_views = best_model.predict(X_temp)
                avg_views = pred_views.mean()
                
                # Store result for this hour
                day_results[int(hour)] = float(avg_views)
                
                # Update overall best
                if avg_views > best_views:
                    best_views = avg_views
                    best_day = day
                    best_hour = hour
            
            # Store results for this day
            optimal_times["by_day"][int(day)] = {
                "best_hour": int(sorted(day_results.items(), key=lambda x: x[1], reverse=True)[0][0]),
                "expected_views": float(max(day_results.values())),
                "hourly_views": day_results
            }
        
        # Set overall best
        optimal_times["overall"] = {
            "best_day": int(best_day),
            "best_hour": int(best_hour),
            "expected_views": float(best_views)
        }
        
        # If tag feature exists, find best times by tag
        if tag_feature in X.columns:
            # Get unique tags
            all_tags = []
            for tags_str in X[tag_feature].dropna().unique():
                if isinstance(tags_str, str):
                    tags = [tag.strip() for tag in tags_str.split(",")]
                    all_tags.extend(tags)
            
            unique_tags = list(set(all_tags))
            top_tags = sorted([(tag, all_tags.count(tag)) for tag in set(all_tags)], 
                             key=lambda x: x[1], reverse=True)[:10]
            
            # Find best time for each popular tag
            for tag, _ in top_tags:
                # Filter videos with this tag
                tag_mask = X[tag_feature].fillna("").str.contains(tag, case=False)
                
                if tag_mask.sum() > 0:
                    X_tag = X[tag_mask].copy()
                    
                    # Find best day and hour
                    best_tag_views = -1
                    best_tag_day = None
                    best_tag_hour = None
                    
                    for day in days:
                        for hour in hours:
                            X_temp = X_tag.copy()
                            X_temp[day_feature] = day
                            X_temp[time_feature] = hour
                            
                            pred_views = best_model.predict(X_temp)
                            avg_views = pred_views.mean()
                            
                            if avg_views > best_tag_views:
                                best_tag_views = avg_views
                                best_tag_day = day
                                best_tag_hour = hour
                    
                    # Store results for this tag
                    optimal_times["by_tag"][tag] = {
                        "best_day": int(best_tag_day),
                        "best_hour": int(best_tag_hour),
                        "expected_views": float(best_tag_views)
                    }
        
        return optimal_times
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save training and evaluation results to disk
        
        Parameters:
        -----------
        results : dict
            Training results to save
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results for saving (ensure serializable)
        save_results = {}
        for model_name, model_results in results.items():
            save_results[model_name] = {}
            
            for k, v in model_results.items():
                # Convert non-serializable values
                if isinstance(v, dict):
                    save_results[model_name][k] = {
                        sk: (float(sv) if isinstance(sv, np.number) else sv)
                        for sk, sv in v.items()
                    }
                else:
                    save_results[model_name][k] = v
        
        # Add rankings
        save_results["rankings"] = self.model_rankings
        save_results["best_model"] = {
            "name": self.best_model_name,
            "score": float(self.best_model_score) if isinstance(self.best_model_score, np.number) else self.best_model_score,
            "params": self.best_model_params
        }
        
        # Save to file
        results_path = RESULTS_DIR / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        logger.info(f"Saved training results to {results_path}")


# Create singleton instance
model_trainer = ModelTrainer() 