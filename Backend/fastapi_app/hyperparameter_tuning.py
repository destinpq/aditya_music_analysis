"""
Simple hyperparameter tuning for regression models.
Provides utilities for grid search optimization of model hyperparameters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, ParameterGrid, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from itertools import product
import time
from sklearn.datasets import make_regression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define RMSE scorer
rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)

def create_parameter_grid(model_name: str, level: str = 'light') -> Dict[str, List[Any]]:
    """
    Create parameter grid for the specified model
    
    Args:
        model_name: Name of the model (xgboost, lightgbm, catboost)
        level: Intensity of hyperparameter search ('light', 'medium', 'exhaustive')
        
    Returns:
        Parameter grid dictionary
    """
    model_name = model_name.lower()
    
    if level == 'light':
        # Light tuning for quick results
        if model_name == 'xgboost':
            return {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }
        elif model_name == 'lightgbm':
            return {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }
        elif model_name == 'catboost':
            return {
                'learning_rate': [0.01, 0.1],
                'depth': [3, 6]
            }
    elif level == 'medium':
        # Medium tuning for better results
        if model_name == 'xgboost':
            return {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 6, 9],
                'n_estimators': [50, 100, 200],
                'subsample': [0.7, 0.9]
            }
        elif model_name == 'lightgbm':
            return {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 6, 9],
                'n_estimators': [50, 100, 200],
                'subsample': [0.7, 0.9]
            }
        elif model_name == 'catboost':
            return {
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [3, 6, 9],
                'iterations': [50, 100, 200]
            }
    elif level == 'exhaustive':
        # Exhaustive tuning for best results
        if model_name == 'xgboost':
            return {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9, 12],
                'n_estimators': [50, 100, 200, 300],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5]
            }
        elif model_name == 'lightgbm':
            return {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9, 12],
                'n_estimators': [50, 100, 200, 300],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
        elif model_name == 'catboost':
            return {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'depth': [3, 6, 9, 12],
                'iterations': [50, 100, 200, 300],
                'l2_leaf_reg': [1, 3, 5, 7],
                'random_strength': [0.1, 1, 10]
            }
    
    # Default case
    logger.warning(f"Unknown model '{model_name}' or level '{level}'. Using default parameters.")
    return {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6]
    }

def optimize_hyperparameters(
    model,
    X,
    y,
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    n_jobs: int = -1,
    use_randomized: bool = False,
    n_iter: int = 10,
    verbose: int = 1
) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters using grid search or randomized search
    
    Args:
        model: Model instance
        X: Training features
        y: Target variable
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        n_jobs: Number of jobs to run in parallel
        use_randomized: Whether to use randomized search
        n_iter: Number of iterations for randomized search
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_params, best_score)
    """
    if use_randomized:
        logger.info(f"Starting randomized search with {n_iter} iterations and {cv}-fold CV")
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=rmse_scorer,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=42
        )
    else:
        logger.info(f"Starting grid search with {cv}-fold CV")
        search = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=rmse_scorer,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose
        )
    
    # Fit search
    search.fit(X, y)
    
    # Get best parameters and score
    best_params = search.best_params_
    best_score = -search.best_score_  # Convert back to RMSE from negative RMSE
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best RMSE: {best_score:.4f}")
    
    return best_params, best_score

def get_optimal_params(
    model_name: str,
    X,
    y,
    level: str = 'medium',
    cv: int = 5,
    n_jobs: int = -1,
    use_randomized: bool = False
) -> Dict[str, Any]:
    """
    Get optimal parameters for a specific model
    
    Args:
        model_name: Name of the model
        X: Training features
        y: Target variable
        level: Intensity of hyperparameter search
        cv: Number of cross-validation folds
        n_jobs: Number of jobs to run in parallel
        use_randomized: Whether to use randomized search
        
    Returns:
        Dictionary of optimal parameters
    """
    # Create parameter grid
    param_grid = create_parameter_grid(model_name, level)
    
    # Create base model
    model = create_base_model(model_name)
    
    # Optimize hyperparameters
    best_params, _ = optimize_hyperparameters(
        model, X, y, param_grid, cv, n_jobs, use_randomized
    )
    
    return best_params

def create_base_model(model_name: str) -> Any:
    """
    Create base model instance
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'xgboost':
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(objective='reg:squarederror')
        except ImportError:
            logger.error("XGBoost not installed. Please install it with: pip install xgboost")
            raise
    elif model_name == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(objective='regression')
        except ImportError:
            logger.error("LightGBM not installed. Please install it with: pip install lightgbm")
            raise
    elif model_name == 'catboost':
        try:
            from catboost import CatBoost
            return CatBoost(loss_function='RMSE', verbose=False)
        except ImportError:
            logger.error("CatBoost not installed. Please install it with: pip install catboost")
            raise
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def grid_search(X: np.ndarray, y: np.ndarray, param_grid: Dict[str, List[Any]], 
                model_creator: Callable, n_splits: int = 5, random_state: int = 42,
                metric: str = 'rmse') -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    Perform grid search for hyperparameter tuning.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter values as values
    model_creator : callable
        Function that takes parameters and returns a model instance
    n_splits : int, optional
        Number of splits for cross-validation
    random_state : int, optional
        Random seed for reproducibility
    metric : str, optional
        Metric to optimize ('rmse' or 'r2')
        
    Returns:
    --------
    tuple
        Tuple with best parameters, best score, and metrics for the best parameters
    """
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    # Initialize variables to store results
    best_params = None
    best_score = float('inf') if metric == 'rmse' else float('-inf')
    best_metrics = None
    
    # Set up cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Start timer
    start_time = time.time()
    
    # Total number of combinations
    total_combinations = len(param_combinations)
    logger.info(f"Starting grid search with {total_combinations} parameter combinations")
    
    # Iterate over all parameter combinations
    for i, param_values in enumerate(param_combinations):
        # Create parameter dictionary
        params = {name: value for name, value in zip(param_names, param_values)}
        
        # Cross-validation scores
        cv_rmse = []
        cv_r2 = []
        
        # Perform cross-validation
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_creator(params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            cv_rmse.append(rmse)
            cv_r2.append(r2)
        
        # Calculate average metrics
        avg_rmse = np.mean(cv_rmse)
        avg_r2 = np.mean(cv_r2)
        
        # Check if this is the best model
        score = avg_rmse if metric == 'rmse' else -avg_r2  # Negative because we minimize
        if (metric == 'rmse' and score < best_score) or (metric == 'r2' and score > best_score):
            best_score = score
            best_params = params
            best_metrics = {'rmse': avg_rmse, 'r2': avg_r2}
        
        # Log progress
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time / (i + 1) * (total_combinations - i - 1)
        logger.info(f"Combination {i+1}/{total_combinations} - RMSE: {avg_rmse:.4f}, R2: {avg_r2:.4f} - "
                    f"Elapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s")
    
    logger.info(f"Grid search completed in {time.time() - start_time:.2f}s")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score ({metric}): {best_metrics[metric]:.4f}")
    
    return best_params, best_metrics[metric], best_metrics

def tune_xgboost(X: np.ndarray, y: np.ndarray, 
                 param_grid: Optional[Dict[str, List[Any]]] = None,
                 n_splits: int = 5, random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Tune XGBoost regressor.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    param_grid : dict, optional
        Dictionary with parameter names as keys and lists of parameter values as values
    n_splits : int, optional
        Number of splits for cross-validation
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        Tuple with best parameters and metrics for the best parameters
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not available. Install with: pip install xgboost")
        return None, None
    
    # Default parameter grid
    default_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # Use provided parameter grid or default
    param_grid = param_grid or default_param_grid
    
    # Define model creator function
    def create_xgboost(params):
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=random_state,
            **params
        )
    
    # Perform grid search
    best_params, _, best_metrics = grid_search(
        X, y, param_grid, create_xgboost,
        n_splits=n_splits, random_state=random_state
    )
    
    return best_params, best_metrics

def tune_lightgbm(X: np.ndarray, y: np.ndarray, 
                  param_grid: Optional[Dict[str, List[Any]]] = None,
                  n_splits: int = 5, random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Tune LightGBM regressor.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    param_grid : dict, optional
        Dictionary with parameter names as keys and lists of parameter values as values
    n_splits : int, optional
        Number of splits for cross-validation
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        Tuple with best parameters and metrics for the best parameters
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not available. Install with: pip install lightgbm")
        return None, None
    
    # Default parameter grid
    default_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100],
        'num_leaves': [31],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # Use provided parameter grid or default
    param_grid = param_grid or default_param_grid
    
    # Define model creator function
    def create_lightgbm(params):
        return lgb.LGBMRegressor(
            objective='regression',
            random_state=random_state,
            **params
        )
    
    # Perform grid search
    best_params, _, best_metrics = grid_search(
        X, y, param_grid, create_lightgbm,
        n_splits=n_splits, random_state=random_state
    )
    
    return best_params, best_metrics

def tune_catboost(X: np.ndarray, y: np.ndarray, 
                  param_grid: Optional[Dict[str, List[Any]]] = None,
                  n_splits: int = 5, random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Tune CatBoost regressor.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    param_grid : dict, optional
        Dictionary with parameter names as keys and lists of parameter values as values
    n_splits : int, optional
        Number of splits for cross-validation
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        Tuple with best parameters and metrics for the best parameters
    """
    try:
        import catboost as cb
    except ImportError:
        logger.error("CatBoost not available. Install with: pip install catboost")
        return None, None
    
    # Default parameter grid
    default_param_grid = {
        'depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'iterations': [100],
        'l2_leaf_reg': [3],
    }
    
    # Use provided parameter grid or default
    param_grid = param_grid or default_param_grid
    
    # Define model creator function
    def create_catboost(params):
        return cb.CatBoostRegressor(
            loss_function='RMSE',
            random_seed=random_state,
            verbose=0,
            **params
        )
    
    # Perform grid search
    best_params, _, best_metrics = grid_search(
        X, y, param_grid, create_catboost,
        n_splits=n_splits, random_state=random_state
    )
    
    return best_params, best_metrics

def tune_all_models(X: np.ndarray, y: np.ndarray, 
                    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
                    n_splits: int = 5, random_state: int = 42) -> Dict[str, Tuple[Dict[str, Any], Dict[str, float]]]:
    """
    Tune all available models.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    param_grids : dict, optional
        Dictionary with model names as keys and parameter grids as values
    n_splits : int, optional
        Number of splits for cross-validation
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with model names as keys and tuples with best parameters and metrics as values
    """
    param_grids = param_grids or {}
    results = {}
    
    # Tune XGBoost
    logger.info("Tuning XGBoost...")
    try:
        xgb_params, xgb_metrics = tune_xgboost(
            X, y,
            param_grid=param_grids.get('xgboost'),
            n_splits=n_splits,
            random_state=random_state
        )
        results['xgboost'] = (xgb_params, xgb_metrics)
    except Exception as e:
        logger.error(f"Error tuning XGBoost: {e}")
    
    # Tune LightGBM
    logger.info("Tuning LightGBM...")
    try:
        lgb_params, lgb_metrics = tune_lightgbm(
            X, y,
            param_grid=param_grids.get('lightgbm'),
            n_splits=n_splits,
            random_state=random_state
        )
        results['lightgbm'] = (lgb_params, lgb_metrics)
    except Exception as e:
        logger.error(f"Error tuning LightGBM: {e}")
    
    # Tune CatBoost
    logger.info("Tuning CatBoost...")
    try:
        cb_params, cb_metrics = tune_catboost(
            X, y,
            param_grid=param_grids.get('catboost'),
            n_splits=n_splits,
            random_state=random_state
        )
        results['catboost'] = (cb_params, cb_metrics)
    except Exception as e:
        logger.error(f"Error tuning CatBoost: {e}")
    
    # Log results
    logger.info("Tuning results:")
    for model_name, (params, metrics) in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Best parameters: {params}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  R2: {metrics['r2']:.4f}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # Tune all models with smaller parameter grids for demonstration
    param_grids = {
        'xgboost': {
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.3]
        },
        'lightgbm': {
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.3]
        },
        'catboost': {
            'depth': [3, 5],
            'learning_rate': [0.1, 0.3]
        }
    }
    
    # Tune models
    results = tune_all_models(X, y, param_grids=param_grids, n_splits=3)
    
    # Print best parameters and metrics for each model
    print("\nBest parameters and metrics for each model:")
    for model_name, (params, metrics) in results.items():
        print(f"{model_name}:")
        print(f"  Best parameters: {params}")
        print(f"  Validation RMSE: {metrics['rmse']:.4f}")
        print(f"  Validation R²: {metrics['r2']:.4f}") 