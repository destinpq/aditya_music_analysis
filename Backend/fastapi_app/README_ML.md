# Advanced ML Models for YouTube Analytics

This module implements advanced regression models for predicting YouTube video metrics based on historical data. The implementation includes four industry-standard models:

1. XGBoost - Extreme Gradient Boosting
2. LightGBM - Light Gradient Boosting Machine
3. CatBoost - Gradient boosting on decision trees
4. TensorFlow - Deep Neural Network with multiple hidden layers

## Features

- Standardized interface for all regression models
- Hyperparameter tuning via grid search
- Model training, evaluation, and prediction
- Feature importance analysis
- Model persistence (save/load functionality)
- RESTful API endpoints for all operations

## Models

All models implement a common interface through the `BaseRegressor` class and include:

### Traditional Gradient Boosting Models
- **XGBoost** - Extreme Gradient Boosting
- **LightGBM** - Light Gradient Boosting Machine
- **CatBoost** - Gradient boosting on decision trees

### Deep Learning Model
- **TensorFlow Neural Network** - Deep neural network with 3 hidden layers and dropout regularization

## API Endpoints

### Training Models

#### Traditional Models (XGBoost, LightGBM, CatBoost)

**Endpoint:** `POST /api/ml/train`

Train regression models on a dataset.

**Parameters:**
- `dataset_id` (int): ID of the dataset to use for training
- `model_types` (list, optional): List of model types to train. If not provided, trains all available models.
- `target_column` (str): Target column to predict
- `feature_columns` (list, optional): Feature columns to use. If not provided, uses all columns except target.
- `test_size` (float, default=0.2): Proportion of data to use for testing

**Response:**
```json
{
  "status": "success",
  "message": "Models trained and saved",
  "models": ["xgboost", "lightgbm", "catboost"],
  "evaluation": {
    "xgboost": {
      "rmse": 0.123,
      "r2": 0.987
    },
    "lightgbm": {
      "rmse": 0.125,
      "r2": 0.985
    },
    "catboost": {
      "rmse": 0.120,
      "r2": 0.988
    }
  },
  "metadata": {...}
}
```

#### TensorFlow Neural Network

**Endpoint:** `POST /api/ml/tensorflow/train`

Train a TensorFlow neural network model on a dataset.

**Parameters:**
- `dataset_id` (int): ID of the dataset to use for training
- `target_column` (str): Target column to predict
- `feature_columns` (list, optional): Feature columns to use. If not provided, uses all columns except target.
- `test_size` (float, default=0.2): Proportion of data to use for testing
- `params` (object, optional): Custom parameters for the TensorFlow model

**Response:**
```json
{
  "status": "success",
  "message": "TensorFlow model trained and saved",
  "evaluation": {
    "rmse": 0.119,
    "r2": 0.989,
    "test_size": 4000
  },
  "metadata": {
    "dataset_id": 1,
    "target_column": "view_count",
    "feature_columns": [...],
    "params": {
      "hidden_layers": [128, 64, 32],
      "dropout_rate": 0.3,
      "learning_rate": 0.001,
      "epochs": 50,
      "batch_size": 32
    }
  }
}
```

### Hyperparameter Tuning

**Endpoint:** `POST /api/ml/tune`

Tune model hyperparameters using grid search.

**Parameters:**
- `dataset_id` (int): ID of the dataset to use for tuning
- `target_column` (str): Target column to predict
- `feature_columns` (list, optional): Feature columns to use. If not provided, uses all columns except target.
- `test_size` (float, default=0.2): Proportion of data to use for testing
- `n_splits` (int, default=3): Number of cross-validation splits

**Response:**
```json
{
  "status": "success",
  "message": "Hyperparameter tuning completed successfully",
  "results": {
    "dataset_id": 1,
    "target_column": "view_count",
    "feature_columns": [...],
    "n_splits": 3,
    "results": {
      "xgboost": {
        "params": {...},
        "metrics": {...}
      },
      "lightgbm": {
        "params": {...},
        "metrics": {...}
      },
      "catboost": {
        "params": {...},
        "metrics": {...}
      }
    }
  }
}
```

### Making Predictions

#### Traditional Models (XGBoost, LightGBM, CatBoost)

**Endpoint:** `POST /api/ml/predict`

Make predictions using trained models.

**Parameters:**
- `dataset_id` (int): ID of the dataset to use for prediction
- `target_column` (str): Target column to predict
- `feature_columns` (list, optional): Feature columns to use. If not provided, uses saved feature columns.
- `model_type` (str, optional): Model type to use for prediction. If not provided, uses all available models.
- `video_ids` (list, optional): List of video IDs to predict for. If not provided, predicts for all videos.

**Response:**
```json
{
  "video_predictions": [
    {
      "video_id": "abc123",
      "xgboost_prediction": 12345.67,
      "lightgbm_prediction": 12400.89,
      "catboost_prediction": 12500.12
    },
    {...}
  ],
  "model_metrics": {
    "xgboost": {...},
    "lightgbm": {...},
    "catboost": {...}
  }
}
```

#### TensorFlow Neural Network

**Endpoint:** `POST /api/ml/tensorflow/predict`

Make predictions using a trained TensorFlow model.

**Parameters:**
- `dataset_id` (int): ID of the dataset to use for prediction
- `target_column` (str): Target column to predict
- `feature_columns` (list, optional): Feature columns to use. If not provided, uses saved feature columns.
- `video_ids` (list, optional): List of video IDs to predict for. If not provided, predicts for all videos.

**Response:**
```json
{
  "video_predictions": [
    {
      "video_id": "abc123",
      "tensorflow_prediction": 12345.67
    },
    {...}
  ],
  "model_metrics": {
    "rmse": 0.119,
    "r2": 0.989,
    "test_size": 4000
  }
}
```

### List Trained Models

#### Traditional Models (XGBoost, LightGBM, CatBoost)

**Endpoint:** `GET /api/ml/models`

Get information about all trained traditional models.

**Response:**
```json
{
  "models": [
    {
      "dataset_id": 1,
      "target_column": "view_count",
      "feature_columns": [...],
      "models": ["xgboost", "lightgbm", "catboost"],
      "training_samples": 16000,
      "test_samples": 4000,
      "evaluation": {...},
      "has_tuning": true
    },
    {...}
  ],
  "message": "Found 2 trained model sets"
}
```

#### TensorFlow Models

**Endpoint:** `GET /api/ml/tensorflow/models`

Get information about all trained TensorFlow models.

**Response:**
```json
{
  "models": [
    {
      "dataset_id": 1,
      "target_column": "view_count",
      "feature_columns": [...],
      "training_samples": 16000,
      "test_samples": 4000,
      "evaluation": {...},
      "model_type": "tensorflow",
      "architecture": {
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3
      }
    },
    {...}
  ],
  "message": "Found 2 trained TensorFlow models"
}
```

### Delete Models

#### Traditional Models (XGBoost, LightGBM, CatBoost)

**Endpoint:** `DELETE /api/ml/models/{dataset_id}`

Delete trained models for a specific dataset.

**Response:**
```json
{
  "status": "success",
  "message": "Models for dataset 1 deleted successfully"
}
```

#### TensorFlow Model

**Endpoint:** `DELETE /api/ml/tensorflow/models/{dataset_id}`

Delete a trained TensorFlow model for a specific dataset.

**Response:**
```json
{
  "status": "success",
  "message": "TensorFlow model for dataset 1 deleted successfully"
}
```

## Usage Example

1. Upload a dataset using the existing upload endpoint
2. Train traditional models:
   ```bash
   curl -X POST "http://localhost:8000/api/ml/train?dataset_id=1&target_column=view_count"
   ```
3. Train a TensorFlow model:
   ```bash
   curl -X POST "http://localhost:8000/api/ml/tensorflow/train?dataset_id=1&target_column=view_count"
   ```
4. Perform hyperparameter tuning for traditional models:
   ```bash
   curl -X POST "http://localhost:8000/api/ml/tune?dataset_id=1&target_column=view_count"
   ```
5. Make predictions with traditional models:
   ```bash
   curl -X POST "http://localhost:8000/api/ml/predict?dataset_id=1&target_column=view_count"
   ```
6. Make predictions with TensorFlow model:
   ```bash
   curl -X POST "http://localhost:8000/api/ml/tensorflow/predict?dataset_id=1&target_column=view_count"
   ```

## Dependencies

The machine learning functionality requires the following Python packages:
- xgboost
- lightgbm
- catboost
- tensorflow
- scikit-learn
- numpy
- pandas
- joblib

These can be installed using pip:
```bash
pip install -r requirements.txt
``` 