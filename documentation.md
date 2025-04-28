# Aditya Music Analytics Platform Documentation

## Overview

The Aditya Music Analytics Platform is a comprehensive solution for YouTube channel analytics, offering data visualization, machine learning predictions, and revenue analytics. The system consists of two main components:

1. **Frontend (adv-analytics)** - A Next.js application with modern UI built using Ant Design
2. **Backend (Backend/fastapi_app)** - A FastAPI server providing data processing and machine learning capabilities

This documentation focuses on the prediction functionality and what capabilities are available in the system.

## Prediction Capabilities

### Core Prediction Features

The platform offers several types of predictions:

1. **Video Performance Prediction** - Predict views, likes, and engagement for videos
2. **AI-Driven Content Recommendations** - Analyze best posting times and content types
3. **Revenue Forecasting** - Estimate potential earnings from videos
4. **Trend Analysis** - Identify trending topics and optimization opportunities

### Machine Learning Models

The platform uses four advanced ML models:

1. **XGBoost** - Extreme Gradient Boosting for high accuracy predictions
2. **LightGBM** - Light Gradient Boosting Machine for efficient processing
3. **CatBoost** - Gradient boosting on decision trees with superior handling of categorical features
4. **TensorFlow Neural Network** - Deep learning for complex pattern recognition

## How to Use Prediction Features

### 1. AI Video Performance Prediction

The AI prediction feature can be accessed from the Analytics Dashboard by clicking the "AI Prediction" button. It allows you to:

- Configure content type (Music Video, Tutorial, Vlog, Short Form)
- Select optimal day of week for publishing
- Choose best time of day for maximum reach
- Get comprehensive predictions including:
  - Predicted views, likes, and comments
  - Engagement rate estimates
  - Revenue potential calculations
  - 30-day growth projections

### 2. Dataset Analysis and ML Training

1. Upload a YouTube analytics CSV file through the Datasets page
2. Navigate to the ML Lab section
3. Select the dataset and configure:
   - Target metric (View Count, Engagement Rate, etc.)
   - Test set size for model validation
   - Machine learning models to use
4. Run model training or hyperparameter tuning
5. View performance metrics and analysis results

### 3. Making Predictions

1. From the ML Lab, select "Make Predictions" tab
2. Choose a trained model
3. Enter video details or upload a file with multiple videos
4. Get predictions for each video

## Detailed Step-by-Step Guide to All AI Features

### 1. AI Video Performance Prediction

**Access Path**: Analytics Dashboard → Dataset Details → AI Prediction Button

1. Navigate to the Analytics page and select a dataset
2. On the dataset details page, look for the blue "AI Prediction" button in the top right
3. In the prediction modal:
   - Select your content type (Music Video, Tutorial, Vlog, Short Form)
   - Choose the day of week for publishing (Monday-Sunday)
   - Select the time of day (Morning, Midday, Evening, etc.)
4. Click "Generate Prediction" to see:
   - Predicted views, likes, and comments
   - Expected engagement rate
   - Estimated revenue potential
   - Day-by-day growth projections for the next 30 days

### 2. ML Model Training

**Access Path**: ML Lab → Train Models Tab

1. Go to ML Lab from the main navigation
2. In the "Train Models" tab:
   - Select your dataset from the dropdown
   - Choose a target metric (View Count, Likes, Comments, Engagement Rate)
   - Adjust the test set size slider (typically 20% is good)
   - Select which ML models to use by clicking the selection cards
3. Click "Train Models" to start the training process
4. For advanced optimization, use "Run Hyperparameter Tuning" button

### 3. Making Custom Predictions

**Access Path**: ML Lab → Make Predictions Tab

1. Navigate to the ML Lab → "Make Predictions" tab
2. Select a trained model from the dropdown
3. Choose input method:
   - Manual Entry: Fill in video details directly
   - Upload File: Upload a CSV with multiple videos
4. For manual entry, complete all fields:
   - Video title
   - Duration
   - Upload day
   - Category
   - Tags
5. Click "Make Prediction" for single predictions or "Batch Predict" for files
6. Review results in the "Recent Predictions" table below

### 4. Dataset Processing with AI

**Access Path**: ML Lab → Process Datasets Tab

1. Go to ML Lab → "Process Datasets" tab
2. Select your source dataset
3. Enter a name for the processed dataset
4. Set a date range filter if needed
5. Configure column settings:
   - Select which columns to include
   - Choose how to handle missing values
6. Select processing options:
   - Remove duplicates
   - Normalize numeric values
   - Generate feature statistics
7. Click "Process Dataset" to create an enhanced dataset
8. Use "Preview Changes" to verify before processing

### 5. Advanced Features

1. **Feature Importance Analysis**:
   - After training models, view which features most impact predictions
   - Access through ML Lab → Train Models → After successful training

2. **Optimal Posting Schedule**:
   - Use AI prediction results to identify optimal publishing times
   - Available in the "Optimal Posting Time" section of prediction results

3. **Revenue Forecasting**:
   - Find in Analytics → Dataset Details → Revenue Analytics section
   - Click "AI Prediction" to get revenue forecasts

4. **A/B Test Simulation**:
   - ML Lab → Make Predictions → Enter multiple variations of video details
   - Compare predicted performance for each variation

Each of these features works with or without the backend API running - if the API is unavailable, the system uses fallback mock data to demonstrate functionality.

## Technical Implementation

### Frontend Components

The prediction functionality is implemented in:

- `app/analytics/[datasetId]/page.tsx` - Detailed analytics with AI prediction modal
- `app/ml-lab/page.tsx` - ML Lab with model training and prediction interfaces
- `app/components/ModelTrainer.tsx` - Component for ML model training

### Backend Services

The prediction functionality is powered by:

- `prediction_service.py` - Core service for making predictions
- `model_trainer.py` - Model training and evaluation functionality
- `advanced_ml_models.py` - Implementation of all ML models
- `hyperparameter_tuning.py` - Optimizing model performance 

### API Endpoints

#### Model Training
- `POST /api/ml/train` - Train ML models on dataset
- `POST /api/ml/tensorflow/train` - Train TensorFlow model
- `POST /api/ml/tune` - Perform hyperparameter tuning

#### Making Predictions
- `POST /api/ml/predict` - Get predictions from ML models
- `POST /api/ml/tensorflow/predict` - Get predictions from TensorFlow
- `POST /api/predict/ai-video/` - AI-driven video performance prediction

#### Model Management
- `GET /api/ml/models` - List all trained models
- `GET /api/ml/tensorflow/models` - List TensorFlow models
- `DELETE /api/ml/models/{dataset_id}` - Remove models
- `DELETE /api/ml/tensorflow/models/{dataset_id}` - Remove TensorFlow models

## Error Handling

The system includes robust error handling:

1. **API Connection Issues** - Fallback to mock prediction data when API is unavailable
2. **Missing Features** - Validation of required input parameters
3. **Model Loading Errors** - Automatic model retraining on failure

## Features by Page

### Dashboard

- Quick overview of dataset metrics
- Analytics cards with key performance indicators
- Direct access to detailed analytics

### Analytics

- In-depth dataset analysis
- Interactive charts and visualizations
- AI prediction capability via modal dialog
- Revenue analytics by country
- Time-based performance analysis

### ML Lab

1. **Train Models Tab**
   - Dataset selection
   - Target metric configuration
   - Test set size adjustment
   - Model selection
   - Training and hyperparameter tuning

2. **Make Predictions Tab**
   - Model selection
   - Manual data entry or file upload
   - Batch prediction capability
   - Results visualization

3. **Process Datasets Tab**
   - Data preprocessing options
   - Feature engineering
   - Dataset transformation and filtering

## Troubleshooting

Common issues and solutions:

1. **"Failed to load resource" error for prediction API**:
   - The system will automatically use mock data when the API endpoint is unavailable
   - Check that the FastAPI backend is running on port 1111
   - Verify API base URL in frontend configuration

2. **Model training fails**:
   - Ensure the dataset has sufficient data (minimum recommended: 1000 rows)
   - Check for missing values in important columns
   - Try with a smaller test size (e.g., 0.1 instead of 0.2)

3. **Slow prediction response**:
   - Large datasets may require more processing time
   - Consider using LightGBM for faster predictions with slightly lower accuracy
   - Implement data filtering to reduce dataset size

## Deployment Configuration

The frontend runs on port 2222 by default:
```
npm run dev -- -p 2222
```

The backend FastAPI server runs on port 1111:
```
uvicorn main:app --reload --port 1111
```

## Future Developments

Planned enhancements to prediction functionality:

1. Integration with real-time YouTube API data
2. Enhanced NLP for title and description analysis
3. Competitive channel analytics and benchmarking
4. Content scheduling optimization
5. Audience demographic prediction 