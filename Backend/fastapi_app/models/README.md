# Machine Learning Models Directory

This directory stores trained machine learning models and feature processors for the YouTube Analytics API.

## Files

- `youtube_models.pkl`: Trained prediction models for views, likes, and comments
- `feature_processor.pkl`: Fitted feature processor for transforming raw YouTube metadata

## Model Types

1. **View Count Prediction Model**: RandomForest model trained to predict view counts
2. **Like Count Prediction Model**: GradientBoosting model trained to predict like counts
3. **Comment Count Prediction Model**: GradientBoosting model trained to predict comment counts

## Feature Processing

The feature processor transforms raw YouTube video metadata into machine learning features:

- Text features from video title (TF-IDF)
- Duration features (seconds, categories)
- Publishing time features (day of week, hour, time of day)
- Tag-based features (count, TF-IDF)
- Category encoding

## Usage

Models are automatically loaded when the API is started. New models can be trained via the `/api/train/models` endpoint.

## Earnings Calculator

Earnings are calculated based on predicted view counts and CPM (Cost Per Mille) rates for different countries, with special attention to India's market rates.

## API Documentation

See the main API documentation for information on how to use these models through the API endpoints:

- `/api/predict/video` - Predict performance metrics for a new video
- `/api/earnings/calculate` - Calculate estimated earnings
- `/api/cpm/rates` - Get CPM rates by country 