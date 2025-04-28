# Processed Datasets Directory

This directory stores processed datasets used for machine learning training. Each dataset is stored in its own subdirectory.

## Dataset Structure

Each dataset directory contains:

- `preprocessing.pkl`: Saved preprocessing state (feature columns, target columns, encoders)
- `X_train_*.csv`: Training features for different targets
- `X_test_*.csv`: Test features for different targets
- `y_train_*.csv`: Training target values
- `y_test_*.csv`: Test target values

## Target Variables

By default, three target variables are processed:

1. `view_count`: Number of video views
2. `like_count`: Number of video likes
3. `comment_count`: Number of video comments

## Features

Common features extracted from YouTube video data:

- `DurationSeconds`: Video duration in seconds
- `PublishMonth`: Month when video was published (1-12)
- `PublishDay`: Day of month when video was published (1-31)
- `PublishHour`: Hour when video was published (0-23)
- `PublishDayOfWeek`: Day of week (0=Monday, 6=Sunday)
- `IsWeekend`: Whether published on weekend (0 or 1)
- `TagCount`: Number of tags
- `Category_*`: One-hot encoded category features

## API Access

The processed datasets can be accessed via these API endpoints:

- `GET /api/datasets/processed`: List available processed datasets
- `GET /api/datasets/processed/{dataset_name}`: Get info about a specific dataset
- `POST /api/datasets/process`: Process a dataset from the database
- `POST /api/datasets/process-file`: Process an uploaded CSV file
- `POST /api/datasets/process-new-data`: Process new data using an existing dataset's preprocessing state 