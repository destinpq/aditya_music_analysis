"""
API endpoints for YouTube view count prediction models.
Exposes functionality for predictions, recommendations, and model analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np

from database import get_db
from prediction_service import PredictionService
from routes import get_videos_by_dataset

# Setup router
router = APIRouter(prefix="/api/ml", tags=["ML Models"])

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize prediction service as a global variable
prediction_service = PredictionService()

# ----- Pydantic Models -----

class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    features: Dict[str, Any] = Field(..., description="Features for prediction")
    model_name: Optional[str] = Field(None, description="Model to use (xgboost, lightgbm, catboost, or all)")
    use_ensemble: bool = Field(False, description="Whether to use ensemble prediction")

class TimeRecommendationRequest(BaseModel):
    """Request model for time recommendations"""
    base_features: Dict[str, Any] = Field(..., description="Base features for the video")
    time_features: List[str] = Field(..., description="Names of time-related features")
    time_range: Optional[List[int]] = Field(None, description="Range of hours to consider [start, end]")
    top_n: int = Field(3, description="Number of top recommendations to return")

class DayRecommendationRequest(BaseModel):
    """Request model for day recommendations"""
    base_features: Dict[str, Any] = Field(..., description="Base features for the video")
    day_features: List[str] = Field(..., description="Names of day-related features")
    top_n: int = Field(3, description="Number of top recommendations to return")

class TagAnalysisRequest(BaseModel):
    """Request model for tag impact analysis"""
    base_features: Dict[str, Any] = Field(..., description="Base features for the video")
    tag_features: List[str] = Field(..., description="Names of tag-related features")
    top_n: int = Field(5, description="Number of top tags to return")

class DatasetFeaturesRequest(BaseModel):
    """Request model for extracting features from dataset"""
    dataset_id: int = Field(..., description="Dataset ID to extract features from")
    video_id: Optional[str] = Field(None, description="Specific video ID to extract features for")
    include_categorical: bool = Field(True, description="Whether to include categorical features")
    include_time_features: bool = Field(True, description="Whether to include time features")
    include_tag_features: bool = Field(True, description="Whether to include tag features")

# ----- Helper Functions -----

def get_video_features(db: Session, dataset_id: int, video_id: Optional[str] = None):
    """
    Extract features from a specific video or create sample features
    """
    # Get videos from the specified dataset
    videos = get_videos_by_dataset(db, dataset_id)
    if not videos:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found or empty")
    
    # If a specific video ID is provided, find that video
    if video_id:
        for video in videos:
            if video.video_id == video_id:
                # Extract features for this specific video
                features = {
                    "duration_seconds": video.duration_seconds,
                    "category_id": video.category_id,
                    "title_length": len(video.title or ""),
                    "description_length": len(video.description or ""),
                    "tag_count": len(video.tags.split(",")) if video.tags else 0,
                }
                
                # Add time features
                if video.published_at:
                    features.update({
                        "hour_published": video.published_at.hour,
                        "day_published": video.published_at.weekday(),
                        "month_published": video.published_at.month,
                        "is_weekend": 1 if video.published_at.weekday() >= 5 else 0,
                        "is_morning": 1 if 5 <= video.published_at.hour < 12 else 0,
                        "is_afternoon": 1 if 12 <= video.published_at.hour < 17 else 0,
                        "is_evening": 1 if 17 <= video.published_at.hour < 21 else 0,
                        "is_night": 1 if video.published_at.hour >= 21 or video.published_at.hour < 5 else 0,
                    })
                
                return features
        
        # If we get here, the video was not found
        raise HTTPException(status_code=404, detail=f"Video with ID {video_id} not found in dataset {dataset_id}")
    
    # If no specific video is provided, create sample features from the first video
    sample_video = videos[0]
    features = {
        "duration_seconds": sample_video.duration_seconds,
        "category_id": sample_video.category_id,
        "title_length": len(sample_video.title or ""),
        "description_length": len(sample_video.description or ""),
        "tag_count": len(sample_video.tags.split(",")) if sample_video.tags else 0,
    }
    
    # Add time features (with zeros for all)
    features.update({
        "hour_published": 0,
        "day_published": 0,
        "month_published": 0,
        "is_weekend": 0,
        "is_morning": 0,
        "is_afternoon": 0,
        "is_evening": 0,
        "is_night": 0,
    })
    
    return features

# ----- API Endpoints -----

@router.get("/models", response_model=List[str])
async def get_available_models():
    """Get list of available trained models"""
    # Load models if not loaded
    if not prediction_service.loaded:
        prediction_service.load_models()
    
    return list(prediction_service.models.keys())

@router.post("/predict")
async def predict_views(request: PredictionRequest):
    """
    Make predictions using trained models
    """
    try:
        # Convert dict to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        predictions = prediction_service.predict(
            features=features_df,
            model_name=request.model_name,
            use_ensemble=request.use_ensemble
        )
        
        return {
            "predictions": predictions,
            "features_used": list(request.features.keys())
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-importance")
async def get_feature_importance(model_name: Optional[str] = None):
    """
    Get feature importance for trained models
    """
    try:
        importance_data = prediction_service.analyze_feature_importance(model_name)
        
        # Convert DataFrame to list of dicts for JSON serialization
        result = {}
        for model, df in importance_data.items():
            result[model] = df.to_dict(orient="records")
        
        return result
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend-time")
async def recommend_posting_time(request: TimeRecommendationRequest):
    """
    Recommend optimal posting times based on predicted view counts
    """
    try:
        # Convert dict to DataFrame
        features_df = pd.DataFrame([request.base_features])
        
        # Get recommendations
        recommendations = prediction_service.recommend_posting_time(
            video_features=features_df,
            time_features=request.time_features,
            time_range=request.time_range,
            top_n=request.top_n
        )
        
        return {
            "recommendations": recommendations,
            "features_used": list(request.base_features.keys())
        }
    except Exception as e:
        logger.error(f"Error recommending posting time: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend-day")
async def recommend_posting_day(request: DayRecommendationRequest):
    """
    Recommend optimal posting days based on predicted view counts
    """
    try:
        # Convert dict to DataFrame
        features_df = pd.DataFrame([request.base_features])
        
        # Get recommendations
        recommendations = prediction_service.recommend_posting_day(
            video_features=features_df,
            day_features=request.day_features,
            top_n=request.top_n
        )
        
        return {
            "recommendations": recommendations,
            "features_used": list(request.base_features.keys())
        }
    except Exception as e:
        logger.error(f"Error recommending posting day: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-tags")
async def analyze_tag_impact(request: TagAnalysisRequest):
    """
    Analyze the impact of different tags on predicted view counts
    """
    try:
        # Convert dict to DataFrame
        features_df = pd.DataFrame([request.base_features])
        
        # Get tag impact analysis
        tag_impacts = prediction_service.analyze_tag_impact(
            base_features=features_df,
            tag_features=request.tag_features,
            top_n=request.top_n
        )
        
        # Convert to list format for better presentation
        tag_list = [{"tag": tag, "impact": impact} for tag, impact in tag_impacts.items()]
        
        return {
            "tag_impacts": tag_list,
            "features_used": list(request.base_features.keys())
        }
    except Exception as e:
        logger.error(f"Error analyzing tag impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dataset-features")
async def get_dataset_features(
    request: DatasetFeaturesRequest,
    db: Session = Depends(get_db)
):
    """
    Extract features from a dataset for use with the models
    """
    try:
        # Get basic features from a video in the dataset
        features = get_video_features(db, request.dataset_id, request.video_id)
        
        # Group features by type
        result = {
            "base_features": {k: v for k, v in features.items() 
                              if not k.startswith(("hour_", "day_", "is_", "tag_"))},
        }
        
        # Add time features if requested
        if request.include_time_features:
            result["time_features"] = [k for k in features.keys() 
                                      if k.startswith(("hour_", "is_morning", "is_afternoon", "is_evening", "is_night"))]
        
        # Add day features if requested
        if request.include_time_features:
            result["day_features"] = [k for k in features.keys() 
                                     if k.startswith(("day_", "is_weekend", "is_weekday"))]
        
        # Add tag features if requested and available
        if request.include_tag_features:
            # Get videos from the dataset
            videos = get_videos_by_dataset(db, request.dataset_id)
            
            # Extract unique tags from videos
            all_tags = set()
            for video in videos:
                if video.tags:
                    tags = [tag.strip() for tag in video.tags.split(',')]
                    all_tags.update(tags)
            
            # Add tag features
            result["tag_features"] = sorted(list(all_tags))
        
        return result
    except Exception as e:
        logger.error(f"Error extracting dataset features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}/analysis")
async def analyze_dataset(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Analyze a dataset and provide insights
    """
    try:
        # Get videos from the dataset
        videos = get_videos_by_dataset(db, dataset_id)
        if not videos:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found or empty")
        
        # Extract basic metrics
        view_counts = [v.view_count for v in videos]
        like_counts = [v.like_count for v in videos]
        comment_counts = [v.comment_count for v in videos]
        
        # Calculate engagement metrics
        engagement_rates = [(v.like_count + v.comment_count) / v.view_count if v.view_count > 0 else 0 
                           for v in videos]
        
        # Group by category
        categories = {}
        for video in videos:
            cat_id = video.category_id
            if cat_id not in categories:
                categories[cat_id] = []
            categories[cat_id].append(video)
        
        # Calculate category metrics
        category_metrics = {}
        for cat_id, cat_videos in categories.items():
            cat_views = [v.view_count for v in cat_videos]
            cat_likes = [v.like_count for v in cat_videos]
            cat_comments = [v.comment_count for v in cat_videos]
            cat_engagement = [(v.like_count + v.comment_count) / v.view_count if v.view_count > 0 else 0 
                             for v in cat_videos]
            
            category_metrics[cat_id] = {
                "count": len(cat_videos),
                "avg_views": np.mean(cat_views).item() if cat_views else 0,
                "avg_likes": np.mean(cat_likes).item() if cat_likes else 0,
                "avg_comments": np.mean(cat_comments).item() if cat_comments else 0,
                "avg_engagement": np.mean(cat_engagement).item() if cat_engagement else 0
            }
        
        # Group by publish hour
        hour_groups = {}
        for video in videos:
            if not video.published_at:
                continue
            
            hour = video.published_at.hour
            if hour not in hour_groups:
                hour_groups[hour] = []
            hour_groups[hour].append(video)
        
        # Calculate hour metrics
        hour_metrics = {}
        for hour, hour_videos in hour_groups.items():
            hour_views = [v.view_count for v in hour_videos]
            hour_engagement = [(v.like_count + v.comment_count) / v.view_count if v.view_count > 0 else 0 
                              for v in hour_videos]
            
            hour_metrics[hour] = {
                "count": len(hour_videos),
                "avg_views": np.mean(hour_views).item() if hour_views else 0,
                "avg_engagement": np.mean(hour_engagement).item() if hour_engagement else 0
            }
        
        # Group by publish day
        day_groups = {}
        for video in videos:
            if not video.published_at:
                continue
            
            day = video.published_at.weekday()
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append(video)
        
        # Calculate day metrics
        day_metrics = {}
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day, day_videos in day_groups.items():
            day_views = [v.view_count for v in day_videos]
            day_engagement = [(v.like_count + v.comment_count) / v.view_count if v.view_count > 0 else 0 
                             for v in day_videos]
            
            day_metrics[day_names[day]] = {
                "count": len(day_videos),
                "avg_views": np.mean(day_views).item() if day_views else 0,
                "avg_engagement": np.mean(day_engagement).item() if day_engagement else 0
            }
        
        return {
            "dataset_metrics": {
                "video_count": len(videos),
                "avg_views": np.mean(view_counts).item() if view_counts else 0,
                "median_views": np.median(view_counts).item() if view_counts else 0,
                "avg_likes": np.mean(like_counts).item() if like_counts else 0,
                "avg_comments": np.mean(comment_counts).item() if comment_counts else 0,
                "avg_engagement": np.mean(engagement_rates).item() if engagement_rates else 0
            },
            "category_metrics": category_metrics,
            "hour_metrics": hour_metrics,
            "day_metrics": day_metrics
        }
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Check health of the ML service"""
    return {
        "status": "operational",
        "models_loaded": prediction_service.loaded,
        "model_count": len(prediction_service.models) if prediction_service.loaded else 0
    } 