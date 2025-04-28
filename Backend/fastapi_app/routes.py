from fastapi import APIRouter, Depends, HTTPException, Request, status, UploadFile, Form, File, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from database import SessionLocal, Dataset, VideoData, VideoStats, VideoEngagement, VideoMetaInfo, VideoTag
from sqlalchemy import func, desc
import schemas
from fastapi.responses import JSONResponse
import traceback
from ml_models import prediction_models
from earnings_calculator import earnings_calculator
from feature_processor import feature_processor
import pandas as pd
from dataset_processor import dataset_processor
import os
from advanced_ml_models import AdvancedRegressionModels
from hyperparameter_tuning import tune_all_models
import numpy as np
import tempfile
import json

router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={
        404: {"description": "Not found", "model": schemas.ErrorResponse},
        500: {"description": "Internal server error", "model": schemas.ErrorResponse}
    }
)

# Dependency
def get_db():
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail={"success": False, "message": "Database connection error", "error": str(e)}
        )
    finally:
        if db:
            db.close()

# Global instance of the models manager
ml_models_manager = AdvancedRegressionModels()
models_directory = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(models_directory, exist_ok=True)

@router.get("/health", response_model=schemas.HealthResponse)
async def health_check():
    try:
        return {"status": "ok", "message": "API is running"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": str(e)}
        )

@router.get("/datasets", response_model=List[schemas.Dataset])
async def get_datasets(db: Session = Depends(get_db)):
    try:
        datasets = db.query(Dataset).all()
        return [{"id": ds.id, "filename": ds.filename, "upload_date": ds.upload_date} for ds in datasets]
    except Exception as e:
        print(f"Error fetching datasets: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to fetch datasets", "error": str(e)}
        )

@router.get("/datasets/{dataset_id}/videos", response_model=List[schemas.VideoResponse])
async def get_videos_by_dataset(dataset_id: int, page: int = 1, limit: int = 100, sort_by: Optional[str] = None, sort_order: Optional[str] = "desc", db: Session = Depends(get_db)):
    try:
        # Check if dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "message": f"Dataset with ID {dataset_id} not found"}
            )
            
        offset = (page - 1) * limit
        query = db.query(VideoData).filter(VideoData.dataset_id == dataset_id)
        
        # Apply sorting if specified
        if sort_by:
            # Determine which table to sort from based on sort_by field
            if sort_by in ["view_count", "like_count", "dislike_count", "favorite_count", "comment_count"]:
                # Join with stats table for these fields
                query = query.join(VideoStats)
                sort_column = getattr(VideoStats, sort_by)
            elif sort_by in ["engagement_rate", "like_ratio"]:
                # Join with engagement table for these fields
                query = query.join(VideoEngagement)
                sort_column = getattr(VideoEngagement, sort_by)
            else:
                # Sort by VideoData fields (title, published_at, etc)
                if hasattr(VideoData, sort_by):
                    sort_column = getattr(VideoData, sort_by)
                else:
                    # Default to id if field doesn't exist
                    sort_column = VideoData.id
            
            # Apply sort direction
            if sort_order and sort_order.lower() == "asc":
                query = query.order_by(sort_column)
            else:
                query = query.order_by(desc(sort_column))
        
        videos = query.offset(offset).limit(limit).all()
        
        results = []
        for video in videos:
            # Get stats and engagement data
            stats = db.query(VideoStats).filter(VideoStats.video_id == video.id).first()
            engagement = db.query(VideoEngagement).filter(VideoEngagement.video_id == video.id).first()
            meta_info = db.query(VideoMetaInfo).filter(VideoMetaInfo.video_id == video.id).first()
            
            video_data = {
                "id": video.id,
                "video_id": video.video_id,
                "title": video.title,
                "published_at": video.published_at,
                "stats": {
                    "view_count": stats.view_count if stats else 0,
                    "like_count": stats.like_count if stats else 0,
                    "dislike_count": stats.dislike_count if stats else 0,
                    "favorite_count": stats.favorite_count if stats else 0,
                    "comment_count": stats.comment_count if stats else 0
                },
                "engagement": {
                    "engagement_rate": engagement.engagement_rate if engagement else 0,
                    "like_ratio": engagement.like_ratio if engagement else 0,
                    "comment_ratio": 0  # Added to match frontend interface
                },
                "meta_info": {
                    "duration": meta_info.duration if meta_info else 0,
                    "channel_id": meta_info.channel_id if meta_info else None,
                    "category_id": meta_info.category_id if meta_info else None,
                    "is_unlisted": meta_info.is_unlisted if meta_info else False
                },
                "tags": []  # Added empty tags array to match frontend interface
            }
            results.append(video_data)
            
        return results
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching videos: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to fetch videos", "error": str(e)}
        )

@router.get("/datasets/{dataset_id}/stats", response_model=schemas.DatasetStats)
async def get_dataset_stats(dataset_id: int, db: Session = Depends(get_db)):
    try:
        # Check if dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "message": f"Dataset with ID {dataset_id} not found"}
            )
            
        # Get total videos count
        total_videos = db.query(func.count(VideoData.id)).filter(
            VideoData.dataset_id == dataset_id
        ).scalar()
        
        # Get total views
        total_views = db.query(func.sum(VideoStats.view_count)).join(
            VideoData, VideoData.id == VideoStats.video_id
        ).filter(
            VideoData.dataset_id == dataset_id
        ).scalar() or 0
        
        # Get average engagement rate
        avg_engagement = db.query(func.avg(VideoEngagement.engagement_rate)).join(
            VideoData, VideoData.id == VideoEngagement.video_id
        ).filter(
            VideoData.dataset_id == dataset_id
        ).scalar() or 0
        
        # Get average like ratio
        avg_like_ratio = db.query(func.avg(VideoEngagement.like_ratio)).join(
            VideoData, VideoData.id == VideoEngagement.video_id
        ).filter(
            VideoData.dataset_id == dataset_id
        ).scalar() or 0
        
        # Get top 5 videos by views
        top_videos = db.query(
            VideoData.video_id,
            VideoData.title,
            VideoStats.view_count
        ).join(
            VideoStats, VideoData.id == VideoStats.video_id
        ).filter(
            VideoData.dataset_id == dataset_id
        ).order_by(
            desc(VideoStats.view_count)
        ).limit(5).all()
        
        top_videos_list = [
            {"video_id": vid[0], "title": vid[1], "views": vid[2]}
            for vid in top_videos
        ]
        
        return {
            "total_videos": total_videos,
            "total_views": total_views,
            "avg_engagement_rate": float(avg_engagement),
            "avg_like_ratio": float(avg_like_ratio),
            "top_videos": top_videos_list
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching dataset stats: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to fetch dataset statistics", "error": str(e)}
        )

@router.get("/search", response_model=List[schemas.VideoResponse])
async def search_videos(
    query: str, 
    dataset_id: Optional[int] = None, 
    limit: int = 50, 
    db: Session = Depends(get_db)
):
    try:
        base_query = db.query(VideoData)
        
        if dataset_id:
            base_query = base_query.filter(VideoData.dataset_id == dataset_id)
        
        videos = base_query.filter(
            VideoData.title.ilike(f"%{query}%")
        ).limit(limit).all()
        
        results = []
        for video in videos:
            # Get stats and engagement data
            stats = db.query(VideoStats).filter(VideoStats.video_id == video.id).first()
            engagement = db.query(VideoEngagement).filter(VideoEngagement.video_id == video.id).first()
            
            video_data = {
                "id": video.id,
                "video_id": video.video_id,
                "title": video.title,
                "published_at": video.published_at,
                "stats": {
                    "view_count": stats.view_count if stats else 0,
                    "like_count": stats.like_count if stats else 0,
                    "dislike_count": stats.dislike_count if stats else 0,
                    "favorite_count": stats.favorite_count if stats else 0,
                    "comment_count": stats.comment_count if stats else 0
                },
                "engagement": {
                    "engagement_rate": engagement.engagement_rate if engagement else 0,
                    "like_ratio": engagement.like_ratio if engagement else 0
                }
            }
            results.append(video_data)
            
        return results
    except Exception as e:
        print(f"Error searching videos: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to search videos", "error": str(e)}
        )

@router.get("/revenue/{dataset_id}", response_model=schemas.RevenueResponse)
async def get_revenue_analytics(
    dataset_id: int, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        # Check if dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "message": f"Dataset with ID {dataset_id} not found"}
            )
        
        # Get total views for revenue calculation
        query = db.query(
            VideoStats.view_count,
            VideoData.published_at
        ).join(
            VideoData, VideoData.id == VideoStats.video_id
        ).filter(
            VideoData.dataset_id == dataset_id
        )
        
        if start_date and end_date:
            query = query.filter(
                VideoData.published_at >= start_date,
                VideoData.published_at <= end_date
            )
        
        videos = query.all()
        
        # Calculate estimated revenue based on standard YouTube metrics by country
        country_rates = {
            "US": 0.00318,  # $3.18 per 1000 views
            "UK": 0.00276,  # $2.76 per 1000 views
            "CA": 0.00252,  # $2.52 per 1000 views
            "AU": 0.00224,  # $2.24 per 1000 views
            "DE": 0.00192,  # $1.92 per 1000 views
            "IN": 0.00050,  # $0.50 per 1000 views
            "Other": 0.00115  # $1.15 per 1000 views (average for other countries)
        }
        
        total_views = sum(video[0] for video in videos)
        
        # Simulate country distribution of views based on global YouTube demographic data
        country_distribution = {
            "US": 0.34,
            "UK": 0.08,
            "CA": 0.07,
            "AU": 0.05,
            "DE": 0.06,
            "IN": 0.25,
            "Other": 0.15
        }
        
        # Calculate revenue by country
        country_revenue = {}
        for country, rate in country_rates.items():
            country_views = int(total_views * country_distribution[country])
            revenue = (country_views * rate)
            country_revenue[country] = {
                "views": country_views,
                "revenue": round(revenue, 2),
                "rate_per_1000": round(rate * 1000, 2)
            }
        
        total_revenue = sum(data["revenue"] for data in country_revenue.values())
        
        # Calculate revenue over time (last 30 days or specified period)
        time_series_data = []
        
        # Simplified example for demo purposes
        # In a real implementation, this would aggregate by date from the database
        if videos:
            # Create sample time series (placeholder)
            from datetime import datetime, timedelta
            
            # Get the date range
            if start_date and end_date:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                # Default to last 30 days
                end = datetime.now()
                start = end - timedelta(days=30)
            
            # Generate daily data points
            current_date = start
            while current_date <= end:
                # Simulate random fluctuation in daily revenue
                import random
                day_factor = 0.8 + (random.random() * 0.4)  # 0.8 to 1.2 variation
                
                day_revenue = (total_revenue / 30) * day_factor
                
                time_series_data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "revenue": round(day_revenue, 2)
                })
                
                current_date += timedelta(days=1)
        
        return {
            "total_views": total_views,
            "total_revenue": round(total_revenue, 2),
            "country_revenue": country_revenue,
            "time_series": time_series_data
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating revenue: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to calculate revenue analytics", "error": str(e)}
        )

@router.post("/predict/ai-video", response_model=schemas.AIPrediction)
async def predict_ai_video_performance(prediction_request: schemas.AIPredictionRequest, db: Session = Depends(get_db)):
    try:
        # This endpoint simulates AI prediction based on historical data
        # In a real implementation, this would use a trained ML model
        
        # Get average stats from existing videos
        stats_query = db.query(
            func.avg(VideoStats.view_count).label("avg_views"),
            func.avg(VideoStats.like_count).label("avg_likes"),
            func.avg(VideoStats.comment_count).label("avg_comments"),
            func.avg(VideoEngagement.engagement_rate).label("avg_engagement")
        ).join(
            VideoData, VideoData.id == VideoStats.video_id
        ).join(
            VideoEngagement, VideoEngagement.video_id == VideoData.id
        )
        
        if prediction_request.dataset_id:
            stats_query = stats_query.filter(VideoData.dataset_id == prediction_request.dataset_id)
            
        baseline_stats = stats_query.first()
        
        if not baseline_stats or not baseline_stats.avg_views:
            # If no data is available, use placeholder values
            avg_views = 500000
            avg_likes = 25000
            avg_comments = 1500
            avg_engagement = 0.05
        else:
            avg_views = baseline_stats.avg_views or 0
            avg_likes = baseline_stats.avg_likes or 0
            avg_comments = baseline_stats.avg_comments or 0
            avg_engagement = baseline_stats.avg_engagement or 0
        
        # Apply content type multipliers
        content_multipliers = {
            "music_video": {"views": 1.8, "engagement": 1.5},
            "tutorial": {"views": 0.7, "engagement": 1.2},
            "vlog": {"views": 0.5, "engagement": 0.9},
            "short_form": {"views": 2.5, "engagement": 1.8}
        }
        
        multiplier = content_multipliers.get(
            prediction_request.content_type, 
            {"views": 1.0, "engagement": 1.0}
        )
        
        # Apply time factor (day of week, hour of day)
        # Weekends and evenings typically perform better
        day_multipliers = {
            "monday": 0.85,
            "tuesday": 0.9,
            "wednesday": 0.95,
            "thursday": 1.0,
            "friday": 1.1,
            "saturday": 1.3,
            "sunday": 1.2
        }
        
        hour_multipliers = {
            "morning": 0.8,      # 6-10 AM
            "midday": 0.9,       # 10 AM - 2 PM
            "afternoon": 1.0,    # 2-6 PM
            "evening": 1.3,      # 6-10 PM
            "night": 1.1,        # 10 PM - 2 AM
            "late_night": 0.7    # 2-6 AM
        }
        
        day_multiplier = day_multipliers.get(prediction_request.day_of_week.lower(), 1.0)
        hour_multiplier = hour_multipliers.get(prediction_request.time_of_day.lower(), 1.0)
        
        # Calculate predicted metrics
        predicted_views = avg_views * multiplier["views"] * day_multiplier * hour_multiplier
        predicted_likes = (avg_likes / avg_views) * predicted_views
        predicted_comments = (avg_comments / avg_views) * predicted_views
        predicted_engagement = avg_engagement * multiplier["engagement"]
        
        # Add some randomness to the prediction
        import random
        variance = 0.15  # 15% variance
        random_factor = 1.0 + (random.random() * variance * 2 - variance)
        
        predicted_views *= random_factor
        
        # Calculate revenue potential based on predicted views
        avg_cpm = 2.50  # Average CPM (cost per mille/thousand) in USD
        predicted_revenue = (predicted_views / 1000) * avg_cpm
        
        # Project growth over time
        time_series = []
        
        # Calculate cumulative views over 30 days
        cumulative_factor = 0
        daily_gains = [0.38, 0.15, 0.09, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015]
        daily_gains += [0.01] * 20  # Remaining 20 days with 1% gain each
        
        from datetime import datetime, timedelta
        current_date = datetime.now()
        
        for i in range(30):
            day_factor = daily_gains[min(i, len(daily_gains)-1)]
            cumulative_factor += day_factor
            day_views = predicted_views * cumulative_factor
            day_revenue = (day_views / 1000) * avg_cpm
            
            time_series.append({
                "day": i + 1,
                "date": (current_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "views": int(day_views),
                "revenue": round(day_revenue, 2)
            })
        
        return {
            "content_type": prediction_request.content_type,
            "predicted_views": int(predicted_views),
            "predicted_likes": int(predicted_likes),
            "predicted_comments": int(predicted_comments),
            "predicted_engagement_rate": round(predicted_engagement, 4),
            "predicted_revenue": round(predicted_revenue, 2),
            "optimal_posting": {
                "day": max(day_multipliers, key=day_multipliers.get),
                "time": max(hour_multipliers, key=hour_multipliers.get)
            },
            "growth_projection": time_series
        }
    except Exception as e:
        print(f"Error in AI prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to generate AI prediction", "error": str(e)}
        )

@router.delete("/datasets/{dataset_id}", response_model=schemas.SuccessResponse)
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    try:
        # Check if dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "message": f"Dataset with ID {dataset_id} not found"}
            )
        
        # Get all videos for this dataset
        videos = db.query(VideoData).filter(VideoData.dataset_id == dataset_id).all()
        
        # Delete all related data
        for video in videos:
            # Delete stats, engagement, meta info, and tags for each video
            db.query(VideoStats).filter(VideoStats.video_id == video.id).delete()
            db.query(VideoEngagement).filter(VideoEngagement.video_id == video.id).delete()
            db.query(VideoMetaInfo).filter(VideoMetaInfo.video_id == video.id).delete()
            db.query(VideoTag).filter(VideoTag.video_id == video.id).delete()
        
        # Delete all videos
        db.query(VideoData).filter(VideoData.dataset_id == dataset_id).delete()
        
        # Delete the dataset
        db.delete(dataset)
        db.commit()
        
        return {"success": True, "message": f"Dataset {dataset_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error deleting dataset: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to delete dataset", "error": str(e)}
        )

@router.post("/predict/video", response_model=Dict[str, Any])
async def predict_video_performance(prediction_input: schemas.PredictionInput, db: Session = Depends(get_db)):
    """
    Predict view count, like count, and comment count for a video
    """
    try:
        # Process input features
        processed_features = feature_processor.process_features(prediction_input.dict(), fit=False)
        
        # Check if models are trained
        if not prediction_models.is_trained:
            # Try to load pre-trained models
            try:
                prediction_models.load_models()
            except FileNotFoundError:
                # If no pre-trained models, train on sample data from database
                videos = db.query(VideoData).join(VideoStats).limit(1000).all()
                if not videos:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"success": False, "message": "No training data available. Please upload data first."}
                    )
                
                # Prepare training data
                train_data = []
                for video in videos:
                    stats = db.query(VideoStats).filter(VideoStats.video_id == video.id).first()
                    if stats:
                        video_data = {
                            "title": video.title,
                            "published_at": str(video.published_at),
                            "category_id": video.meta_info.category_id if video.meta_info else None,
                            "view_count": stats.view_count,
                            "like_count": stats.like_count,
                            "comment_count": stats.comment_count
                        }
                        train_data.append(video_data)
                
                if not train_data:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"success": False, "message": "No valid training data available."}
                    )
                
                # Process features and train models
                train_df = pd.DataFrame(train_data)
                processed_df = feature_processor.process_features(train_df, fit=True)
                prediction_models.train(processed_df)
                
                # Save models for future use
                feature_processor.save_processor()
                prediction_models.save_models()
        
        # Make predictions
        predictions = prediction_models.predict(processed_features)
        
        # Calculate estimated earnings
        earnings = earnings_calculator.calculate_earnings(predictions["predicted_views"])
        
        return {
            "success": True,
            "predictions": predictions,
            "earnings": earnings
        }
    except Exception as e:
        print(f"Error in video prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to predict video performance", "error": str(e)}
        )

@router.post("/earnings/calculate", response_model=Dict[str, Any])
async def calculate_video_earnings(earnings_input: schemas.EarningsInput):
    """
    Calculate estimated earnings for a video based on view count and CPM
    """
    try:
        earnings = earnings_calculator.calculate_earnings(
            view_count=earnings_input.view_count,
            custom_cpm=earnings_input.custom_cpm,
            country=earnings_input.country,
            geography=earnings_input.geography,
            monetization_rate=earnings_input.monetization_rate,
            ad_impression_rate=earnings_input.ad_impression_rate
        )
        
        return {
            "success": True,
            "earnings": earnings
        }
    except Exception as e:
        print(f"Error calculating earnings: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to calculate earnings", "error": str(e)}
        )

@router.get("/cpm/rates", response_model=Dict[str, Any])
async def get_cpm_rates(country: Optional[str] = None):
    """
    Get CPM rates for different countries
    """
    try:
        rates = earnings_calculator.get_cpm_rates(country)
        
        return {
            "success": True,
            "cpm_rates": rates
        }
    except Exception as e:
        print(f"Error getting CPM rates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to get CPM rates", "error": str(e)}
        )

@router.post("/ml/train", response_model=Dict[str, Any])
def train_ml_models(
    dataset_id: int,
    model_types: List[str] = Query(None, description="List of model types to train. If None, trains all available models."),
    target_column: str = Query(..., description="Target column to predict"),
    feature_columns: List[str] = Query(None, description="Feature columns to use. If None, uses all columns except target."),
    test_size: float = Query(0.2, ge=0.1, le=0.5, description="Proportion of data to use for testing"),
    db: Session = Depends(get_db)
):
    """Train ML models on a dataset"""
    try:
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Get videos
        videos = db.query(VideoData).filter(VideoData.dataset_id == dataset_id).all()
        if not videos:
            raise HTTPException(status_code=404, detail="No videos found for dataset")
        
        # Convert to DataFrame
        data = []
        for video in videos:
            row = {
                "video_id": video.video_id,
                "title": video.title,
                "published_at": video.published_at,
                "view_count": video.view_count,
                "like_count": video.like_count,
                "dislike_count": video.dislike_count,
                "favorite_count": video.favorite_count,
                "comment_count": video.comment_count,
                "engagement_rate": video.engagement_rate if hasattr(video, "engagement_rate") else None,
                "like_ratio": video.like_ratio if hasattr(video, "like_ratio") else None
            }
            # Add tags if available
            if hasattr(video, "tags") and video.tags:
                row["tags"] = video.tags
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Check if target column exists
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column and col != "video_id" and col != "title"]
        else:
            # Check if all feature columns exist
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(status_code=400, 
                                   detail=f"Feature columns {missing_columns} not found in dataset")
        
        # Clean data - remove rows with NaN in target or features
        df = df.dropna(subset=[target_column] + feature_columns)
        
        # Split data
        X = df[feature_columns].values
        y = df[target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train models
        trained_models = ml_models_manager.train_all(X_train, y_train, feature_names=feature_columns)
        
        # Evaluate models
        evaluation = ml_models_manager.evaluate_all(X_test, y_test)
        
        # Save models
        paths = ml_models_manager.save_models(models_directory)
        
        # Save metadata
        metadata = {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "test_size": test_size,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "model_paths": paths,
            "evaluation": evaluation
        }
        
        with open(os.path.join(models_directory, f"metadata_{dataset_id}.json"), "w") as f:
            json.dump(metadata, f)
        
        return {
            "status": "success",
            "message": f"Models trained and saved to {models_directory}",
            "models": list(trained_models.keys()),
            "evaluation": evaluation,
            "metadata": metadata
        }
    
    except Exception as e:
        logger.error(f"Error training ML models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training ML models: {str(e)}")

@router.post("/ml/tune", response_model=Dict[str, Any])
def tune_ml_models(
    dataset_id: int,
    target_column: str = Query(..., description="Target column to predict"),
    feature_columns: List[str] = Query(None, description="Feature columns to use. If None, uses all columns except target."),
    test_size: float = Query(0.2, ge=0.1, le=0.5, description="Proportion of data to use for testing"),
    n_splits: int = Query(3, ge=2, le=10, description="Number of cross-validation splits"),
    db: Session = Depends(get_db)
):
    """Tune hyperparameters for ML models on a dataset"""
    try:
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Get videos
        videos = db.query(VideoData).filter(VideoData.dataset_id == dataset_id).all()
        if not videos:
            raise HTTPException(status_code=404, detail="No videos found for dataset")
        
        # Convert to DataFrame (similar to train_ml_models)
        data = []
        for video in videos:
            row = {
                "video_id": video.video_id,
                "title": video.title,
                "published_at": video.published_at,
                "view_count": video.view_count,
                "like_count": video.like_count,
                "dislike_count": video.dislike_count,
                "favorite_count": video.favorite_count,
                "comment_count": video.comment_count,
                "engagement_rate": video.engagement_rate if hasattr(video, "engagement_rate") else None,
                "like_ratio": video.like_ratio if hasattr(video, "like_ratio") else None
            }
            # Add tags if available
            if hasattr(video, "tags") and video.tags:
                row["tags"] = video.tags
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Check if target column exists
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column and col != "video_id" and col != "title"]
        else:
            # Check if all feature columns exist
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(status_code=400, 
                                   detail=f"Feature columns {missing_columns} not found in dataset")
        
        # Clean data - remove rows with NaN in target or features
        df = df.dropna(subset=[target_column] + feature_columns)
        
        # Get data
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Define parameter grids for tuning
        param_grids = {
            'xgboost': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'lightgbm': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200],
                'num_leaves': [31, 63],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'catboost': {
                'depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'iterations': [100, 200],
                'l2_leaf_reg': [1, 3, 5]
            }
        }
        
        # Perform hyperparameter tuning
        tuning_results = tune_all_models(X, y, param_grids=param_grids, n_splits=n_splits)
        
        # Save tuning results
        tuning_metadata = {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "n_splits": n_splits,
            "results": {model: {"params": params, "metrics": metrics} 
                        for model, (params, metrics) in tuning_results.items()}
        }
        
        with open(os.path.join(models_directory, f"tuning_{dataset_id}.json"), "w") as f:
            json.dump(tuning_metadata, f)
        
        return {
            "status": "success",
            "message": "Hyperparameter tuning completed successfully",
            "results": tuning_metadata
        }
    
    except Exception as e:
        logger.error(f"Error tuning ML models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error tuning ML models: {str(e)}")

@router.post("/ml/predict", response_model=Dict[str, Any])
def predict_with_ml_models(
    dataset_id: int,
    target_column: str = Query(..., description="Target column to predict"),
    feature_columns: List[str] = Query(None, description="Feature columns to use. If None, uses saved feature columns."),
    model_type: str = Query(None, description="Model type to use for prediction. If None, uses all available models."),
    video_ids: List[str] = Query(None, description="List of video IDs to predict for. If None, predicts for all videos."),
    db: Session = Depends(get_db)
):
    """Make predictions using trained ML models"""
    try:
        # Check if models exist
        metadata_path = os.path.join(models_directory, f"metadata_{dataset_id}.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail=f"No trained models found for dataset {dataset_id}")
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Check if target column matches
        if metadata["target_column"] != target_column:
            raise HTTPException(status_code=400, 
                              detail=f"Target column '{target_column}' doesn't match trained model target '{metadata['target_column']}'")
        
        # Use saved feature columns if not specified
        if feature_columns is None:
            feature_columns = metadata["feature_columns"]
        else:
            # Check if features match trained model
            missing_features = [f for f in feature_columns if f not in metadata["feature_columns"]]
            if missing_features:
                raise HTTPException(status_code=400, 
                                  detail=f"Feature columns {missing_features} not used in trained model")
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Get videos
        query = db.query(VideoData).filter(VideoData.dataset_id == dataset_id)
        if video_ids:
            query = query.filter(VideoData.video_id.in_(video_ids))
        videos = query.all()
        
        if not videos:
            raise HTTPException(status_code=404, detail="No videos found for dataset")
        
        # Convert to DataFrame
        data = []
        for video in videos:
            row = {
                "video_id": video.video_id,
                "title": video.title,
                "published_at": video.published_at,
                "view_count": video.view_count,
                "like_count": video.like_count,
                "dislike_count": video.dislike_count,
                "favorite_count": video.favorite_count,
                "comment_count": video.comment_count,
                "engagement_rate": video.engagement_rate if hasattr(video, "engagement_rate") else None,
                "like_ratio": video.like_ratio if hasattr(video, "like_ratio") else None
            }
            # Add tags if available
            if hasattr(video, "tags") and video.tags:
                row["tags"] = video.tags
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Prepare feature matrix
        X = df[feature_columns].values
        video_ids = df["video_id"].values
        
        # Load models
        ml_models_manager.load_models(models_directory)
        
        # Make predictions
        predictions = ml_models_manager.predict(X, model_type=model_type)
        
        # Format results
        results = {"video_predictions": []}
        
        for i, video_id in enumerate(video_ids):
            video_result = {"video_id": video_id}
            for model_name, preds in predictions.items():
                video_result[f"{model_name}_prediction"] = float(preds[i])
            results["video_predictions"].append(video_result)
        
        # Add model evaluation metrics
        results["model_metrics"] = metadata["evaluation"]
        
        return results
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@router.get("/ml/models", response_model=Dict[str, Any])
def get_ml_models_info():
    """Get information about trained ML models"""
    try:
        # List all metadata files
        metadata_files = [f for f in os.listdir(models_directory) if f.startswith("metadata_")]
        
        if not metadata_files:
            return {"models": [], "message": "No trained models found"}
        
        models_info = []
        
        for metadata_file in metadata_files:
            with open(os.path.join(models_directory, metadata_file), "r") as f:
                metadata = json.load(f)
            
            dataset_id = metadata["dataset_id"]
            
            # Check if tuning was performed
            tuning_file = f"tuning_{dataset_id}.json"
            tuning_path = os.path.join(models_directory, tuning_file)
            has_tuning = os.path.exists(tuning_path)
            
            models_info.append({
                "dataset_id": dataset_id,
                "target_column": metadata["target_column"],
                "feature_columns": metadata["feature_columns"],
                "models": list(metadata["evaluation"].keys()),
                "training_samples": metadata["training_samples"],
                "test_samples": metadata["test_samples"],
                "evaluation": metadata["evaluation"],
                "has_tuning": has_tuning
            })
        
        return {
            "models": models_info,
            "message": f"Found {len(models_info)} trained model sets"
        }
    
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting models info: {str(e)}")

@router.delete("/ml/models/{dataset_id}", response_model=Dict[str, Any])
def delete_ml_models(dataset_id: int):
    """Delete ML models for a specific dataset"""
    try:
        # Check if models exist
        metadata_path = os.path.join(models_directory, f"metadata_{dataset_id}.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail=f"No trained models found for dataset {dataset_id}")
        
        # Load metadata to get model paths
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Delete model files
        for model_path in metadata["model_paths"].values():
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(f"{model_path}.meta"):
                os.remove(f"{model_path}.meta")
        
        # Delete metadata file
        os.remove(metadata_path)
        
        # Delete tuning file if exists
        tuning_path = os.path.join(models_directory, f"tuning_{dataset_id}.json")
        if os.path.exists(tuning_path):
            os.remove(tuning_path)
        
        return {
            "status": "success",
            "message": f"Models for dataset {dataset_id} deleted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error deleting models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting models: {str(e)}")

@router.post("/ml/tensorflow/train", response_model=Dict[str, Any])
def train_tensorflow_model(
    dataset_id: int,
    target_column: str = Query(..., description="Target column to predict"),
    feature_columns: List[str] = Query(None, description="Feature columns to use. If None, uses all columns except target."),
    test_size: float = Query(0.2, ge=0.1, le=0.5, description="Proportion of data to use for testing"),
    params: Dict[str, Any] = None,
    db: Session = Depends(get_db)
):
    """Train TensorFlow neural network model on a dataset"""
    try:
        # Check if TensorFlow is available
        from tensorflow.keras.models import Sequential
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Get videos
        videos = db.query(VideoData).filter(VideoData.dataset_id == dataset_id).all()
        if not videos:
            raise HTTPException(status_code=404, detail="No videos found for dataset")
        
        # Convert to DataFrame (similar to train_ml_models)
        data = []
        for video in videos:
            row = {
                "video_id": video.video_id,
                "title": video.title,
                "published_at": video.published_at,
                "view_count": video.view_count,
                "like_count": video.like_count,
                "dislike_count": video.dislike_count,
                "favorite_count": video.favorite_count,
                "comment_count": video.comment_count,
                "engagement_rate": video.engagement_rate if hasattr(video, "engagement_rate") else None,
                "like_ratio": video.like_ratio if hasattr(video, "like_ratio") else None,
                "duration_seconds": video.duration_seconds if hasattr(video, "duration_seconds") else 0,
                "tag_count": len(video.tags.split(",")) if hasattr(video, "tags") and video.tags else 0
            }
            
            # Add publish day, month, hour features if available
            if hasattr(video, "published_at") and video.published_at:
                row["publish_day"] = video.published_at.day
                row["publish_month"] = video.published_at.month
                row["publish_hour"] = video.published_at.hour
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Check if target column exists
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column and col != "video_id" and col != "title"]
        else:
            # Check if all feature columns exist
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(status_code=400, 
                                   detail=f"Feature columns {missing_columns} not found in dataset")
        
        # Clean data - remove rows with NaN in target or features
        df = df.dropna(subset=[target_column] + feature_columns)
        
        # Split data
        X = df[feature_columns]
        y = df[target_column].values
        
        # Default parameters for TensorFlow model
        default_params = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32,
            'early_stopping_patience': 10,
            'random_state': 42,
            'tfidf_max_features': 1000
        }
        
        # Update with provided parameters
        if params:
            default_params.update(params)
        
        # Create TensorFlow model
        from advanced_ml_models import TensorFlowRegressor
        tf_model = TensorFlowRegressor(params=default_params)
        
        # Train model
        tf_model.fit(X, y, feature_names=feature_columns)
        
        # Evaluate model
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        eval_metrics = tf_model.evaluate(X_test, y_test)
        
        # Save model
        model_path = os.path.join(models_directory, f"tensorflow_{dataset_id}")
        tf_model.save(model_path)
        
        # Save metadata
        metadata = {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "test_size": test_size,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "model_path": model_path,
            "evaluation": eval_metrics,
            "model_type": "tensorflow",
            "params": default_params
        }
        
        with open(os.path.join(models_directory, f"tensorflow_metadata_{dataset_id}.json"), "w") as f:
            json.dump(metadata, f)
        
        return {
            "status": "success",
            "message": f"TensorFlow model trained and saved to {model_path}",
            "evaluation": eval_metrics,
            "metadata": metadata
        }
    
    except ImportError:
        raise HTTPException(status_code=500, detail="TensorFlow is not installed. Install with: pip install tensorflow")
    except Exception as e:
        logger.error(f"Error training TensorFlow model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error training TensorFlow model: {str(e)}")

@router.post("/ml/tensorflow/predict", response_model=Dict[str, Any])
def predict_with_tensorflow(
    dataset_id: int,
    target_column: str = Query(..., description="Target column to predict"),
    feature_columns: List[str] = Query(None, description="Feature columns to use. If None, uses saved feature columns."),
    video_ids: List[str] = Query(None, description="List of video IDs to predict for. If None, predicts for all videos."),
    db: Session = Depends(get_db)
):
    """Make predictions using trained TensorFlow model"""
    try:
        # Check if TensorFlow model exists
        metadata_path = os.path.join(models_directory, f"tensorflow_metadata_{dataset_id}.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail=f"No trained TensorFlow model found for dataset {dataset_id}")
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Check if target column matches
        if metadata["target_column"] != target_column:
            raise HTTPException(status_code=400, 
                              detail=f"Target column '{target_column}' doesn't match trained model target '{metadata['target_column']}'")
        
        # Use saved feature columns if not specified
        if feature_columns is None:
            feature_columns = metadata["feature_columns"]
        else:
            # Check if features match trained model
            missing_features = [f for f in feature_columns if f not in metadata["feature_columns"]]
            if missing_features:
                raise HTTPException(status_code=400, 
                                  detail=f"Feature columns {missing_features} not used in trained model")
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Get videos
        query = db.query(VideoData).filter(VideoData.dataset_id == dataset_id)
        if video_ids:
            query = query.filter(VideoData.video_id.in_(video_ids))
        videos = query.all()
        
        if not videos:
            raise HTTPException(status_code=404, detail="No videos found for dataset")
        
        # Convert to DataFrame
        data = []
        for video in videos:
            row = {
                "video_id": video.video_id,
                "title": video.title,
                "published_at": video.published_at,
                "view_count": video.view_count,
                "like_count": video.like_count,
                "dislike_count": video.dislike_count,
                "favorite_count": video.favorite_count,
                "comment_count": video.comment_count,
                "engagement_rate": video.engagement_rate if hasattr(video, "engagement_rate") else None,
                "like_ratio": video.like_ratio if hasattr(video, "like_ratio") else None,
                "duration_seconds": video.duration_seconds if hasattr(video, "duration_seconds") else 0,
                "tag_count": len(video.tags.split(",")) if hasattr(video, "tags") and video.tags else 0
            }
            
            # Add publish day, month, hour features if available
            if hasattr(video, "published_at") and video.published_at:
                row["publish_day"] = video.published_at.day
                row["publish_month"] = video.published_at.month
                row["publish_hour"] = video.published_at.hour
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Select features
        X = df[feature_columns]
        video_ids = df["video_id"].values
        
        # Load TensorFlow model
        from advanced_ml_models import TensorFlowRegressor
        tf_model = TensorFlowRegressor()
        model_path = metadata["model_path"]
        
        # Check if model file exists
        if not os.path.exists(f"{model_path}.meta"):
            raise HTTPException(status_code=404, detail=f"TensorFlow model files not found at {model_path}")
        
        # Load model
        tf_model.load(model_path)
        
        # Make predictions
        predictions = tf_model.predict(X)
        
        # Format results
        results = {"video_predictions": []}
        
        for i, video_id in enumerate(video_ids):
            video_result = {
                "video_id": video_id,
                "tensorflow_prediction": float(predictions[i])
            }
            results["video_predictions"].append(video_result)
        
        # Add model evaluation metrics
        results["model_metrics"] = metadata["evaluation"]
        
        return results
    
    except ImportError:
        raise HTTPException(status_code=500, detail="TensorFlow is not installed. Install with: pip install tensorflow")
    except Exception as e:
        logger.error(f"Error making TensorFlow predictions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error making TensorFlow predictions: {str(e)}")

@router.get("/ml/tensorflow/models", response_model=Dict[str, Any])
def get_tensorflow_models_info():
    """Get information about trained TensorFlow models"""
    try:
        # List all TensorFlow metadata files
        metadata_files = [f for f in os.listdir(models_directory) if f.startswith("tensorflow_metadata_")]
        
        if not metadata_files:
            return {"models": [], "message": "No trained TensorFlow models found"}
        
        models_info = []
        
        for metadata_file in metadata_files:
            with open(os.path.join(models_directory, metadata_file), "r") as f:
                metadata = json.load(f)
            
            dataset_id = metadata["dataset_id"]
            
            models_info.append({
                "dataset_id": dataset_id,
                "target_column": metadata["target_column"],
                "feature_columns": metadata["feature_columns"],
                "training_samples": metadata["training_samples"],
                "test_samples": metadata["test_samples"],
                "evaluation": metadata["evaluation"],
                "model_type": "tensorflow",
                "architecture": {
                    "hidden_layers": metadata["params"]["hidden_layers"],
                    "dropout_rate": metadata["params"]["dropout_rate"]
                }
            })
        
        return {
            "models": models_info,
            "message": f"Found {len(models_info)} trained TensorFlow models"
        }
    
    except Exception as e:
        logger.error(f"Error getting TensorFlow models info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting TensorFlow models info: {str(e)}")

@router.delete("/ml/tensorflow/models/{dataset_id}", response_model=Dict[str, Any])
def delete_tensorflow_model(dataset_id: int):
    """Delete TensorFlow model for a specific dataset"""
    try:
        # Check if model exists
        metadata_path = os.path.join(models_directory, f"tensorflow_metadata_{dataset_id}.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail=f"No trained TensorFlow model found for dataset {dataset_id}")
        
        # Load metadata to get model path
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        model_path = metadata["model_path"]
        
        # Delete model files
        if os.path.exists(f"{model_path}.meta"):
            os.remove(f"{model_path}.meta")
        
        if os.path.exists(f"{model_path}.prep"):
            os.remove(f"{model_path}.prep")
        
        # Delete TensorFlow model directory if it exists
        tf_model_path = f"{model_path}_tf_model"
        if os.path.exists(tf_model_path):
            import shutil
            shutil.rmtree(tf_model_path)
        
        # Delete metadata file
        os.remove(metadata_path)
        
        return {
            "status": "success",
            "message": f"TensorFlow model for dataset {dataset_id} deleted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error deleting TensorFlow model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting TensorFlow model: {str(e)}")

@router.post("/datasets/process", response_model=Dict[str, Any])
async def process_dataset(
    request: schemas.DatasetProcessRequest,
    db: Session = Depends(get_db)
):
    """
    Process a dataset for machine learning training, splitting into train/test sets
    """
    try:
        # Check if dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "message": f"Dataset with ID {request.dataset_id} not found"}
            )
        
        # Get videos from dataset
        videos = db.query(VideoData).filter(VideoData.dataset_id == request.dataset_id).all()
        if not videos:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"success": False, "message": f"No videos found in dataset {request.dataset_id}"}
            )
        
        # Prepare data for processing
        data = []
        for video in videos:
            # Get stats
            stats = db.query(VideoStats).filter(VideoStats.video_id == video.id).first()
            
            # Get metadata
            meta_info = db.query(VideoMetaInfo).filter(VideoMetaInfo.video_id == video.id).first()
            
            # Get tags
            tags = db.query(VideoTag).filter(VideoTag.video_id == video.id).all()
            tags_str = ",".join([tag.tag_name for tag in tags]) if tags else ""
            
            # Combine data
            video_data = {
                "video_id": video.video_id,
                "title": video.title,
                "published_at": str(video.published_at),
                "duration": meta_info.duration if meta_info else 0,
                "category_id": meta_info.category_id if meta_info else "unknown",
                "tags": tags_str,
                "view_count": stats.view_count if stats else 0,
                "like_count": stats.like_count if stats else 0,
                "comment_count": stats.comment_count if stats else 0
            }
            data.append(video_data)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Process dataset
        result = dataset_processor.process_dataset(
            df=df,
            test_size=request.test_size,
            random_state=request.random_state,
            dataset_name=request.dataset_name
        )
        
        # Format response
        response = {
            "success": True,
            "dataset_name": result["dataset_name"],
            "features": {
                "feature_columns": result["features"],
                "target_columns": result["targets"],
                "num_samples": result["num_samples"],
                "num_features": result["num_features"]
            },
            "split": {
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "train_files": [f"X_train_{target}.csv" for target in result["targets"]],
                "test_files": [f"X_test_{target}.csv" for target in result["targets"]]
            }
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to process dataset", "error": str(e)}
        )

@router.get("/datasets/processed", response_model=Dict[str, Any])
async def list_processed_datasets():
    """
    Get list of available processed datasets
    """
    try:
        datasets = dataset_processor.get_available_datasets()
        
        return {
            "success": True,
            "datasets": datasets,
            "count": len(datasets)
        }
    except Exception as e:
        print(f"Error listing datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to list processed datasets", "error": str(e)}
        )

@router.get("/datasets/processed/{dataset_name}", response_model=Dict[str, Any])
async def get_processed_dataset_info(dataset_name: str):
    """
    Get information about a processed dataset
    """
    try:
        info = dataset_processor.get_dataset_info(dataset_name)
        
        if "error" in info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "message": info["error"]}
            )
        
        return {
            "success": True,
            **info
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting dataset info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to get dataset info", "error": str(e)}
        )

@router.post("/datasets/process-file", response_model=Dict[str, Any])
async def process_csv_file(
    file: UploadFile = File(...),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    dataset_name: str = Form("uploaded_dataset")
):
    """
    Process a CSV file and split into train/test sets
    """
    try:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{dataset_name}.csv"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Read CSV
        df = pd.read_csv(temp_file_path)
        
        # Process dataset
        result = dataset_processor.process_dataset(
            df=df,
            test_size=test_size,
            random_state=random_state,
            dataset_name=dataset_name
        )
        
        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        # Format response
        response = {
            "success": True,
            "dataset_name": result["dataset_name"],
            "features": {
                "feature_columns": result["features"],
                "target_columns": result["targets"],
                "num_samples": result["num_samples"],
                "num_features": result["num_features"]
            },
            "split": {
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "train_files": [f"X_train_{target}.csv" for target in result["targets"]],
                "test_files": [f"X_test_{target}.csv" for target in result["targets"]]
            }
        }
        
        return response
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to process CSV file", "error": str(e)}
        )

@router.post("/datasets/process-new-data", response_model=Dict[str, Any])
async def process_new_data(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
):
    """
    Process new data using existing dataset's preprocessing state
    """
    try:
        # Save uploaded file temporarily
        temp_file_path = f"temp_new_{dataset_name}.csv"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Read CSV
        df = pd.read_csv(temp_file_path)
        
        # Process using existing state
        processed_df = dataset_processor.process_new_data(df, dataset_name)
        
        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        # Save processed data
        output_path = f"processed_new_{dataset_name}.csv"
        processed_df.to_csv(output_path, index=False)
        
        return {
            "success": True,
            "message": "New data processed successfully",
            "dataset_name": dataset_name,
            "processed_file": output_path,
            "num_samples": processed_df.shape[0],
            "num_features": processed_df.shape[1],
            "features": processed_df.columns.tolist()
        }
    except Exception as e:
        print(f"Error processing new data: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to process new data", "error": str(e)}
        )

# Update the catch-all route to include new endpoints
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def api_catch_all(request: Request, path: str):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "success": False,
            "message": f"API endpoint '/api/{path}' not found",
            "status_code": status.HTTP_404_NOT_FOUND,
            "available_endpoints": [
                "/api/health",
                "/api/datasets",
                "/api/datasets/{dataset_id}/videos",
                "/api/datasets/{dataset_id}/stats",
                "/api/search",
                "/api/revenue/{dataset_id}",
                "/api/predict/ai-video",
                "/api/predict/video",
                "/api/earnings/calculate",
                "/api/cpm/rates",
                "/api/ml/train",
                "/api/ml/tune",
                "/api/ml/predict",
                "/api/ml/models",
                "/api/ml/models/{dataset_id}",
                "/api/ml/tensorflow/train",
                "/api/ml/tensorflow/predict",
                "/api/ml/tensorflow/models",
                "/api/ml/tensorflow/models/{dataset_id}",
                "/api/datasets/process",
                "/api/datasets/processed",
                "/api/datasets/processed/{dataset_name}",
                "/api/datasets/process-file",
                "/api/datasets/process-new-data"
            ]
        }
    ) 