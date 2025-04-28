from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
import os
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import base64
import re
import urllib.parse
from googleapiclient.discovery import build
import tempfile
import shutil
import uvicorn
from fastapi.routing import APIRoute
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from sqlalchemy import func, desc
from sqlalchemy.sql import select, and_, or_

# Import database models and functions
import database
from database import SessionLocal, engine, Base
from database import Dataset, VideoData, VideoStats, VideoEngagement, VideoMetaInfo, VideoTag
from database import init_db, store_dataset
import schemas
import routes  # Import the routes module

# Ensure database directory exists
os.makedirs("database", exist_ok=True)

# Create all tables using database models
database.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app with increased file size limit
app = FastAPI(
    title="YouTube Analytics API",
    description="API for analyzing YouTube video performance and statistics",
    version="1.0.0"
)

# Set debug mode - change to False in production
app.debug = True

# Set upload file size limit
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Content-Disposition"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Global exception handlers
@app.exception_handler(StarletteHTTPException)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: Union[StarletteHTTPException, HTTPException]):
    # Get a list of available API endpoints if this is a 404 error
    available_endpoints = []
    if exc.status_code == 404:
        available_endpoints = get_available_api_endpoints(app)
    
    # Determine if this is an API request based on the URL path
    path = request.url.path
    is_api_request = path.startswith("/api") or "api/" in path
    
    # For API requests, always include available endpoints on 404
    # For non-API requests, only include endpoints if explicitly requested
    should_include_endpoints = is_api_request or (
        "include_endpoints" in request.query_params and 
        request.query_params["include_endpoints"].lower() in ("true", "1", "yes")
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "status_code": exc.status_code,
            "request_method": request.method,
            "request_url": str(request.url),
            "available_endpoints": available_endpoints if exc.status_code == 404 and should_include_endpoints else [],
            "documentation_url": "/docs" if exc.status_code == 404 else None
        },
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors and return a structured response with detailed error information.
    """
    errors = []
    for error in exc.errors():
        error_info = {
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        }
        errors.append(error_info)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "Validation error in request data",
            "errors": errors,
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "request_method": request.method,
            "request_url": str(request.url),
            "documentation_url": "/docs"
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "An unexpected error occurred",
            "error": str(exc),
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )

# Dependency
def get_db():
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection error")
    finally:
        if db:
            db.close()

@app.on_event("startup")
async def startup():
    database.init_db()

# Replace template-based endpoints with JSON responses
@app.get("/")
async def read_root():
    return {"message": "Welcome to YouTube Analytics API", "docs": "/docs"}

# Upload file endpoint
@app.post("/upload/", response_model=schemas.Dataset)
async def upload_file(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    print(f"========== UPLOAD REQUEST RECEIVED ==========")
    print(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    # Log client connection information
    request_client = f"Client: {request.client.host}:{request.client.port}"
    print(f"Upload request from {request_client}")
    print(f"Request headers: {request.headers}")
    
    # Create a temporary file
    suffix = os.path.splitext(file.filename)[1]
    
    try:
        # Create temp file with a reasonable buffer size
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            # Process in larger chunks to handle big files efficiently
            chunk_size = 1024 * 1024 * 2  # 2MB chunks
            total_bytes = 0
            
            # Copy the file in chunks
            while chunk := await file.read(chunk_size):
                temp.write(chunk)
                total_bytes += len(chunk)
                print(f"Read {total_bytes} bytes so far")
            
            temp_path = temp.name
        
        print(f"Temp file created at: {temp_path} - Total size: {total_bytes} bytes")
        print("Starting CSV processing...")
        
        try:
            # Process and store the dataset
            df = pd.read_csv(temp_path)
            row_count = len(df)
            print(f"CSV loaded with {row_count} rows")
            
            print("Starting database insertion...")
            dataset = database.store_dataset(db, df, file.filename)
            print(f"Dataset created with ID: {dataset.id}")
            
            # Add row_count to the response
            result = {
                "id": dataset.id, 
                "filename": dataset.filename, 
                "upload_date": dataset.upload_date,
                "row_count": row_count
            }
            print(f"Returning successful response: {result}")
            return result
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            import traceback
            print(f"Stack trace: ", traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")
    except Exception as e:
        print(f"Error handling upload: {str(e)}")
        import traceback
        print(f"Stack trace: ", traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up the temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"Temp file deleted: {temp_path}")
        print(f"========== UPLOAD REQUEST COMPLETED ==========")

# Download report endpoint
@app.get("/download_report")
async def download_report():
    report_path = "reports/latest_report.pdf"
    if os.path.exists(report_path):
        return FileResponse(path=report_path, filename="youtube_analytics_report.pdf")
    else:
        return {"error": "Report not found"}

# API endpoints for data analysis
@app.get("/api/day_analysis")
async def day_analysis(db: Session = Depends(get_db)):
    # Implement day analysis logic here
    return {"message": "Day analysis endpoint"}

@app.get("/api/time_analysis")
async def time_analysis(db: Session = Depends(get_db)):
    # Implement time analysis logic here
    return {"message": "Time analysis endpoint"}

@app.get("/api/tag_analysis")
async def tag_analysis(db: Session = Depends(get_db)):
    # Implement tag analysis logic here
    return {"message": "Tag analysis endpoint"}

# Additional API endpoints for sheet data
@app.get("/get_sheet_data")
async def get_sheet_data():
    # Implement sheet data retrieval logic here
    return {"message": "Sheet data endpoint"}

@app.post("/analyze_sheet_url")
async def analyze_sheet_url(url: str = Form(...), db: Session = Depends(get_db)):
    # Implement sheet URL analysis logic here
    return {"message": "Sheet URL analysis endpoint"}

@app.get("/datasets/", response_model=List[schemas.Dataset])
def get_datasets(db: Session = Depends(get_db)):
    datasets = db.query(database.Dataset).all()
    return datasets

@app.get("/api/datasets", response_model=List[schemas.Dataset])
def get_api_datasets(db: Session = Depends(get_db)):
    try:
        datasets = db.query(database.Dataset).all()
        return datasets
    except Exception as e:
        print(f"Error fetching datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Failed to fetch datasets", "error": str(e)}
        )

@app.get("/datasets/{dataset_id}/videos/", response_model=List[Dict[str, Any]])
def get_videos_by_dataset(
    dataset_id: int, 
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = "desc",
    limit: Optional[int] = None,
    db: Session = Depends(get_db)
):
    # Start with a query that joins all the tables
    query = db.query(database.VideoData).\
        filter(database.VideoData.dataset_id == dataset_id)
    
    # Apply sorting if requested
    if sort_by:
        # Handle different sort fields
        if sort_by == "view_count":
            # Join with VideoStats and sort by view_count
            query = query.join(database.VideoStats, database.VideoData.id == database.VideoStats.video_id)
            if sort_order.lower() == "asc":
                query = query.order_by(database.VideoStats.view_count)
            else:
                query = query.order_by(desc(database.VideoStats.view_count))
        
        elif sort_by == "like_count":
            # Join with VideoStats and sort by like_count
            query = query.join(database.VideoStats, database.VideoData.id == database.VideoStats.video_id)
            if sort_order.lower() == "asc":
                query = query.order_by(database.VideoStats.like_count)
            else:
                query = query.order_by(desc(database.VideoStats.like_count))
        
        elif sort_by == "comment_count":
            # Join with VideoStats and sort by comment_count
            query = query.join(database.VideoStats, database.VideoData.id == database.VideoStats.video_id)
            if sort_order.lower() == "asc":
                query = query.order_by(database.VideoStats.comment_count)
            else:
                query = query.order_by(desc(database.VideoStats.comment_count))
        
        elif sort_by == "engagement_rate":
            # Join with VideoEngagement and sort by engagement_rate
            query = query.join(database.VideoEngagement, database.VideoData.id == database.VideoEngagement.video_id)
            if sort_order.lower() == "asc":
                query = query.order_by(database.VideoEngagement.engagement_rate)
            else:
                query = query.order_by(desc(database.VideoEngagement.engagement_rate))
        
        elif sort_by == "like_ratio":
            # Join with VideoEngagement and sort by like_ratio
            query = query.join(database.VideoEngagement, database.VideoData.id == database.VideoEngagement.video_id)
            if sort_order.lower() == "asc":
                query = query.order_by(database.VideoEngagement.like_ratio)
            else:
                query = query.order_by(desc(database.VideoEngagement.like_ratio))
        
        elif sort_by == "duration":
            # Join with VideoMetaInfo and sort by duration
            query = query.join(database.VideoMetaInfo, database.VideoData.id == database.VideoMetaInfo.video_id)
            if sort_order.lower() == "asc":
                query = query.order_by(database.VideoMetaInfo.duration)
            else:
                query = query.order_by(desc(database.VideoMetaInfo.duration))
        
        elif sort_by == "published_at":
            # Sort by published_at
            if sort_order.lower() == "asc":
                query = query.order_by(database.VideoData.published_at)
            else:
                query = query.order_by(desc(database.VideoData.published_at))
        
        elif sort_by == "title":
            # Sort by title
            if sort_order.lower() == "asc":
                query = query.order_by(database.VideoData.title)
            else:
                query = query.order_by(desc(database.VideoData.title))
    
    # Apply limit if specified
    if limit:
        query = query.limit(limit)
    
    # Execute query to get videos
    videos = query.all()
    
    result = []
    for video in videos:
        stats = db.query(database.VideoStats).filter(database.VideoStats.video_id == video.id).first()
        engagement = db.query(database.VideoEngagement).filter(database.VideoEngagement.video_id == video.id).first()
        meta_info = db.query(database.VideoMetaInfo).filter(database.VideoMetaInfo.video_id == video.id).first()
        tags = db.query(database.VideoTag).filter(database.VideoTag.video_id == video.id).all()
        
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
                "comment_count": stats.comment_count if stats else 0,
            },
            "engagement": {
                "engagement_rate": engagement.engagement_rate if engagement else 0.0,
                "like_ratio": engagement.like_ratio if engagement else 0.0,
                "comment_ratio": engagement.comment_ratio if engagement else 0.0,
            },
            "meta_info": {
                "duration": meta_info.duration if meta_info else 0,
                "channel_id": meta_info.channel_id if meta_info else None,
                "category_id": meta_info.category_id if meta_info else None,
                "is_unlisted": meta_info.is_unlisted if meta_info else False,
            },
            "tags": [tag.tag_name for tag in tags]
        }
        result.append(video_data)
    
    return result

@app.get("/videos/{video_id}", response_model=Dict[str, Any])
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(database.VideoData).filter(database.VideoData.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    stats = db.query(database.VideoStats).filter(database.VideoStats.video_id == video.id).first()
    engagement = db.query(database.VideoEngagement).filter(database.VideoEngagement.video_id == video.id).first()
    meta_info = db.query(database.VideoMetaInfo).filter(database.VideoMetaInfo.video_id == video.id).first()
    tags = db.query(database.VideoTag).filter(database.VideoTag.video_id == video.id).all()
    
    return {
        "id": video.id,
        "video_id": video.video_id,
        "title": video.title,
        "published_at": video.published_at,
        "stats": {
            "view_count": stats.view_count if stats else 0,
            "like_count": stats.like_count if stats else 0,
            "dislike_count": stats.dislike_count if stats else 0,
            "favorite_count": stats.favorite_count if stats else 0,
            "comment_count": stats.comment_count if stats else 0,
        },
        "engagement": {
            "engagement_rate": engagement.engagement_rate if engagement else 0.0,
            "like_ratio": engagement.like_ratio if engagement else 0.0,
            "comment_ratio": engagement.comment_ratio if engagement else 0.0,
        },
        "meta_info": {
            "duration": meta_info.duration if meta_info else 0,
            "channel_id": meta_info.channel_id if meta_info else None,
            "category_id": meta_info.category_id if meta_info else None,
            "is_unlisted": meta_info.is_unlisted if meta_info else False,
        },
        "tags": [tag.tag_name for tag in tags]
    }

@app.get("/analysis/{dataset_id}", response_model=schemas.AnalysisResults)
def analyze_dataset(dataset_id: int, db: Session = Depends(get_db)):
    # Check if dataset exists
    dataset = db.query(database.Dataset).filter(database.Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    print(f"Analyzing dataset id={dataset_id}, filename={dataset.filename}")
    
    # Get all videos for the dataset
    videos = db.query(database.VideoData).filter(database.VideoData.dataset_id == dataset_id).all()
    video_ids = [video.id for video in videos]
    
    if not video_ids:
        raise HTTPException(status_code=404, detail="No videos found in dataset")
    
    print(f"Found {len(video_ids)} videos in dataset")
    
    # Debug: Check if stats records exist
    stats_count = db.query(database.VideoStats).filter(database.VideoStats.video_id.in_(video_ids)).count()
    print(f"Found {stats_count} stats records")
    
    # Get aggregate stats
    stats_query = db.query(
        func.avg(database.VideoStats.view_count).label('avg_views'),
        func.avg(database.VideoStats.like_count).label('avg_likes'),
        func.avg(database.VideoStats.comment_count).label('avg_comments'),
        func.sum(database.VideoStats.view_count).label('total_views'),
        func.sum(database.VideoStats.like_count).label('total_likes'),
        func.sum(database.VideoStats.comment_count).label('total_comments')
    ).filter(database.VideoStats.video_id.in_(video_ids)).first()
    
    print(f"Stats query results: {stats_query}")
    
    # Get aggregate engagement
    engagement_query = db.query(
        func.avg(database.VideoEngagement.engagement_rate).label('avg_engagement'),
        func.avg(database.VideoEngagement.like_ratio).label('avg_like_ratio'),
        func.avg(database.VideoEngagement.comment_ratio).label('avg_comment_ratio')
    ).filter(database.VideoEngagement.video_id.in_(video_ids)).first()
    
    print(f"Engagement query results: {engagement_query}")
    
    # Get top performing videos by views
    top_videos_by_views = db.query(
        database.VideoData.title,
        database.VideoStats.view_count
    ).join(
        database.VideoStats, database.VideoData.id == database.VideoStats.video_id
    ).filter(
        database.VideoData.id.in_(video_ids)
    ).order_by(
        desc(database.VideoStats.view_count)
    ).limit(5).all()
    
    print(f"Top videos by views: {top_videos_by_views}")
    
    # Get top performing videos by engagement
    top_videos_by_engagement = db.query(
        database.VideoData.title,
        database.VideoEngagement.engagement_rate
    ).join(
        database.VideoEngagement, database.VideoData.id == database.VideoEngagement.video_id
    ).filter(
        database.VideoData.id.in_(video_ids)
    ).order_by(
        desc(database.VideoEngagement.engagement_rate)
    ).limit(5).all()
    
    print(f"Top videos by engagement: {top_videos_by_engagement}")
    
    # Count videos by duration ranges
    duration_distribution = db.query(
        database.VideoMetaInfo.duration
    ).filter(
        database.VideoMetaInfo.video_id.in_(video_ids)
    ).all()
    
    # Process duration distribution
    duration_ranges = {
        "0-5 min": 0,
        "5-10 min": 0, 
        "10-20 min": 0,
        "20+ min": 0
    }
    
    for (duration,) in duration_distribution:
        # Convert duration to minutes
        minutes = duration / 60
        if minutes <= 5:
            duration_ranges["0-5 min"] += 1
        elif minutes <= 10:
            duration_ranges["5-10 min"] += 1
        elif minutes <= 20:
            duration_ranges["10-20 min"] += 1
        else:
            duration_ranges["20+ min"] += 1
    
    # Get most common tags
    tag_counts = db.query(
        database.VideoTag.tag_name,
        func.count(database.VideoTag.id).label('count')
    ).filter(
        database.VideoTag.video_id.in_(video_ids)
    ).group_by(
        database.VideoTag.tag_name
    ).order_by(
        desc('count')
    ).limit(10).all()
    
    # Prepare results in required format
    results = []
    
    # Add average metrics
    results.append({
        "title": "Average Views",
        "data": {"value": stats_query.avg_views if stats_query.avg_views else 0}
    })
    
    results.append({
        "title": "Average Likes",
        "data": {"value": stats_query.avg_likes if stats_query.avg_likes else 0}
    })
    
    results.append({
        "title": "Average Comments",
        "data": {"value": stats_query.avg_comments if stats_query.avg_comments else 0}
    })
    
    results.append({
        "title": "Average Engagement Rate",
        "data": {"value": engagement_query.avg_engagement if engagement_query.avg_engagement else 0}
    })
    
    # Add total metrics
    results.append({
        "title": "Total Views",
        "data": {"value": stats_query.total_views if stats_query.total_views else 0}
    })
    
    results.append({
        "title": "Total Likes",
        "data": {"value": stats_query.total_likes if stats_query.total_likes else 0}
    })
    
    results.append({
        "title": "Total Comments",
        "data": {"value": stats_query.total_comments if stats_query.total_comments else 0}
    })
    
    # Add top videos by views
    top_views_data = [{"title": title, "views": views} for title, views in top_videos_by_views]
    results.append({
        "title": "Top Videos by Views",
        "data": {"value": top_views_data}
    })
    
    # Add top videos by engagement
    top_engagement_data = [{"title": title, "engagement_rate": rate} for title, rate in top_videos_by_engagement]
    results.append({
        "title": "Top Videos by Engagement",
        "data": {"value": top_engagement_data}
    })
    
    # Add duration distribution
    results.append({
        "title": "Video Duration Distribution",
        "data": {"value": duration_ranges}
    })
    
    # Add tag distribution
    tag_data = [{"tag": tag, "count": count} for tag, count in tag_counts]
    results.append({
        "title": "Most Common Tags",
        "data": {"value": tag_data}
    })
    
    return {"results": results}

# Include API routes
app.include_router(routes.router)

# Add this before the catch-all route
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle any unhandled exceptions and return a consistent error response
    """
    print(f"Unhandled exception occurred: {str(exc)}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "An unexpected error occurred",
            "error": str(exc),
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "request_method": request.method,
            "request_url": str(request.url),
            "documentation_url": "/docs"
        },
    )

# Improved get_available_api_endpoints function
def get_available_api_endpoints(app: FastAPI) -> List[Dict[str, Any]]:
    """
    Get a list of all available API endpoints in the application.
    Excludes internal FastAPI routes and documentation endpoints.
    
    Args:
        app: The FastAPI application instance
        
    Returns:
        List of dictionaries containing information about each endpoint
    """
    endpoints = []
    
    for route in app.routes:
        if isinstance(route, APIRoute):
            # Skip documentation endpoints
            if route.path in ["/docs", "/redoc", "/openapi.json"]:
                continue
                
            # Skip internal FastAPI routes
            if route.path.startswith("/docs") or route.path.startswith("/redoc") or route.path.startswith("/openapi"):
                continue
                
            # Include all other API routes
            endpoints.append({
                "path": str(route.path),
                "methods": list(route.methods),
                "name": route.name or "unnamed",
                "description": route.description or ""
            })
            
    return endpoints

# Catch-all route for non-existent endpoints
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, path_name: str):
    available_endpoints = get_available_api_endpoints(app)
    
    # Check if the path is likely an API request
    if path_name.startswith("api/") or path_name.startswith("/api/"):
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": f"API endpoint '{path_name}' does not exist",
                "status_code": 404,
                "request_method": request.method,
                "request_url": str(request.url),
                "available_endpoints": available_endpoints,
                "documentation_url": "/docs"
            },
        )
    
    # For non-API routes, provide a simpler 404 response
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Not Found",
            "status_code": 404,
            "request_method": request.method,
            "request_url": str(request.url),
            "available_endpoints": available_endpoints,
            "documentation_url": "/docs"
        }
    )

# Add this before the catch-all route
@app.delete("/api/datasets/{dataset_id}", response_model=schemas.SuccessResponse)
@app.delete("/datasets/{dataset_id}", response_model=schemas.SuccessResponse)
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

# Add this endpoint before the catch-all route
@app.get("/api/revenue/{dataset_id}/", response_model=schemas.RevenueResponse)
async def get_revenue(dataset_id: int, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None, 
                    db: Session = Depends(get_db)):
    # Check if dataset exists
    dataset = db.query(database.Dataset).filter(database.Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get all videos for the dataset
    videos = db.query(database.VideoData).filter(database.VideoData.dataset_id == dataset_id).all()
    video_ids = [video.id for video in videos]
    
    if not video_ids:
        raise HTTPException(status_code=404, detail="No videos found in dataset")
    
    # Get stats for these videos
    stats = db.query(database.VideoStats).filter(database.VideoStats.video_id.in_(video_ids)).all()
    
    # Calculate total views
    total_views = sum(stat.view_count for stat in stats)
    
    # Simulate revenue calculation - $1 per 1000 views
    rate_per_1000 = 1.0
    total_revenue = (total_views / 1000) * rate_per_1000
    
    # Generate dummy country data
    countries = ["US", "IN", "GB", "CA", "AU"]
    country_revenue = {}
    
    import random
    for country in countries:
        country_views = int(total_views * random.uniform(0.05, 0.4))
        country_rev = (country_views / 1000) * rate_per_1000
        country_revenue[country] = {
            "views": country_views,
            "revenue": country_rev,
            "rate_per_1000": rate_per_1000
        }
    
    # Generate time series data
    time_series = []
    from datetime import datetime, timedelta
    
    # Default to last 30 days if no dates provided
    end = datetime.now()
    start = end - timedelta(days=30)
    
    if end_date:
        try:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except:
            end = datetime.strptime(end_date, "%Y-%m-%d")
    
    if start_date:
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except:
            start = datetime.strptime(start_date, "%Y-%m-%d")
    
    days = (end - start).days + 1
    daily_revenue = total_revenue / days
    
    for i in range(days):
        current_date = start + timedelta(days=i)
        # Add some randomness to make the chart more interesting
        day_revenue = daily_revenue * random.uniform(0.5, 1.5)
        time_series.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "revenue": day_revenue
        })
    
    return {
        "total_views": total_views,
        "total_revenue": total_revenue,
        "country_revenue": country_revenue,
        "time_series": time_series
    }

# Add this endpoint before the catch-all route
@app.post("/api/predict/ai-video/", response_model=schemas.AIPrediction)
async def predict_video_performance(
    prediction_request: schemas.AIPredictionRequest,
    db: Session = Depends(get_db)
):
    """Generate AI-based predictions for video performance based on content type and posting schedule."""
    
    # If dataset_id is provided, use it to get historical data for better prediction
    base_metrics = {
        "music_video": {
            "views": 50000,
            "likes": 5000,
            "comments": 500,
            "engagement_rate": 0.12,
            "revenue_per_1000": 2.50
        },
        "tutorial": {
            "views": 30000,
            "likes": 3000,
            "comments": 800,
            "engagement_rate": 0.15,
            "revenue_per_1000": 3.20
        },
        "vlog": {
            "views": 25000,
            "likes": 2500,
            "comments": 600,
            "engagement_rate": 0.13,
            "revenue_per_1000": 2.10
        },
        "short_form": {
            "views": 100000,
            "likes": 15000,
            "comments": 1000,
            "engagement_rate": 0.18,
            "revenue_per_1000": 1.80
        }
    }
    
    # Day of week modifiers (Friday and Saturday perform better)
    day_modifiers = {
        "monday": 0.85,
        "tuesday": 0.90,
        "wednesday": 0.95,
        "thursday": 1.0,
        "friday": 1.30,
        "saturday": 1.25,
        "sunday": 1.10
    }
    
    # Time of day modifiers (Evening and night perform better)
    time_modifiers = {
        "morning": 0.80,
        "midday": 0.90,
        "afternoon": 1.0,
        "evening": 1.25,
        "night": 1.15,
        "late_night": 0.70
    }
    
    # Get the base metrics for the content type
    content_type = prediction_request.content_type
    if content_type not in base_metrics:
        content_type = "music_video"  # Default
    
    base = base_metrics[content_type]
    
    # Apply modifiers
    day_mod = day_modifiers.get(prediction_request.day_of_week, 1.0)
    time_mod = time_modifiers.get(prediction_request.time_of_day, 1.0)
    
    # If dataset_id is provided, use some of its statistics to influence the prediction
    dataset_mod = 1.0
    if prediction_request.dataset_id:
        try:
            # Get videos from the dataset
            videos = db.query(database.VideoData).filter(
                database.VideoData.dataset_id == prediction_request.dataset_id
            ).all()
            
            if videos:
                video_ids = [video.id for video in videos]
                
                # Get average views and engagement
                stats_query = db.query(
                    func.avg(database.VideoStats.view_count).label('avg_views'),
                    func.avg(database.VideoStats.like_count).label('avg_likes'),
                    func.avg(database.VideoStats.comment_count).label('avg_comments')
                ).filter(database.VideoStats.video_id.in_(video_ids)).first()
                
                engagement_query = db.query(
                    func.avg(database.VideoEngagement.engagement_rate).label('avg_engagement')
                ).filter(database.VideoEngagement.video_id.in_(video_ids)).first()
                
                if stats_query.avg_views:
                    # Scale dataset modifier based on average views compared to base views
                    dataset_mod = min(2.0, max(0.5, stats_query.avg_views / base["views"]))
        except Exception as e:
            print(f"Error using dataset for prediction: {e}")
    
    # Calculate final predictions with some randomness
    import random
    random_factor = random.uniform(0.9, 1.1)
    
    predicted_views = int(base["views"] * day_mod * time_mod * dataset_mod * random_factor)
    predicted_likes = int(base["likes"] * day_mod * time_mod * dataset_mod * random_factor)
    predicted_comments = int(base["comments"] * day_mod * time_mod * dataset_mod * random_factor)
    predicted_engagement = base["engagement_rate"] * day_mod * time_mod * random_factor
    predicted_revenue = (predicted_views / 1000) * base["revenue_per_1000"]
    
    # Determine optimal posting
    optimal_day = max(day_modifiers.items(), key=lambda x: x[1])[0]
    optimal_time = max(time_modifiers.items(), key=lambda x: x[1])[0]
    
    # Generate 30-day growth projection
    growth_projection = []
    
    from datetime import datetime, timedelta
    start_date = datetime.now()
    
    cumulative_views = 0
    daily_view_ratio = [0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025, 
                       0.02, 0.02, 0.015, 0.015, 0.015, 0.01, 0.01, 0.01, 0.01, 0.01,
                       0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    
    for day in range(1, 31):
        current_date = start_date + timedelta(days=day-1)
        day_views = int(predicted_views * daily_view_ratio[day-1] * random.uniform(0.9, 1.1))
        cumulative_views += day_views
        day_revenue = (day_views / 1000) * base["revenue_per_1000"]
        
        growth_projection.append({
            "day": day,
            "date": current_date.strftime("%Y-%m-%d"),
            "views": cumulative_views,
            "revenue": day_revenue
        })
    
    return {
        "content_type": content_type,
        "predicted_views": predicted_views,
        "predicted_likes": predicted_likes,
        "predicted_comments": predicted_comments,
        "predicted_engagement_rate": predicted_engagement,
        "predicted_revenue": predicted_revenue,
        "optimal_posting": {
            "day": optimal_day,
            "time": optimal_time
        },
        "growth_projection": growth_projection
    }

if __name__ == "__main__":
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get app settings from environment variables
    port = int(os.getenv("APP_PORT", 1111))
    host = os.getenv("APP_HOST", "0.0.0.0")
    log_level = os.getenv("APP_LOG_LEVEL", "info")
    
    print(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level=log_level) 