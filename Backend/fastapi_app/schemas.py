from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

class DatasetBase(BaseModel):
    filename: str

class DatasetCreate(DatasetBase):
    pass

class Dataset(DatasetBase):
    id: int
    filename: str
    upload_date: datetime
    row_count: int
    
    class Config:
        orm_mode = True

class VideoDataBase(BaseModel):
    video_id: str
    title: str
    published_at: Optional[datetime] = None
    dataset_id: int

class VideoDataCreate(VideoDataBase):
    pass

class VideoData(VideoDataBase):
    id: int
    
    class Config:
        orm_mode = True

class VideoStatsBase(BaseModel):
    view_count: int = 0
    like_count: int = 0
    dislike_count: int = 0
    favorite_count: int = 0
    comment_count: int = 0
    video_id: int

class VideoStatsCreate(VideoStatsBase):
    pass

class VideoStats(VideoStatsBase):
    id: int
    
    class Config:
        orm_mode = True

class VideoEngagementBase(BaseModel):
    engagement_rate: float = 0.0
    like_ratio: float = 0.0
    comment_ratio: float = 0.0
    video_id: int

class VideoEngagementCreate(VideoEngagementBase):
    pass

class VideoEngagement(VideoEngagementBase):
    id: int
    
    class Config:
        orm_mode = True

class VideoMetaInfoBase(BaseModel):
    duration: int = 0
    channel_id: Optional[str] = None
    category_id: Optional[str] = None
    is_unlisted: bool = False
    video_id: int

class VideoMetaInfoCreate(VideoMetaInfoBase):
    pass

class VideoMetaInfo(VideoMetaInfoBase):
    id: int
    
    class Config:
        orm_mode = True

class VideoTagBase(BaseModel):
    tag_name: str
    video_id: int

class VideoTagCreate(VideoTagBase):
    pass

class VideoTag(VideoTagBase):
    id: int
    
    class Config:
        orm_mode = True

class FileUpload(BaseModel):
    filename: str

class AnalysisDataPoint(BaseModel):
    value: Any

class AnalysisItem(BaseModel):
    title: str
    data: AnalysisDataPoint

class AnalysisResults(BaseModel):
    results: List[AnalysisItem]

class DatasetResponse(BaseModel):
    id: int
    filename: str
    record_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class VideoResponse(BaseModel):
    id: int
    video_id: str
    title: str
    published_at: datetime
    view_count: int
    like_count: int
    dislike_count: int
    favorite_count: int
    comment_count: int
    
    class Config:
        from_attributes = True

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error: Optional[str] = None
    status_code: Optional[int] = None

class SuccessResponse(BaseModel):
    success: bool = True
    data: Optional[dict] = None
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

class TopVideo(BaseModel):
    video_id: str
    title: str
    views: int

class DatasetStats(BaseModel):
    total_videos: int
    total_views: int
    avg_engagement_rate: float
    avg_like_ratio: float
    top_videos: List[TopVideo]

class CountryRevenue(BaseModel):
    views: int
    revenue: float
    rate_per_1000: float

class DailyRevenue(BaseModel):
    date: str
    revenue: float

class RevenueResponse(BaseModel):
    total_views: int
    total_revenue: float
    country_revenue: Dict[str, CountryRevenue]
    time_series: List[DailyRevenue]

class AIPredictionRequest(BaseModel):
    content_type: str = "music_video"  # music_video, tutorial, vlog, short_form
    day_of_week: str = "friday"  # monday, tuesday, wednesday, thursday, friday, saturday, sunday
    time_of_day: str = "evening"  # morning, midday, afternoon, evening, night, late_night
    dataset_id: Optional[int] = None  # Optional dataset to base predictions on

class DailyProjection(BaseModel):
    day: int
    date: str
    views: int
    revenue: float

class AIPrediction(BaseModel):
    content_type: str
    predicted_views: int
    predicted_likes: int
    predicted_comments: int
    predicted_engagement_rate: float
    predicted_revenue: float
    optimal_posting: Dict[str, str]
    growth_projection: List[DailyProjection]

# New schemas for prediction and earnings endpoints

class PredictionInput(BaseModel):
    """Input model for video performance prediction"""
    title: Optional[str] = None
    duration: Optional[str] = None
    tags: Optional[str] = None
    published_at: Optional[str] = None
    category_id: Optional[str] = None
    
    # Optional engineered features if already available
    duration_seconds: Optional[int] = None
    tag_count: Optional[int] = None
    publish_month: Optional[int] = None
    publish_day_of_week: Optional[int] = None
    publish_hour: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Full Concert - Aditya Music Awards 2023",
                "duration": "PT1H45M30S",
                "tags": "music,concert,award show,live performance",
                "published_at": "2023-06-15T18:30:00Z", 
                "category_id": "10"
            }
        }

class PredictionResult(BaseModel):
    """Output model for prediction results"""
    predicted_views: int
    predicted_likes: int
    predicted_comments: int
    
class EarningsInput(BaseModel):
    """Input model for earnings calculation"""
    view_count: int = Field(..., description="Number of video views")
    custom_cpm: Optional[float] = Field(None, description="Custom CPM rate")
    country: Optional[str] = Field(None, description="Specific country for calculation")
    geography: Optional[Dict[str, float]] = Field(None, description="Custom geography distribution")
    monetization_rate: Optional[float] = Field(0.85, description="Percentage of monetizable views")
    ad_impression_rate: Optional[float] = Field(0.70, description="Percentage of monetized views with ads")
    
    class Config:
        schema_extra = {
            "example": {
                "view_count": 1000000,
                "custom_cpm": None,
                "country": None,
                "monetization_rate": 0.85
            }
        }

class EarningsByCPM(BaseModel):
    """Model for earnings calculation by CPM"""
    min: float
    average: float
    max: float

class CountryEarnings(BaseModel):
    """Model for country-specific earnings details"""
    views: int
    distribution: float
    cpm: Dict[str, float]
    earnings: EarningsByCPM

class EarningsResult(BaseModel):
    """Output model for earnings calculation"""
    view_count: int
    monetized_views: int
    ad_impression_views: int
    monetization_rate: float
    ad_impression_rate: float
    estimated_earnings: Union[float, EarningsByCPM]
    earnings_by_country: Optional[Dict[str, CountryEarnings]] = None
    geography_distribution: Optional[Dict[str, float]] = None
    country: Optional[str] = None
    cpm_range: Optional[Dict[str, float]] = None
    cpm: Optional[float] = None
    error: Optional[str] = None

class TrainModelInput(BaseModel):
    """Input for model training"""
    dataset_id: int = Field(..., description="ID of the dataset to train on")
    test_size: Optional[float] = Field(0.2, description="Proportion for test split")
    model_name: Optional[str] = Field("youtube_model", description="Name to save the model as")

class TrainModelResult(BaseModel):
    """Result of model training"""
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None

class CPMRate(BaseModel):
    """CPM rate for a country"""
    min: float
    avg: float
    max: float

class CPMRatesResponse(BaseModel):
    """Response with CPM rates"""
    rates: Dict[str, CPMRate]

# Add these schemas at the end of the file

class DatasetProcessRequest(BaseModel):
    """Request model for dataset processing"""
    dataset_id: int
    test_size: Optional[float] = Field(0.2, description="Proportion for test split", ge=0.05, le=0.5)
    random_state: Optional[int] = Field(42, description="Random seed for reproducibility")
    dataset_name: Optional[str] = Field("youtube_dataset", description="Name for the processed dataset")

class DatasetFeatureInfo(BaseModel):
    """Information about dataset features"""
    feature_columns: List[str]
    target_columns: List[str]
    num_samples: int
    num_features: int

class DatasetSplitInfo(BaseModel):
    """Information about train/test split"""
    train_size: int
    test_size: int
    train_files: List[str]
    test_files: List[str]

class DatasetProcessResponse(BaseModel):
    """Response model for dataset processing"""
    success: bool
    dataset_name: str
    features: DatasetFeatureInfo
    split: DatasetSplitInfo
    error: Optional[str] = None

class DatasetListResponse(BaseModel):
    """Response model for listing available datasets"""
    success: bool
    datasets: List[str]
    count: int

class DatasetInfoResponse(BaseModel):
    """Response model for dataset information"""
    success: bool
    dataset_name: str
    feature_columns: List[str]
    target_columns: List[str]
    train_size: int
    test_size: int
    total_size: int
    error: Optional[str] = None

class NewDataProcessRequest(BaseModel):
    """Request model for processing new data with existing dataset state"""
    dataset_name: str = Field(..., description="Name of the reference dataset")
    csv_file_id: Optional[int] = Field(None, description="ID of uploaded CSV file")
    target_columns: Optional[List[str]] = Field(None, description="Optional target columns to include") 