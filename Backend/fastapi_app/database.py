from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create database directory if it doesn't exist (for backward compatibility)
os.makedirs("database", exist_ok=True)

# Get database URL from environment or build it from parameters
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # If DATABASE_URL is not set, build it from individual parts
    DB_USERNAME = os.getenv("DB_USERNAME", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "youtube_analytics")
    DB_SSLMODE = os.getenv("DB_SSLMODE", "disable")
    
    # If SQLite mode is enabled, override with SQLite connection
    if os.getenv("DB_USE_SQLITE", "false").lower() == "true":
        DATABASE_URL = "sqlite:///./database/youtube_analytics.db"
    else:
        # Build PostgreSQL connection string
        DATABASE_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        if DB_SSLMODE:
            DATABASE_URL += f"?sslmode={DB_SSLMODE}"

print(f"Using database connection: {DATABASE_URL}")

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create base class for declarative models
Base = declarative_base()

# Define models
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    row_count = Column(Integer, default=0)
    
    video_data = relationship("VideoData", back_populates="dataset", cascade="all, delete-orphan")

class VideoData(Base):
    __tablename__ = "video_data"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    video_id = Column(String(20), nullable=False)
    title = Column(String(255), nullable=False)
    published_at = Column(DateTime, nullable=True)
    
    dataset = relationship("Dataset", back_populates="video_data")
    stats = relationship("VideoStats", back_populates="video", uselist=False, cascade="all, delete-orphan")
    engagement = relationship("VideoEngagement", back_populates="video", uselist=False, cascade="all, delete-orphan")
    meta_info = relationship("VideoMetaInfo", back_populates="video", uselist=False, cascade="all, delete-orphan")
    tags = relationship("VideoTag", back_populates="video", cascade="all, delete-orphan")

class VideoStats(Base):
    __tablename__ = "video_stats"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("video_data.id"), nullable=False)
    view_count = Column(Integer, nullable=True)
    like_count = Column(Integer, nullable=True)
    dislike_count = Column(Integer, nullable=True)
    favorite_count = Column(Integer, nullable=True)
    comment_count = Column(Integer, nullable=True)
    
    video = relationship("VideoData", back_populates="stats")

class VideoEngagement(Base):
    __tablename__ = "video_engagement"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("video_data.id"), nullable=False)
    engagement_rate = Column(Float, nullable=True)
    like_ratio = Column(Float, nullable=True)
    comment_ratio = Column(Float, nullable=True)
    
    video = relationship("VideoData", back_populates="engagement")

class VideoMetaInfo(Base):
    __tablename__ = "video_meta_info"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("video_data.id"), nullable=False)
    duration = Column(Integer, nullable=True)
    channel_id = Column(String(50), nullable=True)
    category_id = Column(Integer, nullable=True)
    is_unlisted = Column(Boolean, nullable=True)
    
    video = relationship("VideoData", back_populates="meta_info")

class VideoTag(Base):
    __tablename__ = "video_tags"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("video_data.id"), nullable=False)
    tag_name = Column(Text, nullable=False)  # Using Text type for unlimited length
    
    video = relationship("VideoData", back_populates="tags")

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)

def store_dataset(db, df, filename):
    """Store a dataset in the database."""
    # Create a new dataset record
    dataset = Dataset(
        filename=filename,
        upload_date=datetime.datetime.utcnow(),
        row_count=len(df)
    )
    db.add(dataset)
    db.flush()  # Flush to generate ID before adding videos
    
    # Rename columns to match expected format (case insensitive)
    column_mapping = {
        'Video ID': 'video_id',
        'Title': 'title',
        'Published At': 'published_at',
        'Duration': 'duration',
        'View Count': 'view_count',
        'Like Count': 'like_count',
        'Dislike Count': 'dislike_count',
        'Favorite Count': 'favorite_count',
        'Comment Count': 'comment_count',
        'Thumbnail URL': 'thumbnail_url',
        'Category ID': 'category_id',
        'Tags': 'tags'
    }
    
    # Rename columns if they exist
    for orig, new in column_mapping.items():
        if orig in df.columns:
            df[new] = df[orig]
    
    # Debug print columns
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"First row sample: {df.iloc[0].to_dict()}")
    
    for index, row in enumerate(df.iterrows()):
        _, row = row  # Unpack the tuple
        
        # SAFELY EXTRACT ALL DATA WITH TYPE CHECKING
        
        # Ensure basic text fields exist with safe defaults
        video_id = str(row.get('video_id', '')) if pd.notna(row.get('video_id', '')) else ''
        title = str(row.get('title', '')) if pd.notna(row.get('title', '')) else ''
        
        # Print row data for debugging for the first few rows
        if index < 3:
            print(f"Processing row {index+1}: video_id={video_id}, title={title}")
        
        try:
            # Handle published_at date format with robust parsing
            published_at = None
            if 'published_at' in row and pd.notna(row['published_at']):
                try:
                    # Try multiple date formats
                    date_str = str(row['published_at']).strip()
                    try:
                        published_at = pd.to_datetime(date_str)
                    except:
                        # Try common formats explicitly
                        for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                            try:
                                published_at = datetime.datetime.strptime(date_str, date_format)
                                break
                            except:
                                continue
                except Exception as e:
                    print(f"Error parsing date for row {index+1}: {e}")
            
            # Create the video data record
            video_data = VideoData(
                dataset_id=dataset.id,
                video_id=video_id,
                title=title,
                published_at=published_at
            )
            db.add(video_data)
            db.flush()  # Flush to get the video_data.id without committing
            
            # === VIDEO STATS ===
            # Initialize all count values to 0
            view_count = 0
            like_count = 0
            dislike_count = 0 
            favorite_count = 0
            comment_count = 0
            
            # == HANDLE VIEW COUNT ==
            if 'view_count' in row and pd.notna(row['view_count']):
                try:
                    # Convert safely - handle commas and decimals
                    view_count_str = str(row['view_count']).replace(',', '')
                    # Remove any non-numeric characters except decimal point
                    view_count_str = ''.join(c for c in view_count_str if c.isdigit() or c == '.')
                    # Convert to float first, then to int
                    view_count = int(float(view_count_str))
                except Exception as e:
                    print(f"Error parsing view_count for row {index+1}: {e}")
                    view_count = 0
            
            # == HANDLE LIKE COUNT ==
            if 'like_count' in row and pd.notna(row['like_count']):
                try:
                    # Convert safely - handle commas and decimals
                    like_count_str = str(row['like_count']).replace(',', '')
                    # Remove any non-numeric characters except decimal point
                    like_count_str = ''.join(c for c in like_count_str if c.isdigit() or c == '.')
                    # Convert to float first, then to int
                    like_count = int(float(like_count_str))
                except Exception as e:
                    print(f"Error parsing like_count for row {index+1}: {e}")
                    like_count = 0
            
            # == HANDLE DISLIKE COUNT ==
            if 'dislike_count' in row and pd.notna(row['dislike_count']):
                try:
                    # Convert safely - handle commas and decimals
                    dislike_count_str = str(row['dislike_count']).replace(',', '')
                    # Remove any non-numeric characters except decimal point
                    dislike_count_str = ''.join(c for c in dislike_count_str if c.isdigit() or c == '.')
                    # Convert to float first, then to int
                    dislike_count = int(float(dislike_count_str))
                except Exception as e:
                    print(f"Error parsing dislike_count for row {index+1}: {e}")
                    dislike_count = 0
            
            # == HANDLE FAVORITE COUNT ==
            if 'favorite_count' in row and pd.notna(row['favorite_count']):
                try:
                    # Convert safely - handle commas and decimals
                    favorite_count_str = str(row['favorite_count']).replace(',', '')
                    # Remove any non-numeric characters except decimal point
                    favorite_count_str = ''.join(c for c in favorite_count_str if c.isdigit() or c == '.')
                    # Convert to float first, then to int
                    favorite_count = int(float(favorite_count_str))
                except Exception as e:
                    print(f"Error parsing favorite_count for row {index+1}: {e}")
                    favorite_count = 0
            
            # == HANDLE COMMENT COUNT ==
            if 'comment_count' in row and pd.notna(row['comment_count']):
                try:
                    # Convert safely - handle commas and decimals
                    comment_count_str = str(row['comment_count']).replace(',', '')
                    # Remove any non-numeric characters except decimal point
                    comment_count_str = ''.join(c for c in comment_count_str if c.isdigit() or c == '.')
                    # Convert to float first, then to int
                    comment_count = int(float(comment_count_str))
                except Exception as e:
                    print(f"Error parsing comment_count for row {index+1}: {e}")
                    comment_count = 0
            
            # Create the stats record
            video_stats = VideoStats(
                video_id=video_data.id,
                view_count=view_count,
                like_count=like_count,
                dislike_count=dislike_count,
                favorite_count=favorite_count,
                comment_count=comment_count
            )
            db.add(video_stats)
            
            # === VIDEO ENGAGEMENT ===
            # Calculate engagement metrics safely
            engagement_rate = 0
            like_ratio = 0
            comment_ratio = 0
            
            if view_count > 0:
                engagement_rate = (like_count + comment_count) / view_count
                like_ratio = like_count / view_count
                comment_ratio = comment_count / view_count
                
            video_engagement = VideoEngagement(
                video_id=video_data.id,
                engagement_rate=engagement_rate,
                like_ratio=like_ratio,
                comment_ratio=comment_ratio
            )
            db.add(video_engagement)
            
            # === VIDEO META INFO ===
            # Parse duration with comprehensive handling
            duration = 0
            try:
                if 'duration' in row and pd.notna(row['duration']):
                    duration_str = str(row['duration'])
                    
                    # Handle PT format (ISO 8601)
                    if duration_str.startswith('PT'):
                        # Remove PT prefix
                        time_part = duration_str[2:]
                        
                        hours = 0
                        minutes = 0
                        seconds = 0
                        
                        # Extract hours if H is present
                        if 'H' in time_part:
                            h_parts = time_part.split('H')
                            hours = int(h_parts[0])
                            time_part = h_parts[1]
                        
                        # Extract minutes if M is present
                        if 'M' in time_part:
                            m_parts = time_part.split('M')
                            minutes = int(m_parts[0])
                            time_part = m_parts[1]
                        
                        # Extract seconds if S is present
                        if 'S' in time_part:
                            s_parts = time_part.split('S')
                            seconds = int(s_parts[0])
                        
                        # If no H, M, or S but there are digits after PT, interpret as seconds
                        elif time_part.isdigit():
                            seconds = int(time_part)
                        
                        duration = hours * 3600 + minutes * 60 + seconds
                    
                    # Handle HH:MM:SS format
                    elif ':' in duration_str:
                        parts = duration_str.split(':')
                        if len(parts) == 3:
                            hours, minutes, seconds = map(int, parts)
                            duration = hours * 3600 + minutes * 60 + seconds
                        elif len(parts) == 2:
                            minutes, seconds = map(int, parts)
                            duration = minutes * 60 + seconds
                        else:
                            duration = int(parts[0])
                    
                    # Direct seconds
                    elif duration_str.isdigit():
                        duration = int(duration_str)
                    
                    # Try to extract numeric portion if all else fails
                    else:
                        numeric_part = ''.join(c for c in duration_str if c.isdigit())
                        if numeric_part:
                            duration = int(numeric_part)
            except Exception as e:
                print(f"Error parsing duration for row {index+1}: {e}")
                duration = 0
            
            # Handle other meta fields safely
            channel_id = str(row.get('channel_id', '')) if pd.notna(row.get('channel_id', '')) else ''
            
            # Category ID handling
            category_id = 0
            if 'category_id' in row and pd.notna(row['category_id']):
                try:
                    category_id_str = str(row['category_id'])
                    category_id_str = ''.join(c for c in category_id_str if c.isdigit())
                    if category_id_str:
                        category_id = int(category_id_str)
                except Exception as e:
                    print(f"Error parsing category_id for row {index+1}: {e}")
            
            # Boolean fields
            is_unlisted = False
            if 'is_unlisted' in row and pd.notna(row['is_unlisted']):
                try:
                    is_unlisted_val = row['is_unlisted']
                    if isinstance(is_unlisted_val, bool):
                        is_unlisted = is_unlisted_val
                    elif isinstance(is_unlisted_val, str):
                        is_unlisted = is_unlisted_val.lower() in ['true', 'yes', '1', 't', 'y']
                    elif isinstance(is_unlisted_val, (int, float)):
                        is_unlisted = bool(is_unlisted_val)
                except Exception as e:
                    print(f"Error parsing is_unlisted for row {index+1}: {e}")
            
            video_meta = VideoMetaInfo(
                video_id=video_data.id,
                duration=duration,
                channel_id=channel_id,
                category_id=category_id,
                is_unlisted=is_unlisted
            )
            db.add(video_meta)
            
            # === HANDLE TAGS ===
            if 'tags' in row and pd.notna(row['tags']):
                tags = row['tags']
                try:
                    if isinstance(tags, str):
                        # Try to parse as JSON if it's a string
                        if tags.startswith('['):
                            try:
                                import json
                                parsed_tags = json.loads(tags)
                                if isinstance(parsed_tags, list):
                                    tags = parsed_tags
                                else:
                                    tags = [tags]
                            except:
                                # If JSON parsing fails, split by comma
                                tags = [tag.strip() for tag in tags.split(',')]
                        else:
                            # Default to comma-separated
                            tags = [tag.strip() for tag in tags.split(',')]
                    elif not isinstance(tags, list):
                        # Convert to string and then list with one item
                        tags = [str(tags)]
                    
                    # Handle list of tags
                    for tag in tags:
                        if tag and pd.notna(tag):
                            tag_record = VideoTag(
                                video_id=video_data.id,
                                tag_name=str(tag)
                            )
                            db.add(tag_record)
                except Exception as e:
                    print(f"Error processing tags for row {index+1}: {e}")
        except Exception as e:
            print(f"Error processing row {index+1}: {e}")
            continue
        
        # Commit every 100 rows or at the end to optimize performance
        if index % 100 == 0 or index == len(df) - 1:
            db.commit()
            print(f"Progress: {index + 1}/{len(df)} rows processed ({((index + 1) / len(df) * 100):.2f}%)")
    
    # Return the dataset object, not just the ID
    return dataset

def reset_db():
    """Drop all tables and recreate them."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine) 