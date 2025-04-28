from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
import database as db

# Create SQLite database engine
SQLALCHEMY_DATABASE_URL = "sqlite:///database/youtube_analytics.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_database():
    """Query the database and print values for verification"""
    session = SessionLocal()
    
    # Get all datasets
    datasets = session.query(db.Dataset).all()
    print(f"Found {len(datasets)} datasets in the database")
    
    for dataset in datasets:
        print(f"\nDataset ID: {dataset.id}, Filename: {dataset.filename}")
        
        # Get video count
        video_count = session.query(db.VideoData).filter(db.VideoData.dataset_id == dataset.id).count()
        print(f"Total videos in dataset: {video_count}")
        
        # Get sample videos
        videos = session.query(db.VideoData).filter(db.VideoData.dataset_id == dataset.id).limit(5).all()
        
        print("\nSample videos from database:")
        for i, video in enumerate(videos):
            # Get stats for this video
            stats = session.query(db.VideoStats).filter(db.VideoStats.video_id == video.id).first()
            engagement = session.query(db.VideoEngagement).filter(db.VideoEngagement.video_id == video.id).first()
            
            print(f"\nVideo {i+1}:")
            print(f"  ID: {video.video_id}")
            print(f"  Title: {video.title}")
            print(f"  Published At: {video.published_at}")
            
            if stats:
                print(f"  View Count: {stats.view_count}")
                print(f"  Like Count: {stats.like_count}")
                print(f"  Comment Count: {stats.comment_count}")
            
            if engagement:
                print(f"  Engagement Rate: {engagement.engagement_rate:.4f}")
                print(f"  Like Ratio: {engagement.like_ratio:.4f}")
        
        # Get aggregate stats
        stats_query = session.query(
            db.VideoStats.view_count.label('views'),
            db.VideoStats.like_count.label('likes'),
            db.VideoStats.comment_count.label('comments')
        ).join(
            db.VideoData, db.VideoData.id == db.VideoStats.video_id
        ).filter(
            db.VideoData.dataset_id == dataset.id
        ).limit(10).all()
        
        print("\nFirst 10 videos' stats from database (view count, like count, comment count):")
        for i, (views, likes, comments) in enumerate(stats_query):
            print(f"  Video {i+1}: Views={views}, Likes={likes}, Comments={comments}")
        
    session.close()

if __name__ == "__main__":
    check_database() 