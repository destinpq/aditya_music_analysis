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
    DB_USERNAME = os.getenv("DB_USERNAME", "db_username_placeholder")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "db_password_placeholder")
    DB_HOST = os.getenv("DB_HOST", "db_host_placeholder")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "defaultdb")
    DB_SSLMODE = os.getenv("DB_SSLMODE", "require")
    
    # Build PostgreSQL connection string
    SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"
else:
    SQLALCHEMY_DATABASE_URL = DATABASE_URL

# Create the database engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create base class for declarative models
Base = declarative_base()

# ... rest of the file remains unchanged ... 