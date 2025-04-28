"""
Migration script to transfer data from SQLite to PostgreSQL.
"""

import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import database as db

# Load environment variables
load_dotenv()

# SQLite database path
SQLITE_DB_PATH = "database/youtube_analytics.db"

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
    PG_CONNECTION_STRING = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"
else:
    PG_CONNECTION_STRING = DATABASE_URL

def check_sqlite_exists():
    """Check if SQLite database exists."""
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"SQLite database not found at {SQLITE_DB_PATH}")
        return False
    return True

def migrate_data():
    """Migrate data from SQLite to PostgreSQL."""
    if not check_sqlite_exists():
        return False
    
    # Initialize the PostgreSQL database tables
    print("Initializing PostgreSQL database tables...")
    db.init_db()
    
    # Connect to SQLite database
    print("Connecting to SQLite database...")
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    
    # Get list of tables
    tables = sqlite_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()
    
    # Create SQLAlchemy engine for PostgreSQL
    pg_engine = create_engine(PG_CONNECTION_STRING)
    
    # Migrate each table
    for table in tables:
        table_name = table[0]
        print(f"Migrating table: {table_name}")
        
        # Skip SQLite internal tables
        if table_name.startswith('sqlite_'):
            continue
        
        # Read data from SQLite
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
        
        if len(df) > 0:
            print(f"  - Found {len(df)} rows")
            
            # Write data to PostgreSQL
            df.to_sql(table_name, pg_engine, if_exists='append', index=False)
            print(f"  - Migrated to PostgreSQL successfully")
        else:
            print(f"  - Table is empty, skipping")
    
    # Close connections
    sqlite_conn.close()
    
    print("Migration completed successfully!")
    return True

if __name__ == "__main__":
    print("Starting migration from SQLite to PostgreSQL...")
    migrate_data() 