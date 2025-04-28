# Database Migration: SQLite to PostgreSQL

This document describes how to migrate the application from SQLite to PostgreSQL.

## Prerequisites

- Python 3.8+
- pip dependencies installed (see requirements.txt)
- Access to a PostgreSQL database

## Configuration

1. Create a `.env` file in the `Backend/fastapi_app` directory with your PostgreSQL credentials:

```
DB_USERNAME=your_username
DB_PASSWORD=your_password
DB_HOST=your_host
DB_PORT=your_port
DB_NAME=your_database
DB_SSLMODE=require

# Application settings
APP_PORT=1111
APP_HOST=0.0.0.0
APP_LOG_LEVEL=info
```

## Migration Steps

### 1. Initialize the PostgreSQL Database

Run the initialization script to create the tables in PostgreSQL:

```bash
python init_db.py
```

### 2. Migrate Data from SQLite to PostgreSQL (Optional)

If you have existing data in SQLite that you want to transfer to PostgreSQL, run the migration script:

```bash
python migrate_sqlite_to_postgres.py
```

### 3. Run the Application with PostgreSQL

Start the FastAPI application, which will now use PostgreSQL:

```bash
python main.py
```

## Database Schema

The database consists of the following tables:

- `datasets`: Stores information about uploaded datasets
- `video_data`: Stores basic information about videos
- `video_stats`: Stores view, like, and other metrics
- `video_engagement`: Stores engagement metrics
- `video_meta_info`: Stores metadata about videos
- `video_tags`: Stores video tags

## Troubleshooting

If you encounter issues:

1. Verify your PostgreSQL credentials in the `.env` file
2. Ensure the PostgreSQL server is running and accessible
3. Check if the tables were created correctly in PostgreSQL
4. Look for error messages in the application output

For connection issues, try testing the connection with `psql` or another PostgreSQL client.

## Backup and Restore

It's recommended to back up your SQLite database before migration:

```bash
cp database/youtube_analytics.db database/youtube_analytics.db.backup
```

To back up the PostgreSQL database, use `pg_dump`:

```bash
pg_dump -h your_host -p your_port -U your_username -d your_database > backup.sql
```

## Contributors

- DESTIN PQ Team 