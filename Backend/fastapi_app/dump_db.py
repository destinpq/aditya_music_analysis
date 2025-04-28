import os
import sys
from database import SessionLocal, Dataset, VideoData, VideoStats, VideoEngagement, VideoMetaInfo, VideoTag
from sqlalchemy import inspect, MetaData, text
from sqlalchemy.engine import reflection
import json
from datetime import datetime

def format_value(value):
    """Format a value for display"""
    if isinstance(value, datetime):
        return value.isoformat()
    return value

def dump_table(inspector, table_name, db):
    """Dump all rows from a table"""
    print(f"\n{'=' * 80}")
    print(f"TABLE: {table_name}")
    print(f"{'=' * 80}")
    
    # Get column information
    columns = inspector.get_columns(table_name)
    col_names = [col['name'] for col in columns]
    
    # Get all rows using proper SQLAlchemy text() function
    query = text(f"SELECT * FROM {table_name}")
    rows = db.execute(query).fetchall()
    
    # Print count
    print(f"Total rows: {len(rows)}")
    
    # Print column headers
    header = " | ".join(col_names)
    print(f"\n{header}")
    print("-" * len(header))
    
    # Print rows
    for row in rows[:20]:  # Limit to 20 rows
        formatted_row = [str(format_value(value)) for value in row]
        print(" | ".join(formatted_row))
    
    if len(rows) > 20:
        print(f"... and {len(rows) - 20} more rows")

def dump_database():
    """Dump all tables and their contents from the database"""
    db = SessionLocal()
    try:
        # Get database inspector
        inspector = inspect(db.bind)
        
        # Get all table names
        table_names = inspector.get_table_names()
        
        print(f"\nFound {len(table_names)} tables: {', '.join(table_names)}")
        
        # Dump each table
        for table_name in table_names:
            dump_table(inspector, table_name, db)
            
        # Print table relationships
        print("\n\nTable Relationships:")
        print("------------------")
        metadata = MetaData()
        metadata.reflect(bind=db.bind)
        
        for table_name, table in metadata.tables.items():
            for fk in table.foreign_keys:
                print(f"{table_name}.{fk.parent.name} -> {fk.column.table.name}.{fk.column.name}")
        
    except Exception as e:
        print(f"Error dumping database: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    print("Dumping database contents...")
    dump_database()
    print("\nDatabase dump completed.") 