import fastapi_app.database as db
from sqlalchemy import inspect

def drop_all_tables():
    """Drop all tables in the database"""
    inspector = inspect(db.engine)
    
    print("Dropping all tables...")
    # Get all table names
    table_names = inspector.get_table_names()
    
    # Create a new Base metadata with existing tables
    # This is needed because we may have table references that require a specific drop order
    metadata = db.Base.metadata
    
    # Drop all tables
    metadata.drop_all(db.engine)
    print(f"Dropped tables: {', '.join(table_names)}")

def recreate_tables():
    """Recreate all tables"""
    print("Creating tables...")
    db.Base.metadata.create_all(db.engine)
    print("Tables created successfully")

if __name__ == "__main__":
    # First drop all tables
    drop_all_tables()
    
    # Then recreate them with the new schema
    recreate_tables()
    
    print("Database schema reset complete") 