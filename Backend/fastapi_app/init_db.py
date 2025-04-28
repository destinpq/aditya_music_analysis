"""
Initialize the PostgreSQL database with the required tables.
Run this script once when setting up a new database.
"""

from database import init_db, engine, Base

def main():
    print("Creating PostgreSQL database tables...")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print("Database initialization complete.")

if __name__ == "__main__":
    main() 