import pandas as pd
import os
from sqlalchemy.orm import Session
from database import SessionLocal, store_dataset

def import_data():
    """Directly import data from CSV into the database"""
    # Path to the CSV file
    csv_path = os.path.join('..', 'uploads', 'ADITYA_MUSIC.csv')
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    print(f"Loading file: {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"CSV loaded with {len(df)} rows")
        
        # Get a database session
        db = SessionLocal()
        
        # Store the dataset
        dataset = store_dataset(db, df, "ADITYA_MUSIC.csv")
        print(f"Dataset created with ID: {dataset.id}")
        
        db.close()
        
    except Exception as e:
        print(f"Error importing data: {str(e)}")

if __name__ == "__main__":
    import_data() 