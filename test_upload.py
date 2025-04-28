"""
Test script for uploading files to the FastAPI backend.
"""
import os
import sys
import requests
import json
from pathlib import Path

def test_backend_connection():
    """Test if the backend is reachable"""
    try:
        response = requests.get("http://localhost:1111/")
        print(f"Backend connection: {response.status_code}")
        print(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to backend: {e}")
        return False

def test_upload(file_path):
    """Test uploading a file to the backend"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    print(f"Uploading file: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    try:
        with open(file_path, 'rb') as f:
            filename = os.path.basename(file_path)
            response = requests.post(
                "http://localhost:1111/upload/",
                files={"file": (filename, f, "text/csv")}
            )
        
        print(f"Upload response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Dataset created with ID: {data.get('id')}")
            return True
        else:
            print(f"Upload failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error during upload: {e}")
        return False

def main():
    """Main function"""
    if not test_backend_connection():
        print("Backend connection failed. Exiting.")
        return
    
    # Check for sample data file
    sample_data = "../sample_data.csv"
    if not os.path.exists(sample_data):
        print(f"Sample data file not found: {sample_data}")
        sample_data = input("Enter path to CSV file to upload: ")
    
    if test_upload(sample_data):
        print("Upload test successful!")
    else:
        print("Upload test failed!")

if __name__ == "__main__":
    main() 