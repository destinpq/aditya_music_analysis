#!/usr/bin/env python3

import os
import requests
import json
import sys

def upload_csv(file_path, api_url='http://localhost:1111/upload/'):
    """Upload a CSV file to the backend API."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # File info
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    print(f"Uploading {file_name} ({file_size} bytes) to {api_url}")
    
    # Create the multipart/form-data payload
    files = {
        'file': (file_name, open(file_path, 'rb'), 'text/csv')
    }
    
    try:
        # Disable debug logging from urllib3
        import logging
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        # Debug request headers
        print("\nRequest headers:")
        headers = {
            'Accept': 'application/json',
        }
        for key, value in headers.items():
            print(f"  {key}: {value}")

        response = requests.post(
            api_url,
            files=files,
            headers=headers,
        )
        
        # Print response details
        print(f"\nResponse status: {response.status_code}")
        print(f"Response headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        # Try to parse response as JSON
        try:
            response_data = response.json()
            print("\nResponse JSON:")
            print(json.dumps(response_data, indent=2))
            
            if response.status_code == 200:
                print(f"\nUpload successful! Dataset ID: {response_data.get('id')}")
                return True
            else:
                print(f"\nUpload failed with status {response.status_code}")
                return False
        except json.JSONDecodeError:
            print("\nResponse is not valid JSON:")
            print(response.text[:500])  # Print first 500 chars
            return False
            
    except Exception as e:
        print(f"Error during upload: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Close the file handle
        files['file'][1].close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file_path>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not upload_csv(csv_file):
        print("Upload failed")
        sys.exit(1)
    else:
        print("Upload completed successfully") 