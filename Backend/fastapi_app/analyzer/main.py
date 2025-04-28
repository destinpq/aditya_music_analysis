import pandas as pd
import argparse
import os
import json
from . import (
    TitleAnalyzer, 
    PublishedAtAnalyzer, 
    DurationAnalyzer,
    ViewCountAnalyzer,
    LikeCountAnalyzer,
    CommentCountAnalyzer,
    ThumbnailAnalyzer,
    CategoryAnalyzer,
    TagAnalyzer
)


def analyze_youtube_data(file_path, output_dir=None):
    """
    Analyze YouTube data using all analyzers
    
    Args:
        file_path: Path to the CSV file with YouTube data
        output_dir: Directory to save output files (defaults to 'results')
    """
    # Create output directory
    if output_dir is None:
        output_dir = 'results'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Read data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Create dictionary of analyzers with their expected column names
    analyzers = {
        'title': TitleAnalyzer(df),
        'published_at': PublishedAtAnalyzer(df),
        'duration': DurationAnalyzer(df),
        'view_count': ViewCountAnalyzer(df),
        'like_count': LikeCountAnalyzer(df),
        'comment_count': CommentCountAnalyzer(df),
        'thumbnail_url': ThumbnailAnalyzer(df),
        'category_id': CategoryAnalyzer(df),
        'tags': TagAnalyzer(df)
    }
    
    # Find matching columns in the dataframe
    column_mapping = {}
    for expected_name, analyzer in analyzers.items():
        # Look for exact match first
        if expected_name in df.columns:
            column_mapping[expected_name] = expected_name
        else:
            # Look for columns containing the name (case insensitive)
            matches = [col for col in df.columns if expected_name.lower() in col.lower()]
            if matches:
                column_mapping[expected_name] = matches[0]
    
    print("\nDetected column mapping:")
    for expected, actual in column_mapping.items():
        print(f"  {expected} -> {actual}")
    
    # Run all analyzers
    all_results = {}
    
    for name, analyzer in analyzers.items():
        if name in column_mapping:
            print(f"\nRunning {name} analysis...")
            try:
                # Set the correct column name for the analyzer
                analyzer.set_data(df, column_mapping[name])
                
                # Run analysis
                results = analyzer.analyze()
                
                # Create visualizations
                visualizations = analyzer.create_visualization()
                
                # Store results
                all_results[name] = results
                
                print(f"  Analysis complete. Generated {len(visualizations)} visualizations.")
            except Exception as e:
                print(f"  Error analyzing {name}: {str(e)}")
        else:
            print(f"\nSkipping {name} analysis (column not found)")
    
    # Save combined results
    result_file = os.path.join(output_dir, 'analysis_results.json')
    
    # Remove non-serializable data (like numpy arrays) before saving
    clean_results = {}
    for key, value in all_results.items():
        if isinstance(value, dict):
            # Convert numpy values to Python types
            clean_results[key] = _convert_to_serializable(value)
        else:
            print(f"Skipping non-dict result: {key}")
    
    with open(result_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {result_file}")
    print(f"Visualizations saved to {os.path.join(output_dir, 'plots')}")


def _convert_to_serializable(obj):
    """Convert values to JSON serializable format"""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_serializable(v) for v in obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):  # Check for numpy arrays
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze YouTube data from CSV file')
    parser.add_argument('file_path', help='Path to CSV file with YouTube data')
    parser.add_argument('--output', '-o', help='Directory to save output files', default='results')
    
    args = parser.parse_args()
    
    analyze_youtube_data(args.file_path, args.output) 