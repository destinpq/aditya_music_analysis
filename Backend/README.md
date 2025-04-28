# YouTube Analytics Tools

A comprehensive suite of analytics tools for YouTube data analysis. This package provides detailed analysis for each of the main YouTube data points:

- Title analysis
- Publishing date/time patterns
- Video duration analysis
- View count metrics and predictions
- Like count and engagement analysis
- Comment count and audience interaction
- Thumbnail URL analysis
- Category performance analysis
- Tag impact and optimization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/youtube-analytics.git
cd youtube-analytics
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (for text analysis):
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### Command Line

The simplest way to analyze your YouTube data is to use the command line tool:

```bash
python main.py path/to/your/youtube_data.csv --output results_directory
```

This will:
1. Load your YouTube data CSV
2. Run all applicable analyzers
3. Generate visualizations
4. Save all results to the specified output directory

### Using Individual Analyzers

You can also use individual analyzers in your own Python code:

```python
import pandas as pd
from analyzer import TitleAnalyzer, ViewCountAnalyzer

# Load your data
df = pd.read_csv('your_youtube_data.csv')

# Analyze titles
title_analyzer = TitleAnalyzer(df, 'title_column_name')
title_results = title_analyzer.analyze()
title_visualizations = title_analyzer.create_visualization()

# Analyze view counts
view_analyzer = ViewCountAnalyzer(df, 'view_count_column_name')
view_results = view_analyzer.analyze()
view_visualizations = view_analyzer.create_visualization()

# Access the results
print(f"Average title length: {title_results['title_length_stats']['avg_length']}")
print(f"Total views: {view_results['view_stats']['total_views']}")
```

## Expected CSV Format

The tool works with any CSV file containing YouTube data, with flexible column name detection. However, it works best if your CSV has some of these columns:

- `title` - Video title
- `published_at` - Publishing date and time
- `duration` - Video duration
- `view_count` - Number of views
- `like_count` - Number of likes
- `comment_count` - Number of comments
- `thumbnail_url` - URL to the video thumbnail
- `category_id` - YouTube category ID
- `tags` - Video tags (can be comma-separated, JSON array, or pipe-separated)

The analyzers will try to match column names flexibly, so minor variations in naming should work.

## Visualization Examples

The tool generates various visualizations for each analyzer, which are saved in the output directory:

- Title length distribution
- Publishing time patterns
- Duration vs. performance charts
- View count distribution
- Engagement analysis
- Tag performance charts
- And many more...

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 