import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from .base_analyzer import BaseAnalyzer


class DurationAnalyzer(BaseAnalyzer):
    """Analyzer for video durations"""
    
    def __init__(self, df=None, column_name='duration'):
        super().__init__(df, column_name)
    
    def get_column_type(self):
        """Return the type of column this analyzer handles"""
        return "Duration"
    
    def analyze(self):
        """Perform analysis on video duration data"""
        self.validate_data()
        
        # Try to convert duration to seconds if it's not already numeric
        self.df['duration_seconds'] = self._convert_to_seconds(self.df[self.column_name])
        
        # Basic duration statistics
        duration_stats = {
            'avg_duration_seconds': self.df['duration_seconds'].mean(),
            'median_duration_seconds': self.df['duration_seconds'].median(),
            'min_duration_seconds': self.df['duration_seconds'].min(),
            'max_duration_seconds': self.df['duration_seconds'].max(),
            'std_duration_seconds': self.df['duration_seconds'].std()
        }
        
        # Convert seconds to formatted time for easier reading
        for key in ['avg_duration_seconds', 'median_duration_seconds', 'min_duration_seconds', 'max_duration_seconds']:
            seconds = duration_stats[key]
            duration_stats[f"{key.replace('_seconds', '')}_formatted"] = self._format_duration(seconds)
        
        # Create duration categories for analysis
        duration_cats = [
            (0, 30, 'Under 30s'),
            (30, 60, '30s-1min'),
            (60, 180, '1-3min'),
            (180, 300, '3-5min'),
            (300, 600, '5-10min'),
            (600, 1200, '10-20min'),
            (1200, 1800, '20-30min'),
            (1800, 3600, '30-60min'),
            (3600, float('inf'), 'Over 60min')
        ]
        
        self.df['duration_category'] = pd.cut(
            self.df['duration_seconds'],
            bins=[bounds[0] for bounds in duration_cats] + [float('inf')],
            labels=[bounds[2] for bounds in duration_cats],
            include_lowest=True
        )
        
        # Distribution of videos by duration category
        duration_distribution = self.df['duration_category'].value_counts().sort_index().to_dict()
        
        # Duration trends over time if timestamp exists
        duration_trends = None
        if any(col in self.df.columns for col in ['published_at', 'upload_date']):
            date_col = next(col for col in ['published_at', 'upload_date'] if col in self.df.columns)
            
            try:
                if self.df[date_col].dtype != 'datetime64[ns]':
                    self.df['publish_date'] = pd.to_datetime(self.df[date_col], errors='coerce')
                else:
                    self.df['publish_date'] = self.df[date_col]
                
                # Extract year and month
                self.df['publish_year'] = self.df['publish_date'].dt.year
                
                # Calculate average duration by year
                duration_by_year = self.df.groupby('publish_year')['duration_seconds'].mean().to_dict()
                
                # Add to trends
                duration_trends = {
                    'by_year': duration_by_year
                }
            except:
                pass
        
        # Performance by duration if view count exists
        perf_by_duration = None
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            # Calculate average views by duration category
            perf_by_category = self.df.groupby('duration_category')[view_col].mean().to_dict()
            
            # Find optimal duration range
            optimal_duration = max(perf_by_category.items(), key=lambda x: x[1])
            
            # Get more granular by creating custom duration ranges
            self.df['duration_minutes'] = self.df['duration_seconds'] / 60
            
            # Create minute-based ranges for more detailed analysis
            minute_ranges = list(range(0, 61, 1))  # 1-minute intervals up to 60 minutes
            self.df['duration_minute_range'] = pd.cut(
                self.df['duration_minutes'].clip(upper=60),  # Cap at 60 minutes
                bins=minute_ranges,
                labels=[f"{i}-{i+1}" for i in minute_ranges[:-1]],
                include_lowest=True
            )
            
            # Calculate views by minute range
            views_by_minute = self.df.groupby('duration_minute_range')[view_col].mean().to_dict()
            
            # Build the performance data
            perf_by_duration = {
                'by_category': perf_by_category,
                'optimal_duration_category': optimal_duration[0],
                'by_minute': views_by_minute
            }
            
            # Calculate engagement metrics if likes exist
            if 'likes' in self.df.columns or any('like' in col.lower() for col in self.df.columns):
                like_col = 'likes' if 'likes' in self.df.columns else next(col for col in self.df.columns if 'like' in col.lower())
                
                # Calculate like-to-view ratio by duration
                self.df['engagement_rate'] = self.df[like_col] / self.df[view_col]
                engagement_by_category = self.df.groupby('duration_category')['engagement_rate'].mean().to_dict()
                
                perf_by_duration['engagement_by_category'] = engagement_by_category
        
        # Calculate watch time efficiency (views per minute of content)
        watch_time_efficiency = None
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            # Calculate total watch time potential (duration × views)
            self.df['total_watch_time'] = self.df['duration_seconds'] * self.df[view_col]
            
            # Calculate efficiency metrics
            self.df['views_per_second'] = self.df[view_col] / self.df['duration_seconds']
            
            # Aggregate by category
            efficiency_by_category = self.df.groupby('duration_category')['views_per_second'].mean().to_dict()
            
            watch_time_efficiency = {
                'by_category': efficiency_by_category,
                'overall_views_per_second': self.df['views_per_second'].mean(),
                'total_watch_time_hours': self.df['total_watch_time'].sum() / 3600
            }
        
        # Store results
        self.results = {
            'duration_stats': duration_stats,
            'duration_distribution': duration_distribution
        }
        
        if duration_trends:
            self.results['duration_trends'] = duration_trends
            
        if perf_by_duration:
            self.results['performance_by_duration'] = perf_by_duration
            
        if watch_time_efficiency:
            self.results['watch_time_efficiency'] = watch_time_efficiency
            
        return self.results
    
    def _convert_to_seconds(self, duration_series):
        """Convert duration strings to seconds"""
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(duration_series):
            return duration_series
        
        # Make a copy to avoid warnings
        durations = duration_series.astype(str).copy()
        
        # Try to parse different formats
        seconds = []
        for d in durations:
            try:
                # ISO 8601 format (PT1H2M3S)
                if 'PT' in d:
                    # Extract hours, minutes, seconds
                    d = d.replace('PT', '')
                    total_seconds = 0
                    
                    # Extract hours
                    if 'H' in d:
                        h, d = d.split('H')
                        total_seconds += int(h) * 3600
                    
                    # Extract minutes
                    if 'M' in d:
                        m, d = d.split('M')
                        total_seconds += int(m) * 60
                    
                    # Extract seconds
                    if 'S' in d:
                        s = d.replace('S', '')
                        total_seconds += int(s)
                    
                    seconds.append(total_seconds)
                
                # MM:SS format
                elif ':' in d and d.count(':') == 1:
                    m, s = d.split(':')
                    seconds.append(int(m) * 60 + int(s))
                
                # HH:MM:SS format
                elif ':' in d and d.count(':') == 2:
                    h, m, s = d.split(':')
                    seconds.append(int(h) * 3600 + int(m) * 60 + int(s))
                
                # Just a number (assume seconds)
                else:
                    seconds.append(float(d))
            except:
                # If parsing fails, use NaN
                seconds.append(np.nan)
        
        return pd.Series(seconds, index=duration_series.index)
    
    def _format_duration(self, seconds):
        """Format seconds as HH:MM:SS"""
        if pd.isna(seconds):
            return "00:00:00"
        
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def create_visualization(self):
        """Create visualizations for duration analysis"""
        if not self.results:
            self.analyze()
        
        visualizations = {}
        
        # Duration distribution histogram
        fig_hist = px.histogram(
            self.df,
            x='duration_seconds',
            nbins=30,
            labels={'duration_seconds': 'Duration (seconds)'},
            title='Distribution of Video Durations'
        )
        fig_hist.update_layout(template='plotly_white', showlegend=False)
        visualizations['duration_hist'] = self.save_plot(fig_hist, 'duration_histogram.json')
        
        # Videos by duration category
        categories = list(self.results['duration_distribution'].keys())
        counts = list(self.results['duration_distribution'].values())
        
        # Sort by category bounds (already sorted if using pd.cut with sort=True)
        fig_cats = px.bar(
            x=categories,
            y=counts,
            labels={'x': 'Duration Category', 'y': 'Number of Videos'},
            title='Videos by Duration Category'
        )
        fig_cats.update_layout(template='plotly_white')
        visualizations['duration_categories'] = self.save_plot(fig_cats, 'duration_categories.json')
        
        # Performance by duration if available
        if 'performance_by_duration' in self.results:
            perf_data = self.results['performance_by_duration']['by_category']
            categories = list(perf_data.keys())
            perf_values = list(perf_data.values())
            
            fig_perf = px.bar(
                x=categories,
                y=perf_values,
                labels={'x': 'Duration Category', 'y': 'Average Views'},
                title='Performance by Duration Category'
            )
            fig_perf.update_layout(template='plotly_white')
            visualizations['perf_by_duration'] = self.save_plot(fig_perf, 'perf_by_duration.json')
            
            # If we have minute-by-minute data
            if 'by_minute' in self.results['performance_by_duration']:
                minute_data = self.results['performance_by_duration']['by_minute']
                minutes = [key for key in minute_data.keys() if not pd.isna(key)]
                minute_perf = [minute_data[key] for key in minutes]
                
                fig_minute = px.line(
                    x=minutes,
                    y=minute_perf,
                    labels={'x': 'Duration (minutes)', 'y': 'Average Views'},
                    title='Performance by Duration (minutes)'
                )
                fig_minute.update_layout(template='plotly_white')
                visualizations['perf_by_minute'] = self.save_plot(fig_minute, 'perf_by_minute.json')
        
        # Duration trends over time if available
        if 'duration_trends' in self.results and 'by_year' in self.results['duration_trends']:
            year_data = self.results['duration_trends']['by_year']
            years = list(year_data.keys())
            avg_durations = list(year_data.values())
            
            fig_trend = px.line(
                x=years,
                y=avg_durations,
                labels={'x': 'Year', 'y': 'Average Duration (seconds)'},
                title='Average Video Duration by Year'
            )
            fig_trend.update_layout(template='plotly_white')
            visualizations['duration_trend'] = self.save_plot(fig_trend, 'duration_trend.json')
        
        self.results['visualizations'] = visualizations
        return visualizations 