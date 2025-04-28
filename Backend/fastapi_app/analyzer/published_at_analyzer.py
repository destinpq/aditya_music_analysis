import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime, timedelta
import calendar
from .base_analyzer import BaseAnalyzer


class PublishedAtAnalyzer(BaseAnalyzer):
    """Analyzer for video publishing dates and times"""
    
    def __init__(self, df=None, column_name='published_at'):
        super().__init__(df, column_name)
    
    def get_column_type(self):
        """Return the type of column this analyzer handles"""
        return "Published At"
    
    def analyze(self):
        """Perform comprehensive analysis on video publishing dates and times"""
        self.validate_data()
        
        # Convert to datetime if not already
        try:
            if self.df[self.column_name].dtype != 'datetime64[ns]':
                self.df['publish_datetime'] = pd.to_datetime(self.df[self.column_name], errors='coerce')
            else:
                self.df['publish_datetime'] = self.df[self.column_name]
        except Exception as e:
            print(f"Error converting to datetime: {e}")
            # Try different format if standard parsing fails
            try:
                self.df['publish_datetime'] = pd.to_datetime(self.df[self.column_name], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
            except:
                raise ValueError(f"Could not parse dates in column {self.column_name}")
        
        # Remove rows with invalid dates
        valid_dates = self.df['publish_datetime'].notna()
        if valid_dates.sum() < len(self.df) * 0.5:
            raise ValueError(f"More than 50% of dates could not be parsed in column {self.column_name}")
        
        valid_df = self.df[valid_dates].copy()
        
        # Extract date components
        valid_df['year'] = valid_df['publish_datetime'].dt.year
        valid_df['month'] = valid_df['publish_datetime'].dt.month
        valid_df['day'] = valid_df['publish_datetime'].dt.day
        valid_df['hour'] = valid_df['publish_datetime'].dt.hour
        valid_df['minute'] = valid_df['publish_datetime'].dt.minute
        valid_df['day_of_week'] = valid_df['publish_datetime'].dt.dayofweek
        valid_df['day_name'] = valid_df['publish_datetime'].dt.day_name()
        valid_df['month_name'] = valid_df['publish_datetime'].dt.month_name()
        valid_df['quarter'] = valid_df['publish_datetime'].dt.quarter
        
        # Time-based categorization
        valid_df['time_of_day'] = pd.cut(
            valid_df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
            include_lowest=True
        )
        
        # Basic date stats
        date_range = {
            'earliest_date': valid_df['publish_datetime'].min(),
            'latest_date': valid_df['publish_datetime'].max(),
            'date_range_days': (valid_df['publish_datetime'].max() - valid_df['publish_datetime'].min()).days,
            'total_videos': len(valid_df)
        }
        
        # Publishing frequency
        valid_df['publish_date'] = valid_df['publish_datetime'].dt.date
        daily_counts = valid_df.groupby('publish_date').size()
        
        frequency_stats = {
            'total_publishing_days': len(daily_counts),
            'avg_videos_per_publishing_day': daily_counts.mean(),
            'max_videos_per_day': daily_counts.max(),
            'median_videos_per_day': daily_counts.median(),
        }
        
        # Time patterns
        hour_distribution = valid_df['hour'].value_counts().sort_index().to_dict()
        day_distribution = valid_df['day_of_week'].value_counts().sort_index().to_dict()
        day_names = {i: calendar.day_name[i] for i in range(7)}
        day_of_week_named = {day_names[k]: v for k, v in day_distribution.items()}
        
        time_of_day_dist = valid_df['time_of_day'].value_counts().to_dict()
        month_distribution = valid_df['month'].value_counts().sort_index().to_dict()
        month_names = {i: calendar.month_name[i] for i in range(1, 13)}
        month_named = {month_names[k]: v for k, v in month_distribution.items()}
        
        year_distribution = valid_df['year'].value_counts().sort_index().to_dict()
        
        # Publishing consistency
        weekly_consistency = self._calculate_consistency(valid_df, 'day_of_week')
        hourly_consistency = self._calculate_consistency(valid_df, 'hour')
        
        # Publishing gaps
        gaps = self._calculate_publishing_gaps(valid_df)
        
        # Performance analysis by publishing time
        perf_by_time = None
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            # Performance by day of week
            perf_by_day = valid_df.groupby('day_name')[view_col].mean().to_dict()
            
            # Performance by hour
            perf_by_hour = valid_df.groupby('hour')[view_col].mean().to_dict()
            
            # Performance by time of day 
            perf_by_time_of_day = valid_df.groupby('time_of_day')[view_col].mean().to_dict()
            
            # Performance by month
            perf_by_month = valid_df.groupby('month_name')[view_col].mean().to_dict()
            
            perf_by_time = {
                'by_day_of_week': perf_by_day,
                'by_hour': perf_by_hour,
                'by_time_of_day': perf_by_time_of_day,
                'by_month': perf_by_month
            }
        
        # Store results
        self.results = {
            'date_range': date_range,
            'frequency_stats': frequency_stats,
            'hour_distribution': hour_distribution,
            'day_of_week_distribution': day_of_week_named,
            'time_of_day_distribution': time_of_day_dist,
            'month_distribution': month_named,
            'year_distribution': year_distribution,
            'publishing_consistency': {
                'weekly': weekly_consistency,
                'hourly': hourly_consistency
            },
            'publishing_gaps': gaps
        }
        
        if perf_by_time:
            self.results['performance_by_time'] = perf_by_time
            
        return self.results
    
    def _calculate_consistency(self, df, time_unit):
        """Calculate publishing consistency for a given time unit"""
        counts = df[time_unit].value_counts()
        total = len(df)
        
        # Calculate coefficient of variation (lower means more consistent)
        mean = counts.mean()
        std = counts.std()
        cv = (std / mean) if mean > 0 else 0
        
        # Calculate entropy (higher means more uniform distribution)
        probs = counts / total
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(counts)) if len(counts) > 0 else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'coefficient_of_variation': cv,
            'normalized_entropy': normalized_entropy,
            'consistency_score': (1 - cv) * 0.5 + normalized_entropy * 0.5  # 0-1 scale, higher is more consistent
        }
    
    def _calculate_publishing_gaps(self, df):
        """Analyze gaps between publishing dates"""
        # Sort by date
        df_sorted = df.sort_values('publish_datetime')
        
        # Calculate days between publications
        df_sorted['next_date'] = df_sorted['publish_datetime'].shift(-1)
        df_sorted['days_to_next'] = (df_sorted['next_date'] - df_sorted['publish_datetime']).dt.total_seconds() / 86400
        
        # Remove last row which will have NaN
        gap_df = df_sorted.dropna(subset=['days_to_next'])
        
        if len(gap_df) == 0:
            return {
                'avg_gap_days': 0,
                'max_gap_days': 0,
                'gap_distribution': {}
            }
        
        # Calculate gap statistics
        avg_gap = gap_df['days_to_next'].mean()
        max_gap = gap_df['days_to_next'].max()
        
        # Create gap bins
        gap_df['gap_category'] = pd.cut(
            gap_df['days_to_next'],
            bins=[0, 1, 2, 3, 7, 14, 30, 100000],
            labels=['Same day', '1 day', '2 days', '3-6 days', '1-2 weeks', '2-4 weeks', 'Over 4 weeks'],
            include_lowest=True
        )
        
        gap_distribution = gap_df['gap_category'].value_counts().to_dict()
        
        return {
            'avg_gap_days': avg_gap,
            'max_gap_days': max_gap,
            'gap_distribution': gap_distribution
        }
    
    def create_visualization(self):
        """Create visualizations for publishing date analysis"""
        if not self.results:
            self.analyze()
        
        visualizations = {}
        
        # Ensure we have valid data to work with
        valid_df = self.df[self.df['publish_datetime'].notna()].copy()
        if len(valid_df) == 0:
            return {}
        
        # Posts by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = [self.results['day_of_week_distribution'].get(day, 0) for day in day_order]
        
        fig_days = px.bar(
            x=day_order,
            y=day_counts,
            labels={'x': 'Day of Week', 'y': 'Number of Videos'},
            title='Videos Published by Day of Week'
        )
        fig_days.update_layout(template='plotly_white')
        visualizations['days_of_week'] = self.save_plot(fig_days, 'publish_days_of_week.json')
        
        # Posts by hour of day
        hours = list(range(24))
        hour_counts = [self.results['hour_distribution'].get(hour, 0) for hour in hours]
        
        fig_hours = px.bar(
            x=hours,
            y=hour_counts,
            labels={'x': 'Hour of Day (24h)', 'y': 'Number of Videos'},
            title='Videos Published by Hour of Day'
        )
        fig_hours.update_layout(template='plotly_white')
        visualizations['hours_of_day'] = self.save_plot(fig_hours, 'publish_hours.json')
        
        # Posts by month
        month_order = [calendar.month_name[i] for i in range(1, 13)]
        month_counts = [self.results['month_distribution'].get(month, 0) for month in month_order if month != '']
        
        fig_months = px.bar(
            x=[m for m in month_order if m != ''],
            y=month_counts,
            labels={'x': 'Month', 'y': 'Number of Videos'},
            title='Videos Published by Month'
        )
        fig_months.update_layout(template='plotly_white')
        visualizations['months'] = self.save_plot(fig_months, 'publish_months.json')
        
        # Publishing volume over time
        valid_df['year_month'] = valid_df['publish_datetime'].dt.strftime('%Y-%m')
        time_series = valid_df.groupby('year_month').size().reset_index(name='count')
        time_series.columns = ['date', 'count']
        time_series = time_series.sort_values('date')
        
        fig_timeline = px.line(
            time_series,
            x='date',
            y='count',
            labels={'date': 'Month', 'count': 'Number of Videos'},
            title='Publishing Volume Over Time'
        )
        fig_timeline.update_layout(template='plotly_white')
        visualizations['timeline'] = self.save_plot(fig_timeline, 'publish_timeline.json')
        
        # Performance by publishing time if available
        if 'performance_by_time' in self.results:
            # Performance by day of week
            perf_by_day = self.results['performance_by_time']['by_day_of_week']
            day_perf = [perf_by_day.get(day, 0) for day in day_order]
            
            fig_day_perf = px.bar(
                x=day_order,
                y=day_perf,
                labels={'x': 'Day of Week', 'y': 'Average Views'},
                title='Performance by Day of Week'
            )
            fig_day_perf.update_layout(template='plotly_white')
            visualizations['perf_by_day'] = self.save_plot(fig_day_perf, 'perf_by_day.json')
            
        self.results['visualizations'] = visualizations
        return visualizations 