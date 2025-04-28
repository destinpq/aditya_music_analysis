import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
from datetime import datetime, timedelta
from .base_analyzer import BaseAnalyzer


class LikeCountAnalyzer(BaseAnalyzer):
    """Analyzer for video like counts and engagement metrics"""
    
    def __init__(self, df=None, column_name='like_count'):
        super().__init__(df, column_name)
    
    def get_column_type(self):
        """Return the type of column this analyzer handles"""
        return "Like Count"
    
    def analyze(self):
        """Perform comprehensive analysis on video likes and engagement"""
        self.validate_data()
        
        # Ensure like count is numeric
        self.df['likes'] = pd.to_numeric(self.df[self.column_name], errors='coerce')
        
        # Basic like count statistics
        like_stats = {
            'total_likes': self.df['likes'].sum(),
            'avg_likes': self.df['likes'].mean(),
            'median_likes': self.df['likes'].median(),
            'min_likes': self.df['likes'].min(),
            'max_likes': self.df['likes'].max(),
            'std_likes': self.df['likes'].std()
        }
        
        # Create like count categories
        self.df['like_category'] = pd.cut(
            self.df['likes'],
            bins=[0, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, float('inf')],
            labels=['<10', '10-50', '50-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-50K', '50K-100K', '>100K'],
            include_lowest=True
        )
        
        # Distribution of videos by like count category
        like_distribution = self.df['like_category'].value_counts().sort_index().to_dict()
        
        # Engagement analysis (likes to views ratio)
        engagement_analysis = None
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            # Calculate engagement rate
            self.df['engagement_rate'] = (self.df['likes'] / self.df[view_col]).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate engagement stats
            engagement_stats = {
                'avg_engagement_rate': self.df['engagement_rate'].mean() * 100,  # as percentage
                'median_engagement_rate': self.df['engagement_rate'].median() * 100,
                'max_engagement_rate': self.df['engagement_rate'].max() * 100,
                'min_engagement_rate': self.df['engagement_rate'].min() * 100
            }
            
            # Create engagement categories
            self.df['engagement_category'] = pd.cut(
                self.df['engagement_rate'] * 100,  # convert to percentage
                bins=[0, 1, 2, 3, 5, 10, 15, 20, 30, 100],
                labels=['<1%', '1-2%', '2-3%', '3-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%'],
                include_lowest=True
            )
            
            engagement_distribution = self.df['engagement_category'].value_counts().sort_index().to_dict()
            
            # Find correlation between likes and views
            try:
                correlation, p_value = pearsonr(self.df['likes'].dropna(), self.df[view_col].dropna())
                correlation_analysis = {
                    'likes_views_correlation': correlation,
                    'correlation_p_value': p_value,
                    'correlation_strength': self._interpret_correlation(correlation)
                }
            except:
                correlation_analysis = {
                    'likes_views_correlation': None,
                    'correlation_strength': 'Could not calculate'
                }
            
            # Identify videos with unusually high/low engagement
            avg_engagement = self.df['engagement_rate'].mean()
            std_engagement = self.df['engagement_rate'].std()
            
            high_engagement = self.df[self.df['engagement_rate'] > avg_engagement + 2*std_engagement]
            low_engagement = self.df[self.df['engagement_rate'] < avg_engagement - std_engagement]
            
            # Get details for top engaging videos
            title_column = next((col for col in self.df.columns if any(term in col.lower() for term in ['title', 'name', 'video'])), None)
            
            high_engagement_videos = []
            if title_column and not high_engagement.empty:
                for _, row in high_engagement.head(5).iterrows():
                    high_engagement_videos.append({
                        'title': row[title_column] if not pd.isna(row[title_column]) else f"Video {_}",
                        'engagement_rate': row['engagement_rate'] * 100,
                        'likes': row['likes'],
                        'views': row[view_col]
                    })
            
            # Store engagement analysis
            engagement_analysis = {
                'engagement_stats': engagement_stats,
                'engagement_distribution': engagement_distribution,
                'correlation_analysis': correlation_analysis,
                'high_engagement_videos': high_engagement_videos,
                'high_engagement_count': len(high_engagement),
                'low_engagement_count': len(low_engagement)
            }
        
        # Like count trends over time if timestamp exists
        like_trends = None
        if any(col in self.df.columns for col in ['published_at', 'upload_date']):
            date_col = next(col for col in ['published_at', 'upload_date'] if col in self.df.columns)
            
            try:
                if self.df[date_col].dtype != 'datetime64[ns]':
                    self.df['publish_date'] = pd.to_datetime(self.df[date_col], errors='coerce')
                else:
                    self.df['publish_date'] = self.df[date_col]
                
                # Extract year and month
                self.df['publish_year'] = self.df['publish_date'].dt.year
                self.df['publish_month'] = self.df['publish_date'].dt.month
                self.df['publish_yearmonth'] = self.df['publish_date'].dt.strftime('%Y-%m')
                
                # Calculate average likes by year
                likes_by_year = self.df.groupby('publish_year')['likes'].mean().to_dict()
                
                # Calculate likes by month-year
                likes_by_month = self.df.groupby('publish_yearmonth')['likes'].mean().to_dict()
                
                # Add to trends
                like_trends = {
                    'by_year': likes_by_year,
                    'by_month': likes_by_month
                }
                
                # Calculate engagement trends if views exist
                if engagement_analysis:
                    engagement_by_year = self.df.groupby('publish_year')['engagement_rate'].mean().apply(lambda x: x*100).to_dict()
                    engagement_by_month = self.df.groupby('publish_yearmonth')['engagement_rate'].mean().apply(lambda x: x*100).to_dict()
                    
                    like_trends['engagement_by_year'] = engagement_by_year
                    like_trends['engagement_by_month'] = engagement_by_month
            except Exception as e:
                print(f"Error analyzing like trends: {e}")
        
        # Store results
        self.results = {
            'like_stats': like_stats,
            'like_distribution': like_distribution
        }
        
        if engagement_analysis:
            self.results['engagement_analysis'] = engagement_analysis
            
        if like_trends:
            self.results['like_trends'] = like_trends
            
        return self.results
    
    def _interpret_correlation(self, corr):
        """Interpret the strength of correlation coefficient"""
        if corr is None:
            return "Unknown"
        
        abs_corr = abs(corr)
        
        if abs_corr < 0.1:
            return "Negligible"
        elif abs_corr < 0.3:
            return "Weak"
        elif abs_corr < 0.5:
            return "Moderate"
        elif abs_corr < 0.7:
            return "Strong"
        else:
            return "Very Strong"
    
    def create_visualization(self):
        """Create visualizations for like count analysis"""
        if not self.results:
            self.analyze()
        
        visualizations = {}
        
        # Like count distribution histogram
        fig_hist = px.histogram(
            self.df,
            x='likes',
            log_x=True,  # Log scale for better visualization of skewed data
            nbins=30,
            labels={'likes': 'Like Count (log scale)'},
            title='Distribution of Video Like Counts'
        )
        fig_hist.update_layout(template='plotly_white', showlegend=False)
        visualizations['like_hist'] = self.save_plot(fig_hist, 'like_histogram.json')
        
        # Videos by like count category
        categories = list(self.results['like_distribution'].keys())
        counts = list(self.results['like_distribution'].values())
        
        fig_cats = px.bar(
            x=categories,
            y=counts,
            labels={'x': 'Like Count Category', 'y': 'Number of Videos'},
            title='Videos by Like Count Category'
        )
        fig_cats.update_layout(template='plotly_white')
        visualizations['like_categories'] = self.save_plot(fig_cats, 'like_categories.json')
        
        # Likes vs. Views scatter plot
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            fig_scatter = px.scatter(
                self.df,
                x=view_col,
                y='likes',
                log_x=True,
                log_y=True,
                labels={'likes': 'Likes (log scale)', view_col: 'Views (log scale)'},
                title='Likes vs. Views',
                opacity=0.7,
                trendline='ols'
            )
            fig_scatter.update_layout(template='plotly_white')
            visualizations['likes_vs_views'] = self.save_plot(fig_scatter, 'likes_vs_views.json')
            
            # Engagement rate distribution
            if 'engagement_analysis' in self.results:
                fig_engagement = px.histogram(
                    self.df,
                    x='engagement_rate',
                    nbins=30,
                    labels={'engagement_rate': 'Engagement Rate (Likes/Views)'},
                    title='Distribution of Engagement Rates'
                )
                fig_engagement.update_layout(template='plotly_white', showlegend=False)
                visualizations['engagement_dist'] = self.save_plot(fig_engagement, 'engagement_dist.json')
                
                # Engagement by category
                if 'engagement_distribution' in self.results['engagement_analysis']:
                    eng_cats = list(self.results['engagement_analysis']['engagement_distribution'].keys())
                    eng_counts = list(self.results['engagement_analysis']['engagement_distribution'].values())
                    
                    fig_eng_cats = px.bar(
                        x=eng_cats,
                        y=eng_counts,
                        labels={'x': 'Engagement Rate Category', 'y': 'Number of Videos'},
                        title='Videos by Engagement Rate Category'
                    )
                    fig_eng_cats.update_layout(template='plotly_white')
                    visualizations['engagement_categories'] = self.save_plot(fig_eng_cats, 'engagement_categories.json')
        
        # Like trends over time if available
        if 'like_trends' in self.results and 'by_month' in self.results['like_trends']:
            month_data = self.results['like_trends']['by_month']
            months = list(month_data.keys())
            avg_likes = list(month_data.values())
            
            # Sort chronologically
            months_likes = sorted(zip(months, avg_likes), key=lambda x: x[0])
            months = [item[0] for item in months_likes]
            avg_likes = [item[1] for item in months_likes]
            
            fig_trend = px.line(
                x=months,
                y=avg_likes,
                labels={'x': 'Month', 'y': 'Average Likes'},
                title='Average Likes by Month'
            )
            fig_trend.update_layout(template='plotly_white')
            visualizations['like_trend'] = self.save_plot(fig_trend, 'like_trend.json')
            
            # Engagement trend over time if available
            if 'engagement_by_month' in self.results['like_trends']:
                eng_month_data = self.results['like_trends']['engagement_by_month']
                eng_months = list(eng_month_data.keys())
                avg_eng = list(eng_month_data.values())
                
                # Sort chronologically
                months_eng = sorted(zip(eng_months, avg_eng), key=lambda x: x[0])
                eng_months = [item[0] for item in months_eng]
                avg_eng = [item[1] for item in months_eng]
                
                fig_eng_trend = px.line(
                    x=eng_months,
                    y=avg_eng,
                    labels={'x': 'Month', 'y': 'Average Engagement Rate (%)'},
                    title='Average Engagement Rate by Month'
                )
                fig_eng_trend.update_layout(template='plotly_white')
                visualizations['engagement_trend'] = self.save_plot(fig_eng_trend, 'engagement_trend.json')
        
        self.results['visualizations'] = visualizations
        return visualizations 