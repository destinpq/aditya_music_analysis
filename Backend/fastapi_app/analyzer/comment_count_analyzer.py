import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
from .base_analyzer import BaseAnalyzer


class CommentCountAnalyzer(BaseAnalyzer):
    """Analyzer for video comment counts and audience interaction"""
    
    def __init__(self, df=None, column_name='comment_count'):
        super().__init__(df, column_name)
    
    def get_column_type(self):
        """Return the type of column this analyzer handles"""
        return "Comment Count"
    
    def analyze(self):
        """Perform comprehensive analysis on video comments"""
        self.validate_data()
        
        # Ensure comment count is numeric
        self.df['comments'] = pd.to_numeric(self.df[self.column_name], errors='coerce')
        
        # Basic comment count statistics
        comment_stats = {
            'total_comments': self.df['comments'].sum(),
            'avg_comments': self.df['comments'].mean(),
            'median_comments': self.df['comments'].median(),
            'min_comments': self.df['comments'].min(),
            'max_comments': self.df['comments'].max(),
            'std_comments': self.df['comments'].std()
        }
        
        # Create comment count categories
        self.df['comment_category'] = pd.cut(
            self.df['comments'],
            bins=[0, 5, 10, 25, 50, 100, 200, 500, 1000, float('inf')],
            labels=['0-5', '6-10', '11-25', '26-50', '51-100', '101-200', '201-500', '501-1000', '>1000'],
            include_lowest=True
        )
        
        # Distribution of videos by comment count category
        comment_distribution = self.df['comment_category'].value_counts().sort_index().to_dict()
        
        # Percentile analysis
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_values = {f"{p}th_percentile": np.percentile(self.df['comments'].dropna(), p) for p in percentiles}
        
        # Discussion activity analysis
        # Ratio of comments to views
        discussion_activity = None
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            # Calculate discussion rate (comments per 1000 views)
            self.df['discussion_rate'] = (self.df['comments'] / self.df[view_col] * 1000).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate discussion stats
            discussion_stats = {
                'avg_discussion_rate': self.df['discussion_rate'].mean(),
                'median_discussion_rate': self.df['discussion_rate'].median(),
                'max_discussion_rate': self.df['discussion_rate'].max(),
                'min_discussion_rate': self.df['discussion_rate'].min()
            }
            
            # Create discussion categories
            self.df['discussion_category'] = pd.cut(
                self.df['discussion_rate'],
                bins=[0, 1, 2, 5, 10, 20, 50, 100, float('inf')],
                labels=['<1', '1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '>100'],
                include_lowest=True
            )
            
            discussion_distribution = self.df['discussion_category'].value_counts().sort_index().to_dict()
            
            # Find correlation between comments and views
            try:
                comment_view_corr, p_value = pearsonr(self.df['comments'].dropna(), self.df[view_col].dropna())
                correlation_analysis = {
                    'comments_views_correlation': comment_view_corr,
                    'correlation_p_value': p_value,
                    'correlation_strength': self._interpret_correlation(comment_view_corr)
                }
            except:
                correlation_analysis = {
                    'comments_views_correlation': None,
                    'correlation_strength': 'Could not calculate'
                }
            
            # Identify videos with high and low discussion
            avg_discussion = self.df['discussion_rate'].mean()
            std_discussion = self.df['discussion_rate'].std()
            
            high_discussion = self.df[self.df['discussion_rate'] > avg_discussion + 2*std_discussion]
            low_discussion = self.df[self.df['discussion_rate'] < avg_discussion - std_discussion]
            
            # Get details for top discussed videos
            title_column = next((col for col in self.df.columns if any(term in col.lower() for term in ['title', 'name', 'video'])), None)
            
            high_discussion_videos = []
            if title_column and not high_discussion.empty:
                for _, row in high_discussion.head(5).iterrows():
                    high_discussion_videos.append({
                        'title': row[title_column] if not pd.isna(row[title_column]) else f"Video {_}",
                        'discussion_rate': row['discussion_rate'],
                        'comments': row['comments'],
                        'views': row[view_col]
                    })
            
            # Store discussion analysis
            discussion_activity = {
                'discussion_stats': discussion_stats,
                'discussion_distribution': discussion_distribution,
                'correlation_analysis': correlation_analysis,
                'high_discussion_videos': high_discussion_videos,
                'high_discussion_count': len(high_discussion),
                'low_discussion_count': len(low_discussion)
            }
        
        # Comment and likes relationship
        comment_likes_relation = None
        if 'likes' in self.df.columns or any('like' in col.lower() for col in self.df.columns):
            like_col = 'likes' if 'likes' in self.df.columns else next(col for col in self.df.columns if 'like' in col.lower())
            
            # Calculate comments to likes ratio
            self.df['comments_to_likes_ratio'] = (self.df['comments'] / self.df[like_col]).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate ratio stats
            ratio_stats = {
                'avg_comments_per_like': self.df['comments_to_likes_ratio'].mean(),
                'median_comments_per_like': self.df['comments_to_likes_ratio'].median(),
                'comments_per_1000_likes': self.df['comments'].sum() / self.df[like_col].sum() * 1000 if self.df[like_col].sum() > 0 else 0
            }
            
            # Find correlation between comments and likes
            try:
                comment_like_corr, p_value = pearsonr(self.df['comments'].dropna(), self.df[like_col].dropna())
                correlation_analysis = {
                    'comments_likes_correlation': comment_like_corr,
                    'correlation_p_value': p_value,
                    'correlation_strength': self._interpret_correlation(comment_like_corr)
                }
            except:
                correlation_analysis = {
                    'comments_likes_correlation': None,
                    'correlation_strength': 'Could not calculate'
                }
            
            comment_likes_relation = {
                'ratio_stats': ratio_stats,
                'correlation_analysis': correlation_analysis
            }
        
        # Comment count trends over time if timestamp exists
        comment_trends = None
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
                
                # Calculate average comments by year
                comments_by_year = self.df.groupby('publish_year')['comments'].mean().to_dict()
                
                # Calculate comments by month-year
                comments_by_month = self.df.groupby('publish_yearmonth')['comments'].mean().to_dict()
                
                # Add to trends
                comment_trends = {
                    'by_year': comments_by_year,
                    'by_month': comments_by_month
                }
                
                # Calculate discussion rate trends if views exist
                if discussion_activity:
                    discussion_by_year = self.df.groupby('publish_year')['discussion_rate'].mean().to_dict()
                    discussion_by_month = self.df.groupby('publish_yearmonth')['discussion_rate'].mean().to_dict()
                    
                    comment_trends['discussion_by_year'] = discussion_by_year
                    comment_trends['discussion_by_month'] = discussion_by_month
            except Exception as e:
                print(f"Error analyzing comment trends: {e}")
        
        # Analyze the impact of comments on performance
        comment_impact = None
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            # Create high/low comment groups
            median_comments = self.df['comments'].median()
            high_comment_views = self.df[self.df['comments'] > median_comments][view_col].mean()
            low_comment_views = self.df[self.df['comments'] <= median_comments][view_col].mean()
            
            comment_impact = {
                'high_comment_videos_avg_views': high_comment_views,
                'low_comment_videos_avg_views': low_comment_views,
                'view_difference_pct': ((high_comment_views - low_comment_views) / low_comment_views * 100) 
                                       if low_comment_views > 0 else 0
            }
        
        # Store results
        self.results = {
            'comment_stats': comment_stats,
            'comment_distribution': comment_distribution,
            'percentiles': percentile_values
        }
        
        if discussion_activity:
            self.results['discussion_activity'] = discussion_activity
            
        if comment_likes_relation:
            self.results['comment_likes_relation'] = comment_likes_relation
            
        if comment_trends:
            self.results['comment_trends'] = comment_trends
            
        if comment_impact:
            self.results['comment_impact'] = comment_impact
            
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
        """Create visualizations for comment count analysis"""
        if not self.results:
            self.analyze()
        
        visualizations = {}
        
        # Comment count distribution histogram
        fig_hist = px.histogram(
            self.df,
            x='comments',
            log_x=True,  # Log scale for better visualization of skewed data
            nbins=30,
            labels={'comments': 'Comment Count (log scale)'},
            title='Distribution of Video Comment Counts'
        )
        fig_hist.update_layout(template='plotly_white', showlegend=False)
        visualizations['comment_hist'] = self.save_plot(fig_hist, 'comment_histogram.json')
        
        # Videos by comment count category
        categories = list(self.results['comment_distribution'].keys())
        counts = list(self.results['comment_distribution'].values())
        
        fig_cats = px.bar(
            x=categories,
            y=counts,
            labels={'x': 'Comment Count Category', 'y': 'Number of Videos'},
            title='Videos by Comment Count Category'
        )
        fig_cats.update_layout(template='plotly_white')
        visualizations['comment_categories'] = self.save_plot(fig_cats, 'comment_categories.json')
        
        # Comments vs. Views scatter plot
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            fig_scatter = px.scatter(
                self.df,
                x=view_col,
                y='comments',
                log_x=True,
                log_y=True,
                labels={'comments': 'Comments (log scale)', view_col: 'Views (log scale)'},
                title='Comments vs. Views',
                opacity=0.7,
                trendline='ols'
            )
            fig_scatter.update_layout(template='plotly_white')
            visualizations['comments_vs_views'] = self.save_plot(fig_scatter, 'comments_vs_views.json')
            
            # Discussion rate distribution
            if 'discussion_activity' in self.results:
                fig_discussion = px.histogram(
                    self.df,
                    x='discussion_rate',
                    nbins=30,
                    labels={'discussion_rate': 'Discussion Rate (Comments per 1000 Views)'},
                    title='Distribution of Discussion Rates'
                )
                fig_discussion.update_layout(template='plotly_white', showlegend=False)
                visualizations['discussion_dist'] = self.save_plot(fig_discussion, 'discussion_dist.json')
                
                # Discussion by category
                if 'discussion_distribution' in self.results['discussion_activity']:
                    disc_cats = list(self.results['discussion_activity']['discussion_distribution'].keys())
                    disc_counts = list(self.results['discussion_activity']['discussion_distribution'].values())
                    
                    fig_disc_cats = px.bar(
                        x=disc_cats,
                        y=disc_counts,
                        labels={'x': 'Discussion Rate Category', 'y': 'Number of Videos'},
                        title='Videos by Discussion Rate Category'
                    )
                    fig_disc_cats.update_layout(template='plotly_white')
                    visualizations['discussion_categories'] = self.save_plot(fig_disc_cats, 'discussion_categories.json')
        
        # Comments vs. Likes scatter plot
        if 'likes' in self.df.columns or any('like' in col.lower() for col in self.df.columns):
            like_col = 'likes' if 'likes' in self.df.columns else next(col for col in self.df.columns if 'like' in col.lower())
            
            fig_likes_scatter = px.scatter(
                self.df,
                x=like_col,
                y='comments',
                log_x=True,
                log_y=True,
                labels={'comments': 'Comments (log scale)', like_col: 'Likes (log scale)'},
                title='Comments vs. Likes',
                opacity=0.7,
                trendline='ols'
            )
            fig_likes_scatter.update_layout(template='plotly_white')
            visualizations['comments_vs_likes'] = self.save_plot(fig_likes_scatter, 'comments_vs_likes.json')
        
        # Comment trends over time if available
        if 'comment_trends' in self.results and 'by_month' in self.results['comment_trends']:
            month_data = self.results['comment_trends']['by_month']
            months = list(month_data.keys())
            avg_comments = list(month_data.values())
            
            # Sort chronologically
            months_comments = sorted(zip(months, avg_comments), key=lambda x: x[0])
            months = [item[0] for item in months_comments]
            avg_comments = [item[1] for item in months_comments]
            
            fig_trend = px.line(
                x=months,
                y=avg_comments,
                labels={'x': 'Month', 'y': 'Average Comments'},
                title='Average Comments by Month'
            )
            fig_trend.update_layout(template='plotly_white')
            visualizations['comment_trend'] = self.save_plot(fig_trend, 'comment_trend.json')
            
            # Discussion rate trend over time if available
            if 'discussion_by_month' in self.results['comment_trends']:
                disc_month_data = self.results['comment_trends']['discussion_by_month']
                disc_months = list(disc_month_data.keys())
                avg_disc = list(disc_month_data.values())
                
                # Sort chronologically
                months_disc = sorted(zip(disc_months, avg_disc), key=lambda x: x[0])
                disc_months = [item[0] for item in months_disc]
                avg_disc = [item[1] for item in months_disc]
                
                fig_disc_trend = px.line(
                    x=disc_months,
                    y=avg_disc,
                    labels={'x': 'Month', 'y': 'Average Discussion Rate'},
                    title='Average Discussion Rate by Month'
                )
                fig_disc_trend.update_layout(template='plotly_white')
                visualizations['discussion_trend'] = self.save_plot(fig_disc_trend, 'discussion_trend.json')
        
        # Comment impact visualization if available
        if 'comment_impact' in self.results:
            impact_data = self.results['comment_impact']
            
            fig_impact = go.Figure()
            fig_impact.add_trace(go.Bar(
                x=['High Comment Videos', 'Low Comment Videos'],
                y=[impact_data['high_comment_videos_avg_views'], impact_data['low_comment_videos_avg_views']],
                marker_color=['#1f77b4', '#ff7f0e']
            ))
            
            fig_impact.update_layout(
                title='Average Views by Comment Level',
                xaxis_title='Comment Level',
                yaxis_title='Average Views',
                template='plotly_white'
            )
            
            visualizations['comment_impact'] = self.save_plot(fig_impact, 'comment_impact.json')
        
        self.results['visualizations'] = visualizations
        return visualizations 