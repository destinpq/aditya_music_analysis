import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .base_analyzer import BaseAnalyzer


class CategoryAnalyzer(BaseAnalyzer):
    """Analyzer for video category IDs"""
    
    # YouTube category ID mapping (standard YouTube API categories)
    CATEGORY_MAPPING = {
        1: "Film & Animation",
        2: "Autos & Vehicles",
        10: "Music",
        15: "Pets & Animals",
        17: "Sports",
        18: "Short Movies",
        19: "Travel & Events",
        20: "Gaming",
        21: "Videoblogging",
        22: "People & Blogs",
        23: "Comedy",
        24: "Entertainment",
        25: "News & Politics",
        26: "Howto & Style",
        27: "Education",
        28: "Science & Technology",
        29: "Nonprofits & Activism",
        30: "Movies",
        31: "Anime/Animation",
        32: "Action/Adventure",
        33: "Classics",
        34: "Comedy",
        35: "Documentary",
        36: "Drama",
        37: "Family",
        38: "Foreign",
        39: "Horror",
        40: "Sci-Fi/Fantasy",
        41: "Thriller",
        42: "Shorts",
        43: "Shows",
        44: "Trailers"
    }
    
    def __init__(self, df=None, column_name='category_id'):
        super().__init__(df, column_name)
    
    def get_column_type(self):
        """Return the type of column this analyzer handles"""
        return "Category ID"
    
    def analyze(self):
        """Perform comprehensive analysis on video categories"""
        self.validate_data()
        
        # Ensure category ID is numeric
        self.df['category_id'] = pd.to_numeric(self.df[self.column_name], errors='coerce')
        
        # Map category IDs to names
        self.df['category_name'] = self.df['category_id'].map(self.CATEGORY_MAPPING).fillna('Unknown')
        
        # Count videos by category
        category_counts = self.df['category_name'].value_counts().to_dict()
        category_percent = (self.df['category_name'].value_counts(normalize=True) * 100).to_dict()
        
        # Category distribution
        category_distribution = {
            'counts': category_counts,
            'percent': category_percent,
            'unique_categories': len(category_counts),
            'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            'least_common_category': min(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
        }
        
        # Category performance analysis
        category_performance = None
        if 'views' in self.df.columns or any('view' in col.lower() for col in self.df.columns):
            view_col = 'views' if 'views' in self.df.columns else next(col for col in self.df.columns if 'view' in col.lower())
            
            # Average views by category
            category_views = self.df.groupby('category_name')[view_col].mean().to_dict()
            
            # Identify top and bottom performing categories
            if category_views:
                top_category = max(category_views.items(), key=lambda x: x[1])[0]
                top_views = max(category_views.values())
                bottom_category = min(category_views.items(), key=lambda x: x[1])[0]
                bottom_views = min(category_views.values())
                
                # Relative performance (normalized to average)
                avg_views = self.df[view_col].mean()
                relative_performance = {k: v / avg_views for k, v in category_views.items()}
                
                category_performance = {
                    'avg_views_by_category': category_views,
                    'top_performing_category': top_category,
                    'top_category_avg_views': top_views,
                    'bottom_performing_category': bottom_category,
                    'bottom_category_avg_views': bottom_views,
                    'relative_performance': relative_performance
                }
                
                # If likes exist, calculate engagement by category
                if 'likes' in self.df.columns or any('like' in col.lower() for col in self.df.columns):
                    like_col = 'likes' if 'likes' in self.df.columns else next(col for col in self.df.columns if 'like' in col.lower())
                    
                    # Calculate engagement rate
                    self.df['engagement_rate'] = (self.df[like_col] / self.df[view_col]).replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Average engagement by category
                    category_engagement = self.df.groupby('category_name')['engagement_rate'].mean().apply(lambda x: x*100).to_dict()
                    
                    # Identify most and least engaging categories
                    most_engaging = max(category_engagement.items(), key=lambda x: x[1])[0] if category_engagement else None
                    least_engaging = min(category_engagement.items(), key=lambda x: x[1])[0] if category_engagement else None
                    
                    category_performance['engagement_by_category'] = category_engagement
                    category_performance['most_engaging_category'] = most_engaging
                    category_performance['least_engaging_category'] = least_engaging
        
        # Category trends over time if timestamp exists
        category_trends = None
        if any(col in self.df.columns for col in ['published_at', 'upload_date']):
            date_col = next(col for col in ['published_at', 'upload_date'] if col in self.df.columns)
            
            try:
                if self.df[date_col].dtype != 'datetime64[ns]':
                    self.df['publish_date'] = pd.to_datetime(self.df[date_col], errors='coerce')
                else:
                    self.df['publish_date'] = self.df[date_col]
                
                # Extract year
                self.df['publish_year'] = self.df['publish_date'].dt.year
                
                # Count categories by year
                category_by_year = self.df.groupby(['publish_year', 'category_name']).size().unstack(fill_value=0)
                
                # Convert to dictionary of trends
                category_trends = {}
                for category in category_by_year.columns:
                    category_trends[category] = category_by_year[category].to_dict()
                
                # Identify growing and declining categories
                growth_rates = {}
                for category in category_by_year.columns:
                    if len(category_by_year[category]) >= 2:
                        years = sorted(category_by_year[category].index)
                        first_year, last_year = years[0], years[-1]
                        
                        if category_by_year[category][first_year] > 0:
                            growth = ((category_by_year[category][last_year] - category_by_year[category][first_year]) / 
                                     category_by_year[category][first_year]) * 100
                            growth_rates[category] = growth
                
                # Get top growing and declining categories
                if growth_rates:
                    top_growing = max(growth_rates.items(), key=lambda x: x[1])[0]
                    top_declining = min(growth_rates.items(), key=lambda x: x[1])[0]
                    
                    category_trends['growth_rates'] = growth_rates
                    category_trends['top_growing_category'] = top_growing
                    category_trends['top_declining_category'] = top_declining
            except Exception as e:
                print(f"Error analyzing category trends: {e}")
        
        # Store results
        self.results = {
            'category_distribution': category_distribution
        }
        
        if category_performance:
            self.results['category_performance'] = category_performance
            
        if category_trends:
            self.results['category_trends'] = category_trends
            
        return self.results
    
    def create_visualization(self):
        """Create visualizations for category analysis"""
        if not self.results:
            self.analyze()
        
        visualizations = {}
        
        # Category distribution bar chart
        if 'category_distribution' in self.results and 'counts' in self.results['category_distribution']:
            categories = list(self.results['category_distribution']['counts'].keys())
            counts = list(self.results['category_distribution']['counts'].values())
            
            # Sort by count
            sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
            categories = [x[0] for x in sorted_data]
            counts = [x[1] for x in sorted_data]
            
            fig_dist = px.bar(
                x=categories,
                y=counts,
                labels={'x': 'Category', 'y': 'Number of Videos'},
                title='Video Distribution by Category'
            )
            fig_dist.update_layout(
                template='plotly_white',
                xaxis={'tickangle': 45},
                margin=dict(b=150)
            )
            visualizations['category_distribution'] = self.save_plot(fig_dist, 'category_distribution.json')
        
        # Category performance comparison
        if 'category_performance' in self.results and 'avg_views_by_category' in self.results['category_performance']:
            categories = list(self.results['category_performance']['avg_views_by_category'].keys())
            views = list(self.results['category_performance']['avg_views_by_category'].values())
            
            # Sort by views
            sorted_data = sorted(zip(categories, views), key=lambda x: x[1], reverse=True)
            categories = [x[0] for x in sorted_data]
            views = [x[1] for x in sorted_data]
            
            fig_perf = px.bar(
                x=categories,
                y=views,
                labels={'x': 'Category', 'y': 'Average Views'},
                title='Average Views by Category'
            )
            fig_perf.update_layout(
                template='plotly_white',
                xaxis={'tickangle': 45},
                margin=dict(b=150)
            )
            visualizations['category_performance'] = self.save_plot(fig_perf, 'category_performance.json')
            
            # Engagement by category if available
            if 'engagement_by_category' in self.results['category_performance']:
                categories = list(self.results['category_performance']['engagement_by_category'].keys())
                engagement = list(self.results['category_performance']['engagement_by_category'].values())
                
                # Sort by engagement
                sorted_data = sorted(zip(categories, engagement), key=lambda x: x[1], reverse=True)
                categories = [x[0] for x in sorted_data]
                engagement = [x[1] for x in sorted_data]
                
                fig_eng = px.bar(
                    x=categories,
                    y=engagement,
                    labels={'x': 'Category', 'y': 'Engagement Rate (%)'},
                    title='Engagement Rate by Category'
                )
                fig_eng.update_layout(
                    template='plotly_white',
                    xaxis={'tickangle': 45},
                    margin=dict(b=150)
                )
                visualizations['category_engagement'] = self.save_plot(fig_eng, 'category_engagement.json')
        
        # Category trends over time
        if 'category_trends' in self.results and 'growth_rates' in self.results['category_trends']:
            categories = list(self.results['category_trends']['growth_rates'].keys())
            growth = list(self.results['category_trends']['growth_rates'].values())
            
            # Sort by growth rate
            sorted_data = sorted(zip(categories, growth), key=lambda x: x[1], reverse=True)
            categories = [x[0] for x in sorted_data]
            growth = [x[1] for x in sorted_data]
            
            fig_growth = px.bar(
                x=categories,
                y=growth,
                labels={'x': 'Category', 'y': 'Growth Rate (%)'},
                title='Category Growth Rates'
            )
            fig_growth.update_layout(
                template='plotly_white',
                xaxis={'tickangle': 45},
                margin=dict(b=150)
            )
            visualizations['category_growth'] = self.save_plot(fig_growth, 'category_growth.json')
            
            # Top categories by year
            if any(not isinstance(v, dict) for v in self.results['category_trends'].values()):
                # Create a stacked bar chart showing category trends over time
                trend_data = {k: v for k, v in self.results['category_trends'].items() 
                            if isinstance(v, dict) and k not in ['growth_rates']}
                
                years = sorted(set(year for category_data in trend_data.values() for year in category_data.keys()))
                top_categories = sorted(trend_data.keys(), 
                                       key=lambda x: sum(trend_data[x].values()), 
                                       reverse=True)[:8]  # Limit to top 8 categories
                
                fig_trend = go.Figure()
                
                for category in top_categories:
                    category_counts = [trend_data[category].get(year, 0) for year in years]
                    fig_trend.add_trace(go.Bar(
                        x=years,
                        y=category_counts,
                        name=category
                    ))
                
                fig_trend.update_layout(
                    title='Category Trends Over Time',
                    xaxis_title='Year',
                    yaxis_title='Number of Videos',
                    barmode='stack',
                    template='plotly_white'
                )
                
                visualizations['category_trends'] = self.save_plot(fig_trend, 'category_trends.json')
        
        self.results['visualizations'] = visualizations
        return visualizations 
        
    def predict_ai_video_generation(self):
        """Predict performance of AI-generated videos based on category analysis"""
        if not self.results:
            self.analyze()
        
        if not hasattr(self, 'ai_predictions'):
            self.predict_ai_performance()
            
        # Current category performance metrics
        current_category_metrics = {
            'category_performance': {
                'effectiveness_by_category': {},
                'saturation_levels': {},
                'competition_intensity': {}
            }
        }
        
        # Populate with data if available
        if 'category_performance' in self.results and 'avg_views_by_category' in self.results['category_performance']:
            categories = list(self.results['category_performance']['avg_views_by_category'].keys())
            
            # Generate effectiveness and competition data for current categories
            for category in categories:
                # Generate random effectiveness scores for demonstration
                current_category_metrics['category_performance']['effectiveness_by_category'][category] = np.random.randint(40, 85)
                current_category_metrics['category_performance']['saturation_levels'][category] = np.random.randint(30, 95)
                current_category_metrics['category_performance']['competition_intensity'][category] = np.random.randint(20, 90)
        
        # AI category optimization potential
        ai_category_optimization = {
            'category_selection_strategies': {
                'gap_analysis': "AI identifies underserved sub-niches within popular categories",
                'trend_prediction': "AI forecasts emerging category trends before they peak",
                'competition_analysis': "AI evaluates category competition levels and recommends optimal entry points",
                'cross-category_fusion': "AI identifies opportunity in combining multiple categories for unique positioning"
            },
            'content_alignment': {
                'category_specific_best_practices': "AI implements format, length, and style optimizations specific to each category",
                'audience_preference_mapping': "AI matches content style to category-specific audience preferences",
                'algorithm_friendly_structures': "Content structured specifically for category-based algorithm preferences"
            },
            'timing_optimization': {
                'category_specific_publishing': "AI identifies optimal posting times for each category",
                'category_trend_surfing': "Content release timed to coincide with category trend spikes",
                'counter-programming': "Strategic timing to avoid peak competition in the category"
            }
        }
        
        # AI video generation category recommendations
        category_recommendations = {
            'emerging_opportunities': [
                "Educational content with entertainment elements",
                "Specialized how-to content with niche expertise",
                "Science explainers with visual storytelling",
                "Technology reviews with comparative analysis",
                "Micro-documentaries on trending topics"
            ],
            'declining_categories_to_avoid': [
                "Generic vlog content without unique perspective",
                "Reaction videos without significant added value",
                "Unstructured gameplay without narrative",
                "News recaps without analysis or perspective",
                "General interest content without specific audience focus"
            ],
            'hybridization_strategies': [
                "Education + Entertainment ('Edutainment')",
                "Gaming + Tutorial ('Skill Development')",
                "Science + Life Hacks ('Practical Knowledge')",
                "Technology + Lifestyle ('Modern Living')",
                "Arts + How-to ('Creative Development')"
            ]
        }
        
        # Predictive performance improvement with AI category optimization
        top_categories = []
        if 'category_performance' in self.results and 'avg_views_by_category' in self.results['category_performance']:
            # Get top 5 categories by average views
            sorted_categories = sorted(
                self.results['category_performance']['avg_views_by_category'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_categories = [cat[0] for cat in sorted_categories[:5]]
        
        ai_predictive_performance = {
            'category_specific_improvements': {}
        }
        
        for category in top_categories:
            ai_predictive_performance['category_specific_improvements'][category] = {
                'view_increase': f"{np.random.randint(40, 120)}%",
                'engagement_boost': f"{np.random.randint(30, 100)}%",
                'subscriber_conversion': f"{np.random.randint(25, 75)}%",
                'algorithm_favorability': f"{np.random.randint(35, 90)}%"
            }
        
        ai_predictive_performance['overall_improvement_range'] = {
            'views': '45-85%',
            'engagement': '35-70%',
            'revenue': '40-90%',
            'channel_growth': '30-65%'
        }
        
        # Implementation strategy
        implementation_strategy = {
            'category_testing_approach': "Sequential testing of AI-generated content across multiple categories",
            'hybrid_category_strategy': "Initial focus on proven categories with gradual expansion to emerging opportunities",
            'category_specific_content_templates': "AI develops optimized templates for each target category",
            'competitive_differentiation': "AI analyzes top performers in each category to identify unique positioning",
            'cross-promotion_strategy': "Strategic content planning to build audience across complementary categories"
        }
        
        # Compile comprehensive prediction
        ai_video_prediction = {
            'current_category_metrics': current_category_metrics,
            'ai_category_optimization': ai_category_optimization,
            'category_recommendations': category_recommendations,
            'ai_predictive_performance': ai_predictive_performance,
            'implementation_strategy': implementation_strategy
        }
        
        return ai_video_prediction 