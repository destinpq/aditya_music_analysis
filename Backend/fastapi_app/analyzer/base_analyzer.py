import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import plotly.utils as pu
from abc import ABC, abstractmethod


class BaseAnalyzer(ABC):
    """Base class for all data analyzers"""
    
    def __init__(self, df=None, column_name=None):
        """Initialize with a dataframe and target column name"""
        self.df = df
        self.column_name = column_name
        self.plot_path = os.path.join('static', 'plots')
        self.results = {}
        self.ai_predictions = {}
        
    def set_data(self, df, column_name):
        """Set or update the dataframe and column name"""
        self.df = df
        self.column_name = column_name
        return self
        
    def validate_data(self):
        """Check if the data and column are valid"""
        if self.df is None:
            raise ValueError("No dataframe provided")
        
        if self.column_name is None:
            raise ValueError("No column name provided")
            
        if self.column_name not in self.df.columns:
            # Try to find a matching column
            matching_cols = [col for col in self.df.columns 
                            if self.get_column_type().lower() in col.lower()]
            
            if matching_cols:
                self.column_name = matching_cols[0]
            else:
                raise ValueError(f"Column {self.column_name} not found and no matching columns")
                
        return True
    
    @abstractmethod
    def get_column_type(self):
        """Return the type of column this analyzer handles"""
        pass
        
    @abstractmethod
    def analyze(self):
        """Perform analysis on the data"""
        pass
    
    def create_visualization(self):
        """Create visualization based on analysis results"""
        pass
    
    def predict_ai_performance(self):
        """Generate predictions for AI-generated video performance"""
        # Base implementation - should be overridden by subclasses
        if not self.results:
            self.analyze()
        
        # Basic prediction template
        self.ai_predictions = {
            'current_metrics': {},
            'ai_potential': {},
            'improvement_factors': {},
            'recommendations': {}
        }
        
        return self.ai_predictions
    
    def generate_comparative_visualization(self):
        """Create visualizations comparing actual data with AI predictions"""
        # Base implementation - should be overridden by subclasses
        if not self.ai_predictions:
            self.predict_ai_performance()
        
        # This method should be implemented by each analyzer
        return {}
    
    def to_json(self):
        """Convert analysis results to JSON"""
        combined_results = {
            'analysis': self.results,
            'ai_predictions': self.ai_predictions
        }
        return json.dumps(combined_results)
    
    def generate_report(self):
        """Generate a text report of the analysis"""
        report = [f"# {self.get_column_type()} Analysis"]
        
        for key, value in self.results.items():
            if isinstance(value, (int, float, str, bool)):
                report.append(f"- {key}: {value}")
            elif isinstance(value, dict):
                report.append(f"- {key}:")
                for sub_key, sub_value in value.items():
                    report.append(f"  - {sub_key}: {sub_value}")
        
        # Add AI predictions if available
        if self.ai_predictions:
            report.append("\n## AI Video Generation Predictions")
            for key, value in self.ai_predictions.items():
                if isinstance(value, (int, float, str, bool)):
                    report.append(f"- {key}: {value}")
                elif isinstance(value, dict):
                    report.append(f"- {key}:")
                    for sub_key, sub_value in value.items():
                        report.append(f"  - {sub_key}: {sub_value}")
        
        return "\n".join(report)
    
    def _convert_to_serializable(self, data):
        """Helper method to convert data to JSON serializable format"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
        else:
            return data
    
    def save_plot(self, fig, filename):
        """Save a plotly figure to a file and return the JSON"""
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path, exist_ok=True)
            
        plot_json = json.dumps(fig.to_dict(), cls=pu.PlotlyJSONEncoder)
        
        with open(os.path.join(self.plot_path, filename), 'w') as f:
            f.write(plot_json)
            
        return plot_json 