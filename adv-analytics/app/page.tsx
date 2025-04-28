'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import api, { Dataset } from './services/api';

export default function HomePage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [insights, setInsights] = useState<any[]>([]);
  const [recentModels, setRecentModels] = useState<any[]>([]);
  
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // Fetch datasets
        const datasetsResponse = await api.getDatasets();
        setDatasets(datasetsResponse);
        
        // Try to fetch ML models info
        try {
          const mlModelsResponse = await api.mlGetModelsInfo();
          setRecentModels(mlModelsResponse.slice(0, 3)); // Show only 3 most recent
        } catch (error) {
          console.error('Failed to fetch ML models:', error);
          // Non-critical error, don't show in UI
        }
        
        // Generate insights (placeholder for real analytics)
        generateInsights(datasetsResponse);
      } catch (err) {
        setError(`Error loading data: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  const generateInsights = (datasets: Dataset[]) => {
    // For demo, we'll create some placeholder insights
    // In a real app, these would come from actual analysis
    const generatedInsights = [
      {
        id: 1,
        title: 'Optimal Post Time',
        description: 'Videos posted on weekdays between 3-5pm get 27% more views',
        icon: '🕒',
        category: 'timing',
        link: '/ml-lab'
      },
      {
        id: 2,
        title: 'Content Length',
        description: 'Videos between 8-12 minutes have the highest engagement rate',
        icon: '📏',
        category: 'format',
        link: '/ml-lab'
      },
      {
        id: 3,
        title: 'Title Optimization',
        description: 'Including numbers in titles increases click-through rate by 15%',
        icon: '📝',
        category: 'content',
        link: '/ml-lab'
      },
      {
        id: 4,
        title: 'Topic Performance',
        description: 'Tutorial videos are currently outperforming other content types',
        icon: '📈',
        category: 'content',
        link: '/ml-lab'
      }
    ];
    
    setInsights(generatedInsights);
  };
  
  return (
    <div className="container">
      <div className="home-header">
        <h1 className="page-title">YouTube Analytics Dashboard</h1>
        <p className="home-description">
          Advanced video analytics and AI-powered insights for your YouTube content
        </p>
      </div>
      
      {isLoading ? (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading dashboard data...</p>
        </div>
      ) : error ? (
        <div className="error-container">
          <p className="error">{error}</p>
          <button 
            className="button button-primary mt-4"
            onClick={() => window.location.reload()}
          >
            Try Again
          </button>
        </div>
      ) : (
        <div className="dashboard-content">
          <div className="welcome-section">
            <div className="welcome-card">
              <div className="welcome-text">
                <h2>Welcome to Advanced YouTube Analytics</h2>
                <p>
                  Analyze your YouTube data using machine learning to gain deeper insights 
                  and make data-driven decisions for your content strategy.
                </p>
                {datasets.length === 0 ? (
                  <Link href="/dataset" className="button button-primary mt-4">
                    Upload Your First Dataset
                  </Link>
                ) : (
                  <Link href="/dashboard" className="button button-primary mt-4">
                    View Your Analytics
                  </Link>
                )}
              </div>
              <div className="welcome-image">
                <div className="analytics-graphic"></div>
              </div>
            </div>
          </div>
          
          <div className="dashboard-grid">
            <div className="dashboard-column">
              {/* Quick stats */}
              <div className="card stats-overview">
                <h2 className="section-title">Quick Overview</h2>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-icon datasets-icon">📊</div>
                    <div className="stat-value">{datasets.length}</div>
                    <div className="stat-label">Datasets</div>
                  </div>
                  
                  <div className="stat-card">
                    <div className="stat-icon models-icon">🧠</div>
                    <div className="stat-value">{recentModels.length}</div>
                    <div className="stat-label">ML Models</div>
                  </div>
                  
                  <div className="stat-card">
                    <div className="stat-icon videos-icon">🎥</div>
                    <div className="stat-value">
                      {datasets.length > 0 ? '1.2K' : '0'}
                    </div>
                    <div className="stat-label">Videos</div>
                  </div>
                  
                  <div className="stat-card">
                    <div className="stat-icon predictions-icon">🔮</div>
                    <div className="stat-value">
                      {datasets.length > 0 ? '23' : '0'}
                    </div>
                    <div className="stat-label">Predictions</div>
                  </div>
                </div>
              </div>
              
              {/* ML Insights */}
              <div className="card insights-card">
                <div className="card-header">
                  <h2 className="section-title">ML-Powered Insights</h2>
                  <Link href="/ml-lab" className="button button-small button-outline">
                    Train Models
                  </Link>
                </div>
                
                <div className="insights-grid">
                  {insights.map(insight => (
                    <div key={insight.id} className="insight-card">
                      <div className="insight-icon">{insight.icon}</div>
                      <div className="insight-content">
                        <h3 className="insight-title">{insight.title}</h3>
                        <p className="insight-description">{insight.description}</p>
                      </div>
                      <Link href={insight.link} className="insight-link">
                        <span className="insight-link-text">Learn more</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M12 5L19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </Link>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="dashboard-column">
              {/* Recent Datasets */}
              <div className="card recent-datasets">
                <div className="card-header">
                  <h2 className="section-title">Recent Datasets</h2>
                  <Link href="/dataset" className="button button-small button-outline">
                    View All
                  </Link>
                </div>
                
                {datasets.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">📈</div>
                    <h3>No datasets yet</h3>
                    <p>Upload your first dataset to start analyzing your YouTube data</p>
                    <Link href="/dataset" className="button button-primary mt-4">
                      Upload Dataset
                    </Link>
                  </div>
                ) : (
                  <div className="datasets-list">
                    {datasets.slice(0, 5).map(dataset => (
                      <Link 
                        href={`/dashboard/${dataset.id}`} 
                        key={dataset.id} 
                        className="dataset-item"
                      >
                        <div className="dataset-icon">📊</div>
                        <div className="dataset-info">
                          <h3 className="dataset-name">{dataset.filename}</h3>
                          <div className="dataset-meta">
                            <span className="dataset-date">
                              {new Date(dataset.uploadDate).toLocaleDateString()}
                            </span>
                            <span className="dataset-size">
                              {dataset.row_count ? `${dataset.row_count} rows` : 'Unknown size'}
                            </span>
                          </div>
                        </div>
                        <div className="dataset-arrow">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M9 18L15 12L9 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </div>
                      </Link>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Recent Models */}
              <div className="card recent-models">
                <div className="card-header">
                  <h2 className="section-title">ML Models</h2>
                  <Link href="/ml-lab" className="button button-small button-outline">
                    View All
                  </Link>
                </div>
                
                {recentModels.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">🧠</div>
                    <h3>No ML models yet</h3>
                    <p>Train your first machine learning model to get predictive insights</p>
                    <Link href="/ml-lab" className="button button-primary mt-4">
                      Train Models
                    </Link>
                  </div>
                ) : (
                  <div className="models-list">
                    {recentModels.map((model, index) => (
                      <div key={index} className="model-list-item">
                        <div className="model-info">
                          <h3 className="model-name">
                            Model for Dataset #{model.dataset_id}
                          </h3>
                          <div className="model-meta">
                            <span className="model-target">
                              Target: {model.target_column}
                            </span>
                            <span className="model-performance">
                              R² Score: {(model.evaluation[model.models[0]]?.r2_score * 100 || 0).toFixed(2)}%
                            </span>
                          </div>
                        </div>
                        <Link href="/ml-lab" className="model-action">
                          <span className="action-text">View</span>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M9 18L15 12L9 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </Link>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div className="actions-section">
            <div className="action-card">
              <div className="action-icon">📈</div>
              <h3 className="action-title">Upload New Dataset</h3>
              <p className="action-description">
                Upload your YouTube analytics data to gain insights from machine learning models
              </p>
              <Link href="/dataset/upload" className="button button-outline button-full">
                Upload Dataset
              </Link>
            </div>
            
            <div className="action-card">
              <div className="action-icon">🧠</div>
              <h3 className="action-title">Train AI Models</h3>
              <p className="action-description">
                Train custom machine learning models on your data to get personalized insights
              </p>
              <Link href="/ml-lab" className="button button-outline button-full">
                Train Models
              </Link>
            </div>
            
            <div className="action-card">
              <div className="action-icon">🔮</div>
              <h3 className="action-title">Predict Performance</h3>
              <p className="action-description">
                Use trained models to predict how your future videos will perform
              </p>
              <Link href="/ml-lab?tab=predict" className="button button-outline button-full">
                Make Prediction
              </Link>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
