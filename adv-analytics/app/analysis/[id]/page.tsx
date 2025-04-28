'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import api, { AnalysisResults, AnalysisItem, DatasetStats } from '../../services/api';

interface TopVideoItem {
  title: string;
  views?: number;
  engagement_rate?: number;
}

interface TagItem {
  tag: string;
  count: number;
}

// Convert backend stats to frontend analysis format or pass through if already in correct format
const processAnalysisData = (data: any): AnalysisResults => {
  // If data already has 'results' array, it's already in the correct format
  if (data && data.results && Array.isArray(data.results)) {
    console.log("Data already in correct format");
    return data as AnalysisResults;
  }
  
  // Otherwise, convert from stats format to analysis format
  console.log("Converting stats data to analysis format");
  const stats = data as DatasetStats;
  const results: AnalysisItem[] = [
    {
      title: "Total Videos",
      data: { value: stats.total_videos }
    },
    {
      title: "Total Views",
      data: { value: stats.total_views }
    },
    {
      title: "Average Engagement Rate",
      data: { value: stats.avg_engagement_rate }
    },
    {
      title: "Average Like Ratio",
      data: { value: stats.avg_like_ratio }
    }
  ];
  
  // Add top videos if available
  if (stats.top_videos && Array.isArray(stats.top_videos)) {
    results.push({
      title: "Top Videos by Views",
      data: { 
        value: stats.top_videos.map(video => ({
          title: video.title,
          views: video.views
        }))
      }
    });
  }
  
  return { results };
};

export default function AnalysisPage() {
  const params = useParams();
  const id = params?.id as string;
  const [analysisData, setAnalysisData] = useState<AnalysisResults | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const datasetId = parseInt(id);

  useEffect(() => {
    const fetchAnalysis = async () => {
      if (isNaN(datasetId)) {
        setError('Invalid dataset ID');
        setIsLoading(false);
        return;
      }

      console.log(`Fetching analysis for dataset ID: ${datasetId}`);
      try {
        setIsLoading(true);
        
        // First try to get complete analysis data
        try {
          const analysisData = await api.getAnalysis(datasetId);
          console.log('Analysis data received:', analysisData);
          
          // Process the data to ensure correct format
          const formattedData = processAnalysisData(analysisData);
          setAnalysisData(formattedData);
          setError(null);
        } catch (analysisError) {
          console.error('Error fetching analysis, trying stats:', analysisError);
          
          // Fallback to stats
          const statsData = await api.getDatasetStats(datasetId);
          console.log('Stats data received:', statsData);
          
          // Convert stats to analysis format
          const formattedData = processAnalysisData(statsData);
          setAnalysisData(formattedData);
          setError(null);
        }
      } catch (err) {
        console.error('Error fetching all data:', err);
        setError('Failed to load analysis data. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalysis();
  }, [datasetId]);

  // Helper to render different types of analysis data
  const renderAnalysisItem = (item: AnalysisItem, index: number) => {
    const { title, data } = item;
    const value = data.value;
    
    // For simple numeric metrics
    if (typeof value === 'number') {
      return (
        <div className="stats-card fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
          <div className="stats-card-title">{title}</div>
          <div className="stats-card-value">
            {Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2)}
          </div>
        </div>
      );
    }
    
    // For top videos list
    if (Array.isArray(value) && value.length > 0 && 'title' in value[0]) {
      const videoItems = value as TopVideoItem[];
      const metricType = videoItems[0]?.views !== undefined ? 'views' : 'engagement_rate';
      
      // Sort handler for top videos
      const handleSort = (property: 'views' | 'engagement_rate' | 'title') => {
        // Create a sorted copy of the data
        const sortedItems = [...videoItems];
        
        if (property === 'title') {
          // Sort by title
          sortedItems.sort((a, b) => {
            const titleA = a.title.toLowerCase();
            const titleB = b.title.toLowerCase();
            
            if (titleA < titleB) {
              return -1;
            }
            if (titleA > titleB) {
              return 1;
            }
            return 0;
          });
        } else {
          // Sort by views or engagement_rate
          sortedItems.sort((a, b) => {
            const valueA = property === 'views' ? (a.views || 0) : (a.engagement_rate || 0);
            const valueB = property === 'views' ? (b.views || 0) : (b.engagement_rate || 0);
            return valueB - valueA; // Default to descending
          });
        }
        
        // Update the data directly in the analysis data
        if (analysisData && analysisData.results) {
          const updatedResults = [...analysisData.results];
          const itemIndex = updatedResults.findIndex(i => i.title === title);
          
          if (itemIndex !== -1) {
            updatedResults[itemIndex] = {
              ...updatedResults[itemIndex],
              data: {
                value: sortedItems
              }
            };
            
            setAnalysisData({
              ...analysisData,
              results: updatedResults
            });
          }
        }
      };

      return (
        <div className="card fade-in" style={{ gridColumn: 'span 3', animationDelay: `${index * 0.1}s` }}>
          <h3 className="section-title">{title}</h3>
          <div className="flex mb-2">
            <button 
              onClick={() => handleSort('title')}
              className="button button-small mr-2" 
              style={{fontSize: '0.7rem', padding: '2px 6px'}}
            >
              Sort by Title
            </button>
            <button 
              onClick={() => handleSort(metricType as 'views' | 'engagement_rate')}
              className="button button-small"
              style={{fontSize: '0.7rem', padding: '2px 6px'}}
            >
              Sort by {metricType === 'views' ? 'Views' : 'Engagement'}
            </button>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', width: '70%' }}>Title</th>
                  <th style={{ textAlign: 'right' }}>
                    {metricType === 'views' ? 'Views' : 'Engagement Rate'}
                  </th>
                </tr>
              </thead>
              <tbody>
                {videoItems.map((item, index) => (
                  <tr key={index} className="fade-in" style={{ animationDelay: `${0.2 + index * 0.05}s` }}>
                    <td style={{ paddingRight: '1rem' }}>
                      <div className="truncate" style={{ maxWidth: '100%' }} title={item.title}>
                        {item.title}
                      </div>
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: '500' }}>
                      {item.views !== undefined 
                        ? <span>{item.views.toLocaleString()}</span> 
                        : <span style={{ color: 'var(--primary)' }}>{(item.engagement_rate as number * 100).toFixed(2)}%</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }
    
    // For tag distributions
    if (Array.isArray(value) && value.length > 0 && 'tag' in value[0]) {
      const tagItems = value as TagItem[];
      return (
        <div className="card fade-in" style={{ gridColumn: 'span 3', animationDelay: `${index * 0.1}s` }}>
          <h3 className="section-title">{title}</h3>
          <div className="flex" style={{ flexWrap: 'wrap', gap: '0.75rem', marginTop: '1rem' }}>
            {tagItems.map((item, idx) => (
              <div 
                key={idx} 
                className="badge fade-in" 
                style={{ 
                  animationDelay: `${0.3 + idx * 0.05}s`,
                  fontSize: `${Math.max(0.75, Math.min(1.2, 0.8 + (item.count / 20) * 0.4))}rem`,
                  padding: '0.35rem 0.85rem',
                  backgroundColor: `rgba(59, 130, 246, ${Math.min(0.9, Math.max(0.1, item.count / 30))})`,
                  color: `rgba(255, 255, 255, ${Math.min(1, Math.max(0.6, item.count / 15))})`
                }}
              >
                {item.tag} <span style={{ marginLeft: '0.4rem', opacity: 0.8 }}>({item.count})</span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    
    // For duration distribution (object with ranges)
    if (typeof value === 'object' && !Array.isArray(value) && value !== null) {
      const distributionData = value as Record<string, number>;
      const total = Object.values(distributionData).reduce((sum, count) => sum + count, 0);
      
      return (
        <div className="card fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
          <h3 className="section-title">{title}</h3>
          <div style={{ marginTop: '1rem' }}>
            {Object.entries(distributionData).map(([range, count], idx) => {
              const percentage = total > 0 ? (count / total) * 100 : 0;
              return (
                <div key={range} className="mb-2 fade-in" style={{ animationDelay: `${0.2 + idx * 0.1}s` }}>
                  <div className="flex-between mb-1">
                    <span>{range}</span>
                    <span style={{ fontWeight: '600' }}>{count} <span style={{ color: '#94a3b8', fontSize: '0.85rem' }}>({percentage.toFixed(1)}%)</span></span>
                  </div>
                  <div className="progress-bar-container">
                    <div className="progress-bar" style={{ width: `${percentage}%` }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      );
    }
    
    // Fallback for other types
    return (
      <div className="card fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
        <h3 className="section-title">{title}</h3>
        <pre style={{ fontSize: '0.875rem', overflowX: 'auto' }}>{JSON.stringify(value, null, 2)}</pre>
      </div>
    );
  };

  return (
    <main className="main">
      <div className="container">
        <div className="flex-between mb-6">
          <h1 className="page-title slide-up">Dataset Analysis</h1>
          <div className="flex slide-up">
            <button 
              onClick={() => router.push('/')} 
              className="button button-secondary mr-2"
            >
              Back to Datasets
            </button>
            <button 
              onClick={() => router.push(`/dataset/${datasetId}`)}
              className="button"
            >
              View Videos
            </button>
          </div>
        </div>

        {isLoading ? (
          <div className="flex-center" style={{ height: '300px' }}>
            <div className="pulse" style={{ 
              width: '50px', 
              height: '50px', 
              borderRadius: '50%', 
              border: '4px solid rgba(59, 130, 246, 0.3)',
              borderTopColor: 'var(--primary)',
              animation: 'spin 1s linear infinite'
            }}></div>
            <style jsx>{`
              @keyframes spin {
                to { transform: rotate(360deg); }
              }
            `}</style>
          </div>
        ) : error ? (
          <div className="error-container fade-in" style={{ backgroundColor: '#fee2e2', borderLeft: '4px solid #ef4444', color: '#b91c1c', padding: '1rem', marginBottom: '1rem' }}>
            <p>{error}</p>
          </div>
        ) : analysisData && analysisData.results ? (
          <div className="grid">
            {analysisData.results.map((item, index) => (
              <div 
                key={index} 
                style={{ 
                  gridColumn: (item.title.includes('Top') || item.title.includes('Tag')) ? 'span 3' : 'span 1',
                }}
              >
                {renderAnalysisItem(item, index)}
              </div>
            ))}
          </div>
        ) : (
          <div className="card flex-center fade-in" style={{ padding: '3rem', backgroundColor: 'rgba(255, 255, 255, 0.8)' }}>
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: '#94a3b8', marginBottom: '1rem' }}><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>
            <p style={{ color: '#64748b', textAlign: 'center' }}>No analysis data available for this dataset.</p>
          </div>
        )}
      </div>
    </main>
  );
} 