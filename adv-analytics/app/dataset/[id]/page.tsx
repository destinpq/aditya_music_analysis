'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import api, { Video } from '../../services/api';

// Interface for sorting state
interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}

export default function DatasetPage() {
  const params = useParams();
  const id = params?.id as string;
  const [videos, setVideos] = useState<Video[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'view_count', direction: 'desc' });
  const router = useRouter();
  const datasetId = parseInt(id);

  // Function to handle sorting
  const requestSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'desc';
    // If already sorted by this key, toggle direction
    if (sortConfig.key === key && sortConfig.direction === 'desc') {
      direction = 'asc';
    }
    setSortConfig({ key, direction });
  };

  // Get sort icon based on current sorting
  const getSortIcon = (key: string) => {
    if (sortConfig.key !== key) {
      return '↕️'; // Default unsorted icon
    }
    return sortConfig.direction === 'asc' ? '↑' : '↓';
  };

  useEffect(() => {
    const fetchVideos = async () => {
      if (isNaN(datasetId)) {
        setError('Invalid dataset ID');
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        const data = await api.getVideos(
          datasetId, 
          sortConfig.key, 
          sortConfig.direction
        );
        setVideos(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching videos:', err);
        setError('Failed to load videos. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchVideos();
  }, [datasetId, sortConfig]);

  const viewVideo = (videoId: number) => {
    router.push(`/video/${videoId}`);
  };

  const goToAnalysis = () => {
    router.push(`/analysis/${datasetId}`);
  };

  return (
    <main className="container">
      <div className="flex-between mb-6">
        <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>Dataset Videos</h1>
        <div>
          <button 
            onClick={() => router.push('/')} 
            className="button button-secondary mr-2"
          >
            Back to Datasets
          </button>
          <button 
            onClick={goToAnalysis}
            className="button"
          >
            Analyze Dataset
          </button>
        </div>
      </div>

      {isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '16rem' }}>
          <p>Loading videos...</p>
        </div>
      ) : error ? (
        <div className="error-container" style={{ backgroundColor: '#fee2e2', borderLeft: '4px solid #ef4444', color: '#b91c1c', padding: '1rem', marginBottom: '1rem' }}>
          <p>{error}</p>
        </div>
      ) : (
        <>
          <div className="info-box" style={{ backgroundColor: '#f9fafb', padding: '1rem', marginBottom: '1.5rem', borderRadius: '0.5rem' }}>
            <p style={{ color: '#4b5563' }}>Total videos: {videos.length} | Current sort: {sortConfig.key} ({sortConfig.direction})</p>
          </div>

          {videos.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table className="table">
                <thead>
                  <tr>
                    <th onClick={() => requestSort('title')} style={{ cursor: 'pointer' }}>
                      Title {getSortIcon('title')}
                    </th>
                    <th onClick={() => requestSort('published_at')} style={{ textAlign: 'right', cursor: 'pointer' }}>
                      Published {getSortIcon('published_at')}
                    </th>
                    <th onClick={() => requestSort('view_count')} style={{ textAlign: 'right', cursor: 'pointer' }}>
                      Views {getSortIcon('view_count')}
                    </th>
                    <th onClick={() => requestSort('like_count')} style={{ textAlign: 'right', cursor: 'pointer' }}>
                      Likes {getSortIcon('like_count')}
                    </th>
                    <th onClick={() => requestSort('comment_count')} style={{ textAlign: 'right', cursor: 'pointer' }}>
                      Comments {getSortIcon('comment_count')}
                    </th>
                    <th onClick={() => requestSort('engagement_rate')} style={{ textAlign: 'right', cursor: 'pointer' }}>
                      Engagement {getSortIcon('engagement_rate')}
                    </th>
                    <th onClick={() => requestSort('duration')} style={{ textAlign: 'right', cursor: 'pointer' }}>
                      Duration {getSortIcon('duration')}
                    </th>
                    <th style={{ textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {videos.map((video) => (
                    <tr key={video.id}>
                      <td>
                        <div className="truncate" style={{ maxWidth: '20rem' }} title={video.title}>
                          {video.title}
                        </div>
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {new Date(video.published_at).toLocaleDateString()}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {video.stats.view_count.toLocaleString()}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {video.stats.like_count.toLocaleString()}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {video.stats.comment_count.toLocaleString()}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {(video.engagement.engagement_rate * 100).toFixed(2)}%
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {formatDuration(video.meta_info.duration)}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <button
                          onClick={() => viewVideo(video.id)}
                          className="button button-secondary"
                          style={{ fontSize: '0.875rem', padding: '0.25rem 0.5rem' }}
                        >
                          Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ backgroundColor: '#fefce8', padding: '1rem', borderRadius: '0.5rem' }}>
              <p>No videos found in this dataset.</p>
            </div>
          )}
        </>
      )}
    </main>
  );
}

// Helper function to format duration in seconds to minutes and seconds
function formatDuration(seconds: number): string {
  if (!seconds) return '0:00';
  
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  
  if (minutes < 60) {
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  } else {
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}:${remainingMinutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
} 