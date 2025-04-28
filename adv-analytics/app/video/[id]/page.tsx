'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import api, { Video } from '../../services/api';

export default function VideoPage() {
  const params = useParams();
  const id = params?.id as string;
  const [video, setVideo] = useState<Video | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const videoId = parseInt(id);

  useEffect(() => {
    const fetchVideo = async () => {
      if (isNaN(videoId)) {
        setError('Invalid video ID');
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        const data = await api.getVideo(videoId);
        setVideo(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching video:', err);
        setError('Failed to load video data. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchVideo();
  }, [videoId]);

  // Format duration from seconds to HH:MM:SS
  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return [
      hours > 0 ? hours.toString().padStart(2, '0') : null,
      minutes.toString().padStart(2, '0'),
      secs.toString().padStart(2, '0')
    ].filter(Boolean).join(':');
  };

  return (
    <main className="container">
      <div className="mb-6">
        <button 
          onClick={() => router.back()} 
          className="button button-secondary"
        >
          ← Back
        </button>
      </div>

      {isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '16rem' }}>
          <p>Loading video data...</p>
        </div>
      ) : error ? (
        <div className="error-container" style={{ backgroundColor: '#fee2e2', borderLeft: '4px solid #ef4444', color: '#b91c1c', padding: '1rem', marginBottom: '1rem' }}>
          <p>{error}</p>
        </div>
      ) : video ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="card">
            <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>{video.title}</h1>
            <p className="mt-1" style={{ color: '#6b7280' }}>
              YouTube ID: <span style={{ fontFamily: 'monospace' }}>{video.video_id}</span>
            </p>
            <p style={{ color: '#6b7280' }}>
              Published: {new Date(video.published_at).toLocaleDateString()}
            </p>
          </div>

          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
            {/* Performance Metrics */}
            <div className="card">
              <h2 style={{ fontSize: '1.125rem', fontWeight: '500', marginBottom: '1rem' }}>Performance Metrics</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div>
                  <p style={{ color: '#6b7280' }}>Views</p>
                  <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{video.stats.view_count.toLocaleString()}</p>
                </div>
                <div>
                  <p style={{ color: '#6b7280' }}>Likes</p>
                  <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{video.stats.like_count.toLocaleString()}</p>
                </div>
                <div>
                  <p style={{ color: '#6b7280' }}>Comments</p>
                  <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{video.stats.comment_count.toLocaleString()}</p>
                </div>
                {video.stats.dislike_count > 0 && (
                  <div>
                    <p style={{ color: '#6b7280' }}>Dislikes</p>
                    <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{video.stats.dislike_count.toLocaleString()}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Engagement Metrics */}
            <div className="card">
              <h2 style={{ fontSize: '1.125rem', fontWeight: '500', marginBottom: '1rem' }}>Engagement Metrics</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div>
                  <p style={{ color: '#6b7280' }}>Engagement Rate</p>
                  <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{(video.engagement.engagement_rate * 100).toFixed(2)}%</p>
                </div>
                <div>
                  <p style={{ color: '#6b7280' }}>Like Ratio</p>
                  <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{(video.engagement.like_ratio * 100).toFixed(2)}%</p>
                </div>
                <div>
                  <p style={{ color: '#6b7280' }}>Comment Ratio</p>
                  <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{(video.engagement.comment_ratio * 100).toFixed(2)}%</p>
                </div>
              </div>
            </div>

            {/* Video Details */}
            <div className="card">
              <h2 style={{ fontSize: '1.125rem', fontWeight: '500', marginBottom: '1rem' }}>Video Details</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div>
                  <p style={{ color: '#6b7280' }}>Duration</p>
                  <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{formatDuration(video.meta_info.duration)}</p>
                </div>
                {video.meta_info.channel_id && (
                  <div>
                    <p style={{ color: '#6b7280' }}>Channel ID</p>
                    <p style={{ fontSize: '0.875rem', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis' }}>{video.meta_info.channel_id}</p>
                  </div>
                )}
                {video.meta_info.category_id && (
                  <div>
                    <p style={{ color: '#6b7280' }}>Category ID</p>
                    <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{video.meta_info.category_id}</p>
                  </div>
                )}
                <div>
                  <p style={{ color: '#6b7280' }}>Visibility</p>
                  <p style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{video.meta_info.is_unlisted ? 'Unlisted' : 'Public'}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Tags Section */}
          {video.tags && video.tags.length > 0 && (
            <div className="card">
              <h2 style={{ fontSize: '1.125rem', fontWeight: '500', marginBottom: '1rem' }}>Tags</h2>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {video.tags.map((tag, index) => (
                  <span 
                    key={index} 
                    style={{ 
                      padding: '0.25rem 0.5rem', 
                      backgroundColor: '#e6f1fe', 
                      color: '#1e6fda', 
                      borderRadius: '9999px', 
                      fontSize: '0.875rem' 
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* YouTube Embed Link */}
          <div className="card">
            <h2 style={{ fontSize: '1.125rem', fontWeight: '500', marginBottom: '1rem' }}>YouTube Link</h2>
            <a 
              href={`https://www.youtube.com/watch?v=${video.video_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="button"
            >
              View on YouTube
            </a>
          </div>
        </div>
      ) : (
        <div style={{ backgroundColor: '#fefce8', padding: '1rem', borderRadius: '0.5rem' }}>
          <p>No video data found.</p>
        </div>
      )}
    </main>
  );
} 