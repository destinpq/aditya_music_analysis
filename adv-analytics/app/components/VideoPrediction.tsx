import React, { useState, useEffect } from 'react';
import api, { PredictionInput } from '../services/api';

interface VideoPredictionProps {
  onPredictionComplete?: (result: any) => void;
}

const categories = [
  { id: '1', name: 'Film & Animation' },
  { id: '2', name: 'Autos & Vehicles' },
  { id: '10', name: 'Music' },
  { id: '15', name: 'Pets & Animals' },
  { id: '17', name: 'Sports' },
  { id: '19', name: 'Travel & Events' },
  { id: '20', name: 'Gaming' },
  { id: '22', name: 'People & Blogs' },
  { id: '23', name: 'Comedy' },
  { id: '24', name: 'Entertainment' },
  { id: '25', name: 'News & Politics' },
  { id: '26', name: 'Howto & Style' },
  { id: '27', name: 'Education' },
  { id: '28', name: 'Science & Technology' },
  { id: '29', name: 'Nonprofits & Activism' },
];

const VideoPrediction: React.FC<VideoPredictionProps> = ({ onPredictionComplete }) => {
  const [title, setTitle] = useState<string>('');
  const [tags, setTags] = useState<string>('');
  const [duration, setDuration] = useState<string>('');
  const [categoryId, setCategoryId] = useState<string>('10'); // Default to Music
  const [publishDate, setPublishDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const handleDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    
    // Format duration as MM:SS
    const minutes = parseInt(value.split(':')[0] || '0', 10);
    const seconds = parseInt(value.split(':')[1] || '0', 10);
    
    let formattedDuration = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    if (!value.includes(':')) {
      formattedDuration = `${value}:00`;
    }
    
    setDuration(formattedDuration);
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);
    
    try {
      // Format data for prediction
      const predictionData: PredictionInput = {
        title,
        duration, 
        tags,
        category_id: categoryId,
        published_at: publishDate
      };
      
      const result = await api.predictVideo(predictionData);
      setPredictionResult(result.predictions);
      
      if (onPredictionComplete) {
        onPredictionComplete(result);
      }
    } catch (err) {
      setError(`Prediction failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="card video-prediction-card">
      <h2 className="section-title">Predict Video Performance</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label">Video Title</label>
          <input 
            type="text"
            className="form-control"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter video title"
            required
          />
        </div>
        
        <div className="form-row">
          <div className="form-group half">
            <label className="form-label">Duration (MM:SS)</label>
            <input 
              type="text"
              className="form-control"
              value={duration}
              onChange={handleDurationChange}
              placeholder="e.g. 5:30"
              required
            />
          </div>
          
          <div className="form-group half">
            <label className="form-label">Category</label>
            <select 
              className="form-control"
              value={categoryId}
              onChange={(e) => setCategoryId(e.target.value)}
              required
            >
              {categories.map(category => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        
        <div className="form-group">
          <label className="form-label">Tags (comma separated)</label>
          <input 
            type="text"
            className="form-control"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="music, new release, official"
          />
        </div>
        
        <div className="form-group">
          <label className="form-label">Publish Date</label>
          <input 
            type="date"
            className="form-control"
            value={publishDate}
            onChange={(e) => setPublishDate(e.target.value)}
          />
        </div>
        
        {error && <div className="error">{error}</div>}
        
        <button 
          type="submit" 
          className="button button-primary"
          disabled={isLoading}
        >
          {isLoading ? 'Predicting...' : 'Predict Performance'}
        </button>
      </form>
      
      {predictionResult && (
        <div className="prediction-results">
          <h3>Prediction Results</h3>
          
          <div className="result-cards">
            <div className="result-card">
              <div className="result-icon">👁️</div>
              <div className="result-value">{Number(predictionResult.predicted_views).toLocaleString()}</div>
              <div className="result-label">Predicted Views</div>
            </div>
            
            <div className="result-card">
              <div className="result-icon">👍</div>
              <div className="result-value">{Number(predictionResult.predicted_likes).toLocaleString()}</div>
              <div className="result-label">Predicted Likes</div>
            </div>
            
            <div className="result-card">
              <div className="result-icon">💬</div>
              <div className="result-value">{Number(predictionResult.predicted_comments).toLocaleString()}</div>
              <div className="result-label">Predicted Comments</div>
            </div>
            
            <div className="result-card">
              <div className="result-icon">💰</div>
              <div className="result-value">${Number(predictionResult.earnings?.estimated_revenue || 0).toFixed(2)}</div>
              <div className="result-label">Estimated Revenue</div>
            </div>
          </div>
          
          <div className="prediction-analysis">
            <h4>Performance Analysis</h4>
            <p>
              Based on our prediction model, your video is expected to receive approximately{' '}
              <strong>{Number(predictionResult.predicted_views).toLocaleString()} views</strong> and generate around{' '}
              <strong>${Number(predictionResult.earnings?.estimated_revenue || 0).toFixed(2)} in revenue</strong>.
              The engagement rate is predicted to be{' '}
              <strong>{(Number(predictionResult.engagement_rate || 0) * 100).toFixed(2)}%</strong>, which is{' '}
              {Number(predictionResult.engagement_rate || 0) > 0.05 ? 'above' : 'below'} average for this type of content.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoPrediction; 