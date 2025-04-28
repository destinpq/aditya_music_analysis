import React, { useState } from 'react';
import api, { Dataset, MLTrainParams } from '../services/api';

interface ModelTrainerProps {
  datasets: Dataset[];
  onTrainingComplete?: (result: any) => void;
}

const modelTypes = [
  { id: 'xgboost', name: 'XGBoost' },
  { id: 'lightgbm', name: 'LightGBM' },
  { id: 'catboost', name: 'CatBoost' },
  { id: 'random_forest', name: 'Random Forest' },
  { id: 'linear', name: 'Linear Model' },
];

const targetOptions = [
  { id: 'view_count', name: 'View Count' },
  { id: 'like_count', name: 'Like Count' },
  { id: 'comment_count', name: 'Comment Count' },
  { id: 'engagement_rate', name: 'Engagement Rate' },
];

const ModelTrainer: React.FC<ModelTrainerProps> = ({ datasets, onTrainingComplete }) => {
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string>('view_count');
  const [selectedModels, setSelectedModels] = useState<string[]>(['xgboost', 'lightgbm']);
  const [testSize, setTestSize] = useState<number>(0.2);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const handleModelToggle = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter(id => id !== modelId));
    } else {
      setSelectedModels([...selectedModels, modelId]);
    }
  };
  
  const handleTrainModels = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }
    
    if (selectedModels.length === 0) {
      setError('Please select at least one model type');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const params: MLTrainParams = {
        dataset_id: selectedDataset,
        model_types: selectedModels,
        target_column: selectedTarget,
        test_size: testSize
      };
      
      const result = await api.mlTrainModels(params);
      
      setSuccess('Models trained successfully!');
      if (onTrainingComplete) {
        onTrainingComplete(result);
      }
    } catch (err) {
      setError(`Training failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleRunTuning = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const params = {
        dataset_id: selectedDataset,
        target_column: selectedTarget,
        test_size: testSize,
        n_splits: 3
      };
      
      const result = await api.mlTuneModels(params);
      
      setSuccess('Hyperparameter tuning completed successfully!');
      if (onTrainingComplete) {
        onTrainingComplete(result);
      }
    } catch (err) {
      setError(`Tuning failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="card model-trainer">
      <h2 className="section-title">Train ML Models</h2>
      
      <div className="form-group">
        <label className="form-label">Select Dataset</label>
        <select 
          className="form-control"
          value={selectedDataset || ''} 
          onChange={(e) => setSelectedDataset(parseInt(e.target.value) || null)}
        >
          <option value="">-- Select a dataset --</option>
          {datasets.map(dataset => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.filename}
            </option>
          ))}
        </select>
      </div>
      
      <div className="form-group">
        <label className="form-label">Target Metric</label>
        <select 
          className="form-control"
          value={selectedTarget} 
          onChange={(e) => setSelectedTarget(e.target.value)}
        >
          {targetOptions.map(option => (
            <option key={option.id} value={option.id}>
              {option.name}
            </option>
          ))}
        </select>
      </div>
      
      <div className="form-group">
        <label className="form-label">Test Set Size</label>
        <input 
          type="range" 
          min="0.1" 
          max="0.5" 
          step="0.05"
          value={testSize}
          onChange={(e) => setTestSize(parseFloat(e.target.value))}
          className="form-control range-input"
        />
        <div className="range-value">{Math.round(testSize * 100)}%</div>
      </div>
      
      <div className="form-group">
        <label className="form-label">Select Model Types</label>
        <div className="model-options">
          {modelTypes.map(model => (
            <div key={model.id} className="model-option">
              <label className="checkbox-label">
                <input 
                  type="checkbox"
                  checked={selectedModels.includes(model.id)}
                  onChange={() => handleModelToggle(model.id)}
                />
                <span className="checkbox-custom"></span>
                {model.name}
              </label>
            </div>
          ))}
        </div>
      </div>
      
      {error && <div className="error">{error}</div>}
      {success && <div className="success">{success}</div>}
      
      <div className="action-buttons">
        <button 
          className="button button-primary"
          onClick={handleTrainModels}
          disabled={isLoading}
        >
          {isLoading ? 'Training...' : 'Train Models'}
        </button>
        
        <button 
          className="button button-secondary ml-2"
          onClick={handleRunTuning}
          disabled={isLoading}
        >
          {isLoading ? 'Tuning...' : 'Run Hyperparameter Tuning'}
        </button>
      </div>
    </div>
  );
};

export default ModelTrainer; 