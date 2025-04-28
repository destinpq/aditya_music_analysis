import React, { useState, useEffect } from 'react';
import api, { Dataset, DatasetProcessRequest } from '../services/api';

interface DatasetProcessorProps {
  datasets: Dataset[];
  onProcessingComplete?: (result: any) => void;
}

const DatasetProcessor: React.FC<DatasetProcessorProps> = ({ datasets, onProcessingComplete }) => {
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null);
  const [testSize, setTestSize] = useState<number>(0.2);
  const [randomState, setRandomState] = useState<number>(42);
  const [datasetName, setDatasetName] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [processingResult, setProcessingResult] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadMode, setUploadMode] = useState<'existing' | 'new'>('existing');
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [estimatedTime, setEstimatedTime] = useState<string | null>(null);
  
  // Calculate estimated processing time when file or dataset changes
  useEffect(() => {
    if (uploadMode === 'new' && selectedFile) {
      // Estimate based on file size (CSV is text, so rough estimate)
      const estimatedLinesPerMB = 5000; // Rough estimate of CSV lines per MB
      const fileSizeMB = selectedFile.size / (1024 * 1024);
      const estimatedLines = fileSizeMB * estimatedLinesPerMB;
      const processingRatePerMinute = 400; // Processing rate from the log (300-500 rows/min)
      const estimatedMinutes = Math.ceil(estimatedLines / processingRatePerMinute);
      
      if (estimatedMinutes < 1) {
        setEstimatedTime("less than a minute");
      } else if (estimatedMinutes === 1) {
        setEstimatedTime("about 1 minute");
      } else if (estimatedMinutes > 60) {
        const hours = Math.floor(estimatedMinutes / 60);
        const minutes = estimatedMinutes % 60;
        setEstimatedTime(`approximately ${hours} hour${hours > 1 ? 's' : ''} and ${minutes} minute${minutes !== 1 ? 's' : ''}`);
      } else {
        setEstimatedTime(`approximately ${estimatedMinutes} minutes`);
      }
    } else if (uploadMode === 'existing' && selectedDataset) {
      const dataset = datasets.find(d => d.id === selectedDataset);
      if (dataset?.row_count) {
        const processingRatePerMinute = 400; // Processing rate from the log
        const estimatedMinutes = Math.ceil(dataset.row_count / processingRatePerMinute);
        
        if (estimatedMinutes < 1) {
          setEstimatedTime("less than a minute");
        } else if (estimatedMinutes === 1) {
          setEstimatedTime("about 1 minute");
        } else if (estimatedMinutes > 60) {
          const hours = Math.floor(estimatedMinutes / 60);
          const minutes = estimatedMinutes % 60;
          setEstimatedTime(`approximately ${hours} hour${hours > 1 ? 's' : ''} and ${minutes} minute${minutes !== 1 ? 's' : ''}`);
        } else {
          setEstimatedTime(`approximately ${estimatedMinutes} minutes`);
        }
      } else {
        setEstimatedTime(null);
      }
    } else {
      setEstimatedTime(null);
    }
  }, [uploadMode, selectedFile, selectedDataset, datasets]);
  
  const handleSelectDataset = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setSelectedDataset(value ? parseInt(value, 10) : null);
    
    // Generate a default dataset name based on selected dataset
    if (value) {
      const dataset = datasets.find(d => d.id === parseInt(value, 10));
      if (dataset) {
        const baseName = dataset.filename.replace('.csv', '');
        setDatasetName(`${baseName}_processed`);
      }
    }
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      
      // Generate a dataset name based on file name
      const baseName = file.name.replace('.csv', '');
      setDatasetName(`${baseName}_processed`);
    }
  };
  
  const handleProcess = async () => {
    if (uploadMode === 'existing' && !selectedDataset) {
      setError('Please select a dataset');
      return;
    }
    
    if (uploadMode === 'new' && !selectedFile) {
      setError('Please select a file to upload');
      return;
    }
    
    if (!datasetName) {
      setError('Please enter a name for the processed dataset');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setSuccess(null);
    setProcessingResult(null);
    setProcessingProgress(0);
    setProcessingStatus('Initializing processing...');
    
    try {
      let result;
      
      // Simulate progress updates since the API doesn't provide real-time updates
      const progressInterval = setInterval(() => {
        setProcessingProgress(prev => {
          if (prev < 95) {
            // Start slow, then accelerate, then slow down near the end
            const increment = prev < 30 ? 1 : prev < 70 ? 3 : 1;
            const newProgress = Math.min(95, prev + increment);
            
            // Update status message based on progress
            if (newProgress < 10) {
              setProcessingStatus('Initializing data processing...');
            } else if (newProgress < 30) {
              setProcessingStatus('Reading dataset...');
            } else if (newProgress < 50) {
              setProcessingStatus('Processing rows...');
            } else if (newProgress < 70) {
              setProcessingStatus('Creating train/test split...');
            } else if (newProgress < 90) {
              setProcessingStatus('Finalizing dataset preparation...');
            }
            
            return newProgress;
          }
          return prev;
        });
      }, 1000);
      
      if (uploadMode === 'existing') {
        // Process existing dataset
        const request: DatasetProcessRequest = {
          dataset_id: selectedDataset!,
          test_size: testSize,
          random_state: randomState,
          dataset_name: datasetName
        };
        
        result = await api.processDataset(request);
      } else {
        // Process new file
        result = await api.processCSVFile(
          selectedFile!,
          testSize,
          randomState,
          datasetName
        );
      }
      
      clearInterval(progressInterval);
      setProcessingProgress(100);
      setProcessingStatus('Processing complete!');
      setProcessingResult(result);
      setSuccess(`Dataset processed successfully as "${datasetName}"`);
      
      if (onProcessingComplete) {
        onProcessingComplete(result);
      }
    } catch (err) {
      setError(`Processing failed: ${err instanceof Error ? err.message : String(err)}`);
      setProcessingStatus(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="card dataset-processor">
      <h2 className="section-title">Process Dataset for ML</h2>
      
      <div className="form-group">
        <label className="form-label">Processing Mode</label>
        <div className="button-group">
          <button
            className={`button ${uploadMode === 'existing' ? 'button-primary' : 'button-outline'}`}
            onClick={() => setUploadMode('existing')}
            type="button"
          >
            Use Existing Dataset
          </button>
          <button
            className={`button ${uploadMode === 'new' ? 'button-primary' : 'button-outline'}`}
            onClick={() => setUploadMode('new')}
            type="button"
          >
            Upload New File
          </button>
        </div>
      </div>
      
      {uploadMode === 'existing' ? (
        <div className="form-group">
          <label className="form-label">Select Dataset</label>
          <select 
            className="form-control"
            value={selectedDataset || ''}
            onChange={handleSelectDataset}
          >
            <option value="">-- Select a dataset --</option>
            {datasets.map(dataset => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.filename} {dataset.row_count ? `(${dataset.row_count} rows)` : ''}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="form-group">
          <label className="form-label">Upload CSV File</label>
          <div className="file-upload">
            <input 
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="file-input"
              id="csv-upload"
            />
            <label htmlFor="csv-upload" className="file-label">
              {selectedFile ? selectedFile.name : 'Choose a CSV file'}
            </label>
          </div>
        </div>
      )}
      
      {estimatedTime && (
        <div className="info-alert">
          <div className="info-icon">⏱️</div>
          <div className="info-content">
            <div className="info-title">Processing Time Estimate</div>
            <div className="info-message">
              Based on our processing rate (300-500 rows/minute), this dataset may take {estimatedTime} to process.
            </div>
          </div>
        </div>
      )}
      
      <div className="form-group">
        <label className="form-label">Processed Dataset Name</label>
        <input 
          type="text"
          className="form-control"
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
          placeholder="Enter a name for processed dataset"
        />
      </div>
      
      <div className="form-row">
        <div className="form-group half">
          <label className="form-label">Test Size</label>
          <div className="range-wrapper">
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
        </div>
        
        <div className="form-group half">
          <label className="form-label">Random State</label>
          <input 
            type="number"
            className="form-control"
            value={randomState}
            onChange={(e) => setRandomState(parseInt(e.target.value, 10))}
            min="0"
            max="100"
          />
        </div>
      </div>
      
      {error && (
        <div className="error-message">{error}</div>
      )}
      
      {success && (
        <div className="success-message">{success}</div>
      )}
      
      {isLoading && (
        <div className="processing-status">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${processingProgress}%` }}
            ></div>
          </div>
          <div className="progress-text">
            {processingStatus} {processingProgress}%
          </div>
        </div>
      )}
      
      {processingResult && (
        <div className="result-summary">
          <h3 className="result-title">Processing Summary</h3>
          <div className="result-grid">
            <div className="result-item">
              <span className="result-label">Processed Rows:</span>
              <span className="result-value">{processingResult.total_rows || 'N/A'}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Training Set:</span>
              <span className="result-value">{processingResult.train_rows || 'N/A'} rows</span>
            </div>
            <div className="result-item">
              <span className="result-label">Test Set:</span>
              <span className="result-value">{processingResult.test_rows || 'N/A'} rows</span>
            </div>
            <div className="result-item">
              <span className="result-label">Features:</span>
              <span className="result-value">{processingResult.features?.length || 'N/A'}</span>
            </div>
          </div>
        </div>
      )}
      
      <div className="form-actions">
        <button 
          className="button button-primary" 
          onClick={handleProcess}
          disabled={isLoading || (!selectedDataset && uploadMode === 'existing') || (!selectedFile && uploadMode === 'new')}
        >
          {isLoading ? 'Processing...' : 'Process Dataset'}
        </button>
      </div>
      
      {(selectedDataset || selectedFile) && !isLoading && (
        <div className="processing-note">
          <p>
            <strong>Note:</strong> Dataset processing can take several minutes for large datasets. 
            The system processes approximately 300-500 rows per minute. Please be patient.
          </p>
        </div>
      )}
    </div>
  );
};

export default DatasetProcessor; 