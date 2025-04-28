'use client';

import { useState, useRef, useEffect } from 'react';
import api from '../services/api';
import '../styles/upload-form.css';

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);
  const [showSizeWarning, setShowSizeWarning] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (file && file.size > 5 * 1024 * 1024) { // If file is larger than 5MB
      setShowSizeWarning(true);
    } else {
      setShowSizeWarning(false);
    }
  }, [file]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      // Check file type
      const isCSV = selectedFile.type === 'text/csv' || selectedFile.name.endsWith('.csv');
      if (!isCSV) {
        setErrorMessage('You can only upload CSV files!');
        return;
      }
      setFile(selectedFile);
      setErrorMessage(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      // Check file type
      const isCSV = droppedFile.type === 'text/csv' || droppedFile.name.endsWith('.csv');
      if (!isCSV) {
        setErrorMessage('You can only upload CSV files!');
        return;
      }
      setFile(droppedFile);
      setErrorMessage(null);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setErrorMessage('Please select a file first!');
      return;
    }

    setUploading(true);
    setErrorMessage(null);
    setProcessingStatus('Preparing to upload...');

    try {
      // Update progress function
      const updateProgress = (progress: number) => {
        setUploadProgress(progress);
        if (progress < 100) {
          setProcessingStatus(`Uploading file: ${progress}% complete`);
        } else {
          setProcessingStatus('File uploaded. Processing data...');
        }
      };

      // Upload the file
      const response = await api.uploadDataset(file, updateProgress);
      console.log('Upload response:', response);
      
      setProcessingStatus('Processing complete! Analyzing dataset...');
      setCurrentStep(1);
      // In a real app, we would redirect to results page
    } catch (error) {
      console.error('Upload error:', error);
      setErrorMessage(error instanceof Error ? error.message : 'Failed to upload file');
      setProcessingStatus(null);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-card">
      <div className="upload-steps">
        <div className={`step ${currentStep >= 0 ? 'active' : ''}`}>
          <div className="step-number">1</div>
          <div className="step-title">Upload CSV</div>
        </div>
        <div className="step-connector"></div>
        <div className={`step ${currentStep >= 1 ? 'active' : ''}`}>
          <div className="step-number">2</div>
          <div className="step-title">Process Data</div>
        </div>
        <div className="step-connector"></div>
        <div className={`step ${currentStep >= 2 ? 'active' : ''}`}>
          <div className="step-number">3</div>
          <div className="step-title">View Insights</div>
        </div>
      </div>

      {errorMessage && (
        <div className="error-alert">
          <div className="error-icon">⚠️</div>
          <div className="error-content">
            <div className="error-title">Upload Error</div>
            <div className="error-message">{errorMessage}</div>
          </div>
          <button 
            className="error-close" 
            onClick={() => setErrorMessage(null)}
            aria-label="Close error message"
          >
            ×
          </button>
        </div>
      )}

      {showSizeWarning && file && (
        <div className="warning-alert">
          <div className="warning-icon">⏱️</div>
          <div className="warning-content">
            <div className="warning-title">Large Dataset Detected</div>
            <div className="warning-message">
              Your file is {(file.size / (1024 * 1024)).toFixed(2)}MB. Processing large datasets may take several minutes.
              The system will process approximately 300-500 rows per minute.
            </div>
          </div>
          <button 
            className="warning-close" 
            onClick={() => setShowSizeWarning(false)}
            aria-label="Close warning message"
          >
            ×
          </button>
        </div>
      )}

      <div 
        className="upload-dropzone"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        {!file ? (
          <>
            <div className="upload-icon">📁</div>
            <p className="upload-text">Click or drag file to this area to upload</p>
            <p className="upload-hint">
              Support for a single CSV file. Your CSV should contain video data with columns 
              for views, likes, and other metrics.
            </p>
            <input 
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".csv"
              className="file-input"
            />
          </>
        ) : (
          <div className="selected-file">
            <div className="file-info">
              <div className="file-icon">📄</div>
              <div className="file-details">
                <div className="file-name">{file.name}</div>
                <div className="file-size">{(file.size / 1024).toFixed(2)} KB</div>
              </div>
            </div>
            <button 
              className="remove-file" 
              onClick={(e) => {
                e.stopPropagation();
                handleRemoveFile();
              }}
              aria-label="Remove file"
            >
              ×
            </button>
          </div>
        )}
      </div>

      {uploading && (
        <div className="upload-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <div className="progress-text">{processingStatus || `${uploadProgress}% Uploaded`}</div>
        </div>
      )}

      <button
        className={`upload-button ${file ? '' : 'disabled'}`}
        onClick={handleUpload}
        disabled={!file || uploading}
      >
        {uploading ? 'Processing...' : 'Upload & Analyze'}
      </button>
      
      {file && !uploading && (
        <div className="upload-info">
          <p className="info-text">
            <strong>Note:</strong> Processing time depends on file size. Larger files (20,000+ rows) may take several 
            minutes to complete. Please be patient during processing.
          </p>
        </div>
      )}
    </div>
  );
} 