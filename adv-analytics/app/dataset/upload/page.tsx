'use client';

import React from 'react';
import UploadForm from '../../components/UploadForm';

export default function UploadDatasetPage() {
  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Upload Dataset</h1>
        <p className="page-description">
          Upload your YouTube analytics data to gain insights from machine learning models
        </p>
      </div>
      
      <div className="upload-container">
        <div className="upload-info">
          <h2>Data Requirements</h2>
          <p>For best results, your CSV file should contain the following columns:</p>
          <ul className="requirements-list">
            <li>
              <strong>Video ID</strong> - YouTube video identifier
            </li>
            <li>
              <strong>Title</strong> - Video title
            </li>
            <li>
              <strong>Published At</strong> - Publication date and time
            </li>
            <li>
              <strong>View Count</strong> - Number of views
            </li>
            <li>
              <strong>Like Count</strong> - Number of likes
            </li>
            <li>
              <strong>Comment Count</strong> - Number of comments
            </li>
            <li>
              <strong>Duration</strong> - Video duration (seconds or formatted)
            </li>
          </ul>
          <p className="info-note">
            Additional columns like tags, category, and channel information will improve prediction quality.
          </p>
        </div>
        
        <div className="upload-form-container">
          <UploadForm />
        </div>
      </div>
    </div>
  );
} 