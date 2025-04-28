'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import api, { Dataset } from '../services/api';
import '../styles/dataset-pages.css';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDatasets = async () => {
      setIsLoading(true);
      try {
        const response = await api.getDatasets();
        setDatasets(response);
      } catch (err) {
        setError(`Error fetching datasets: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDatasets();
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Datasets</h1>
        <p className="page-description">
          Manage your YouTube analytics datasets and extract insights
        </p>
      </div>

      <div className="page-actions" style={{ textAlign: 'right', marginBottom: '1.5rem' }}>
        <Link href="/dataset/upload" className="button button-primary">
          Upload New Dataset
        </Link>
      </div>

      {isLoading ? (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading datasets...</p>
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
        <div className="datasets-container">
          <div className="datasets-header">
            <h2 className="datasets-title">Your Datasets</h2>
            <span className="datasets-count">{datasets.length} datasets found</span>
          </div>

          {datasets.length === 0 ? (
            <div className="empty-datasets">
              <div className="empty-icon">📊</div>
              <h3 className="empty-title">No datasets yet</h3>
              <p className="empty-description">
                Upload your first dataset to start analyzing your YouTube content performance
              </p>
              <Link href="/dataset/upload" className="button button-primary">
                Upload Dataset
              </Link>
            </div>
          ) : (
            <ul className="datasets-list">
              {datasets.map((dataset) => (
                <li key={dataset.id} className="dataset-item">
                  <div className="dataset-icon">📊</div>
                  <div className="dataset-info">
                    <div className="dataset-name">{dataset.filename}</div>
                    <div className="dataset-meta">
                      <span>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M8 7V3M16 7V3M7 11H17M5 21H19C20.1046 21 21 20.1046 21 19V7C21 5.89543 20.1046 5 19 5H5C3.89543 5 3 5.89543 3 7V19C3 20.1046 3.89543 21 5 21Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        Uploaded {formatDate(dataset.uploadDate)}
                      </span>
                      {dataset.row_count && (
                        <span>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M3 6H21M3 12H21M3 18H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                          {dataset.row_count} videos
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="dataset-actions">
                    <Link href={`/analytics/${dataset.id}`} className="button button-small button-outline">
                      Analyze
                    </Link>
                    <Link href={`/dataset/${dataset.id}`} className="button button-small">
                      View Details
                    </Link>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
} 