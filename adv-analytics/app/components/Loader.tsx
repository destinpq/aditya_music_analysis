'use client';

import React from 'react';

interface LoaderProps {
  size?: 'small' | 'medium' | 'large';
  message?: string;
  fullScreen?: boolean;
  className?: string;
}

export default function Loader({
  size = 'medium',
  message,
  fullScreen = false,
  className = '',
}: LoaderProps) {
  const sizeClass = size === 'small' ? 'loader-small' : size === 'large' ? 'loader-large' : '';
  
  const containerClass = fullScreen ? 'loader-fullscreen' : 'loader-container';

  return (
    <div className={`${containerClass} ${className}`}>
      <div className={`loader ${sizeClass}`}></div>
      {message && (
        <p className="loader-message">{message}</p>
      )}
    </div>
  );
} 