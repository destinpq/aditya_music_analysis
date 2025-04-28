'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Image from 'next/image';
import api from '../services/api';
import './navbar.css';

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [activeDatasetId, setActiveDatasetId] = useState<number | null>(null);
  const pathname = usePathname();

  useEffect(() => {
    // Extract dataset ID from URL if present
    // This handles URLs like /analysis/2, /dataset/2, etc.
    const match = pathname.match(/\/(analysis|dataset|analytics|revenue)\/(\d+)/);
    if (match && match[2]) {
      setActiveDatasetId(parseInt(match[2]));
    } else {
      // Try to load the first dataset if none is active
      const fetchFirstDataset = async () => {
        try {
          const datasets = await api.getDatasets();
          if (datasets && datasets.length > 0) {
            setActiveDatasetId(datasets[0].id);
          }
        } catch (error) {
          console.error("Failed to fetch datasets", error);
        }
      };
      fetchFirstDataset();
    }
  }, [pathname]);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  // Base navigation links always include Dashboard
  const baseLinks = [
    { name: 'Dashboard', href: '/' },
  ];
  
  // Dataset-specific links that require a dataset ID
  const datasetLinks = activeDatasetId ? [
    { name: 'Videos', href: `/dataset/${activeDatasetId}` },
    { name: 'Analytics', href: `/analytics/${activeDatasetId}` },
    { name: 'Analysis', href: `/analysis/${activeDatasetId}` },
  ] : [];

  // Combine the links
  const navLinks = [...baseLinks, ...datasetLinks];

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <Link href="/" className="navbar-logo">
            <Image 
              src="/destinpq.png" 
              alt="DESTIN PQ Logo" 
              width={140} 
              height={48} 
              className="destin-logo"
            />
          </Link>
        </div>
        
        <button 
          className="navbar-toggle" 
          onClick={toggleMenu}
          aria-label="Toggle navigation menu"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 12H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M3 6H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M3 18H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        
        <div className={`navbar-content ${isOpen ? 'open' : ''}`}>
          <ul className="navbar-nav">
            <li className={`nav-item ${pathname === '/' ? 'active' : ''}`}>
              <Link href="/" className="nav-link">Dashboard</Link>
            </li>
            <li className={`nav-item ${pathname?.includes('/dashboard') ? 'active' : ''}`}>
              <Link href="/dashboard" className="nav-link">Analytics</Link>
            </li>
            <li className={`nav-item ${pathname?.includes('/dataset') ? 'active' : ''}`}>
              <Link href="/dataset" className="nav-link">Datasets</Link>
            </li>
            <li className={`nav-item ${pathname?.includes('/ml-lab') ? 'active' : ''}`}>
              <Link href="/ml-lab" className="nav-link">ML Lab</Link>
            </li>
          </ul>
          
          <div className="navbar-search">
            <form className="search-form">
              <input 
                type="text" 
                className="search-input" 
                placeholder="Search videos..." 
              />
              <button type="submit" className="search-button">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M11 19C15.4183 19 19 15.4183 19 11C19 6.58172 15.4183 3 11 3C6.58172 3 3 6.58172 3 11C3 15.4183 6.58172 19 11 19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M21 21L16.65 16.65" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </form>
          </div>
        </div>
      </div>
    </nav>
  );
} 