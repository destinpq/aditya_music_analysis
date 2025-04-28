'use client';

import React from 'react';
import { CountryRevenue } from '../services/api';

interface CountryRevenueMapProps {
  data: Record<string, CountryRevenue>;
  className?: string;
}

export default function CountryRevenueMap({ data, className = '' }: CountryRevenueMapProps) {
  // Sort countries by revenue
  const sortedCountries = Object.entries(data)
    .sort((a, b) => b[1].revenue - a[1].revenue);
  
  // Calculate the total revenue for percentage calculations
  const totalRevenue = sortedCountries.reduce((sum, [, data]) => sum + data.revenue, 0);

  // Country color mapping based on revenue tiers
  const getCountryColor = (revenue: number) => {
    const percentage = (revenue / totalRevenue) * 100;
    
    if (percentage > 25) return 'country-tier-1';
    if (percentage > 15) return 'country-tier-2';
    if (percentage > 10) return 'country-tier-3';
    if (percentage > 5) return 'country-tier-4';
    if (percentage > 2) return 'country-tier-5';
    return 'country-tier-6';
  };

  // Calculate opacity based on revenue percentage
  const getOpacityStyle = (revenue: number) => {
    const percentage = (revenue / totalRevenue);
    return { opacity: 0.7 + (percentage * 0.3) }; // 0.7 to 1.0 opacity range
  };

  return (
    <div className={`country-revenue-map ${className}`}>
      <div className="grid-cols-2">
        {/* Map visualization */}
        <div className="card">
          <h3 className="section-title">Revenue Distribution</h3>
          <div className="grid-cols-3">
            {sortedCountries.map(([country, data]) => {
              const percentage = ((data.revenue / totalRevenue) * 100).toFixed(1);
              
              return (
                <div 
                  key={country}
                  className={`country-block ${getCountryColor(data.revenue)}`}
                  style={{ ...getOpacityStyle(data.revenue) }}
                >
                  <div className="country-name">{country}</div>
                  <div className="country-percentage">{percentage}%</div>
                </div>
              );
            })}
          </div>
          
          <div className="chart-footnote">
            * Map shows relative revenue contribution from each country
          </div>
        </div>
        
        {/* Revenue details table */}
        <div className="card">
          <h3 className="section-title">Revenue by Country</h3>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Country</th>
                  <th className="text-right">Views</th>
                  <th className="text-right">Revenue</th>
                  <th className="text-right">% of Total</th>
                </tr>
              </thead>
              <tbody>
                {sortedCountries.map(([country, data]) => {
                  const percentage = ((data.revenue / totalRevenue) * 100).toFixed(1);
                  
                  return (
                    <tr key={country}>
                      <td className="font-medium">{country}</td>
                      <td className="text-right">{data.views.toLocaleString()}</td>
                      <td className="text-right">${data.revenue.toLocaleString()}</td>
                      <td className="text-right">{percentage}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      <div className="card mt-6">
        <h3 className="section-title">Revenue Rate by Country</h3>
        <div className="grid-cols-7">
          {sortedCountries.map(([country, data]) => (
            <div key={country} className="country-rate-card">
              <div className="country-rate-name">{country}</div>
              <div className="country-rate-value">${data.rate_per_1000.toFixed(2)}</div>
              <div className="country-rate-label">per 1000 views</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 