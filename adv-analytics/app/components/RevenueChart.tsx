'use client';

import React from 'react';
import { DailyRevenue } from '../services/api';
import { Card, Typography } from 'antd';

const { Title } = Typography;

interface RevenueChartProps {
  data: DailyRevenue[];
  className?: string;
}

export default function RevenueChart({ data, className = '' }: RevenueChartProps) {
  if (!data || data.length === 0) {
    return (
      <Card style={{ backgroundColor: 'transparent' }}>
        <div className="empty-chart" style={{ color: 'white' }}>
          <p>No revenue data available</p>
        </div>
      </Card>
    );
  }

  // Find the maximum revenue to properly scale the chart
  const maxRevenue = Math.max(...data.map(item => item.revenue));
  
  return (
    <Card style={{ backgroundColor: 'transparent' }}>
      <Title level={4} style={{ color: 'white' }}>Daily Revenue</Title>
      <div className={`revenue-chart ${className}`}>
        <div className="chart-container">
          {data.map((item, index) => {
            // Calculate the height percentage based on the revenue value
            const heightPercentage = (item.revenue / maxRevenue) * 100;
            
            return (
              <div 
                key={index} 
                className="chart-column"
                style={{ flex: `1 0 ${100 / Math.min(data.length, 30)}%` }}
              >
                <div 
                  className="revenue-bar"
                  style={{ height: `${heightPercentage}%`, backgroundColor: '#1890ff' }}
                >
                  {/* Tooltip */}
                  <div className="revenue-tooltip" style={{ color: 'white', backgroundColor: '#333', padding: '4px 8px', borderRadius: '4px' }}>
                    ₹{item.revenue.toLocaleString()} on {item.date}
                  </div>
                </div>
                
                {/* Only show every nth date label to avoid overcrowding */}
                {index % Math.ceil(data.length / 10) === 0 && (
                  <div className="chart-label" style={{ color: 'white' }}>
                    {item.date.substring(5)} {/* Show only MM-DD */}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        
        <div className="chart-summary" style={{ color: 'white', marginTop: '16px' }}>
          <div>Date Range: {data[0].date} - {data[data.length - 1].date}</div>
          <div>Total Revenue: ₹{data.reduce((sum, item) => sum + item.revenue, 0).toLocaleString()}</div>
        </div>
      </div>
    </Card>
  );
} 