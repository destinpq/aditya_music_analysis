'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { Card, Typography } from 'antd';

const { Title } = Typography;
// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PerformanceMetricsChartProps {
  data: any[];
  // Add other props as needed
}

export default function PerformanceMetricsChart({ data }: PerformanceMetricsChartProps) {
  const layout = {
    title: {
      text: "Performance Metrics",
      font: {
        size: 20,
        color: "white"
      }
    },
    xaxis: {
      title: {
        text: "Metric",
        font: {
          size: 16,
          color: "white"
        }
      },
      tickfont: {
        color: "white"
      }
    },
    yaxis: {
      title: {
        text: "Value",
        font: {
          size: 16,
          color: "white"
        }
      },
      tickfont: {
        color: "white"
      }
    },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 50, b: 50, l: 80, r: 30 },
    showlegend: false,
    barmode: 'group'
  };

  return (
    <Card style={{ backgroundColor: 'transparent' }}>
      <Title level={4} style={{ color: 'white' }}>Performance Metrics</Title>
      <div style={{ height: 400 }}>
        {data && data.length > 0 && (
          <Plot
            data={data}
            layout={layout}
            style={{ width: '100%', height: '100%' }}
            config={{ responsive: true, displayModeBar: false }}
          />
        )}
      </div>
    </Card>
  );
} 