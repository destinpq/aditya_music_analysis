'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { Card, Typography } from 'antd';

const { Title } = Typography;
// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface EngagementMetricsChartProps {
  data: any[];
  // Add other props as needed
}

export default function EngagementMetricsChart({ data }: EngagementMetricsChartProps) {
  const layout = {
    title: {
      text: "Engagement Metrics Over Time",
      font: {
        size: 20,
        color: "black"
      }
    },
    xaxis: {
      title: {
        text: "Date",
        font: {
          size: 16,
          color: "black"
        }
      },
      tickfont: {
        color: "black"
      }
    },
    yaxis: {
      title: {
        text: "Average Engagement Rate (%)",
        font: {
          size: 16,
          color: "black"
        }
      },
      tickfont: {
        color: "black"
      }
    },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    margin: { t: 50, b: 50, l: 80, r: 30 },
    legend: {
      font: {
        color: "black"
      }
    },
    autosize: true
  };

  return (
    <Card style={{ backgroundColor: 'white' }}>
      <Title level={4} style={{ color: 'black' }}>Engagement Metrics Over Time</Title>
      <div style={{ height: 400 }}>
        {data && data.length > 0 && (
          <Plot
            data={data}
            layout={layout}
            style={{ width: '100%', height: '100%' }}
            config={{ 
              responsive: true,
              displayModeBar: false
            }}
          />
        )}
      </div>
    </Card>
  );
} 