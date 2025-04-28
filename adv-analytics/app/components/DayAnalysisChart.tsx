'use client';

import { useEffect, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Spin, Select, DatePicker, Button, Card, Typography, Row, Col, Statistic } from 'antd';
import { RangePickerProps } from 'antd/es/date-picker';
import { BarChartOutlined } from '@ant-design/icons';
import { DayAnalysis } from '../models/types';
import { getDayAnalysis } from '../controllers/analysisController';

const { Title } = Typography;
const { RangePicker } = DatePicker;

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface DayAnalysisChartProps {
  datasetId: number;
}

export default function DayAnalysisChart({ datasetId }: DayAnalysisChartProps) {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<DayAnalysis[]>([]);
  const [dateRange, setDateRange] = useState<[string, string]>(['2001-01-01', new Date().toISOString().split('T')[0]]);
  const [metric, setMetric] = useState<'views' | 'likes' | 'comments' | 'engagement'>('views');
  
  const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getDayAnalysis(datasetId, dateRange[0], dateRange[1]);
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching day analysis data:', error);
    } finally {
      setLoading(false);
    }
  }, [datasetId, dateRange]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDateChange = (value: RangePickerProps['value']) => {
    if (Array.isArray(value) && value[0] && value[1]) {
      setDateRange([
        value[0].format('YYYY-MM-DD'),
        value[1].format('YYYY-MM-DD')
      ]);
    }
  };

  const handleApply = () => {
    fetchData();
  };

  // Find best performing day
  const getBestDay = () => {
    if (metrics.length === 0) return { day: 0, avg: 0, improvement: 0 };
    
    const bestDayIdx = metrics.reduce((maxIdx, m, idx, arr) => 
      arr[idx].avgViews > arr[maxIdx].avgViews ? idx : maxIdx, 0);
    
    const bestDay = dayNames[bestDayIdx];
    const bestDayAvg = metrics[bestDayIdx].avgViews;
    
    // Calculate overall average views across all days
    const totalVideos = metrics.reduce((sum, day) => sum + day.count, 0);
    const totalViews = metrics.reduce((sum, day) => sum + (day.avgViews * day.count), 0);
    const overallAvg = totalVideos > 0 ? totalViews / totalVideos : 0;
    
    const improvement = overallAvg > 0 ? ((bestDayAvg / overallAvg) - 1) * 100 : 0;
    
    return { day: bestDay, avg: bestDayAvg, improvement };
  };

  const { day, avg, improvement } = getBestDay();

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value);
  };

  // Define plotly props to avoid type errors
  const plotlyProps = {
    data: [{
      x: dayNames,
      y: metrics.map(d => d.avgViews),
      type: 'bar',
      marker: {
        color: metrics.map((d, i) => 
          i === metrics.reduce((maxIdx, m, idx, arr) => 
            arr[idx].avgViews > arr[maxIdx].avgViews ? idx : maxIdx, 0) 
            ? '#1890ff' : '#36cfc9'
        )
      },
      text: metrics.map(d => `Videos: ${d.count}<br>Avg Views: ${formatNumber(d.avgViews)}`),
      hoverinfo: 'text+x'
    }],
    layout: {
      title: {
        text: "Average Views by Day of Week",
        font: {
          size: 20,
          color: "white"
        }
      },
      xaxis: {
        title: {
          text: "Day of Week",
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
          text: "Average Views",
          font: {
            size: 16,
            color: "white"
          }
        },
        tickfont: {
          color: "white"
        }
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { t: 50, b: 50, l: 80, r: 30 },
      showlegend: false,
      autosize: true,
    },
    style: { width: '100%', height: 400 },
    config: { 
      responsive: true,
      displayModeBar: false
    }
  };

  return (
    <Card style={{ backgroundColor: 'white' }}>
      <Title level={4} style={{ color: 'black' }}>Day of Week Analysis</Title>
      <p style={{ color: 'black' }}>Find the most favorable day to post videos based on historical performance.</p>
      
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <RangePicker 
            onChange={handleDateChange}
            defaultValue={[null, null]}
          />
        </Col>
        <Col span={8}>
          <Select
            value={metric}
            onChange={setMetric}
            style={{ width: '100%' }}
            options={[
              { value: 'views', label: 'Views' },
              { value: 'likes', label: 'Likes' },
              { value: 'comments', label: 'Comments' },
              { value: 'engagement', label: 'Engagement Rate' },
            ]}
          />
        </Col>
        <Col span={4}>
          <Button type="primary" onClick={handleApply}>Apply</Button>
        </Col>
      </Row>
      
      <Spin spinning={loading}>
        <div style={{ height: 400, backgroundColor: 'white' }}>
          {!loading && metrics.length > 0 && (
            <Plot {...plotlyProps} />
          )}
        </div>
      </Spin>
      
      <Row gutter={16} style={{ marginTop: 24 }}>
        <Col span={8}>
          <Card style={{ backgroundColor: 'white' }}>
            <Statistic
              title={<span style={{ color: 'black' }}>Best Day to Post</span>}
              value={day}
              prefix={<BarChartOutlined style={{ color: 'black' }} />}
              valueStyle={{ color: 'black' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card style={{ backgroundColor: 'white' }}>
            <Statistic
              title={<span style={{ color: 'black' }}>Average Views on Best Day</span>}
              value={avg}
              formatter={value => formatNumber(Number(value))}
              valueStyle={{ color: 'black' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card style={{ backgroundColor: 'white' }}>
            <Statistic
              title={<span style={{ color: 'black' }}>Improvement over Average</span>}
              value={improvement}
              precision={1}
              suffix="%"
              valueStyle={{ color: improvement > 0 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>
    </Card>
  );
} 