'use client';

import { useEffect, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Spin, Select, DatePicker, Button, Card, Typography, Row, Col, Statistic } from 'antd';
import { RangePickerProps } from 'antd/es/date-picker';
import { ClockCircleOutlined } from '@ant-design/icons';
import { TimeAnalysis } from '../models/types';
import { getTimeAnalysis } from '../controllers/analysisController';

const { Title } = Typography;
const { RangePicker } = DatePicker;

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface TimeAnalysisChartProps {
  datasetId: number;
}

export default function TimeAnalysisChart({ datasetId }: TimeAnalysisChartProps) {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<TimeAnalysis[]>([]);
  const [dateRange, setDateRange] = useState<[string, string]>(['2001-01-01', new Date().toISOString().split('T')[0]]);
  const [metric, setMetric] = useState<'views' | 'likes' | 'comments' | 'engagement'>('views');
  const [dayFilter, setDayFilter] = useState<string | undefined>(undefined);

  const dayOptions = [
    { value: undefined, label: 'All Days' },
    { value: 'sunday', label: 'Sunday' },
    { value: 'monday', label: 'Monday' },
    { value: 'tuesday', label: 'Tuesday' },
    { value: 'wednesday', label: 'Wednesday' },
    { value: 'thursday', label: 'Thursday' },
    { value: 'friday', label: 'Friday' },
    { value: 'saturday', label: 'Saturday' },
  ];

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getTimeAnalysis(datasetId, dateRange[0], dateRange[1], dayFilter);
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching time analysis data:', error);
    } finally {
      setLoading(false);
    }
  }, [datasetId, dateRange, dayFilter]);

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

  // Find best performing hour
  const getBestHour = () => {
    if (metrics.length === 0) return { hour: 0, avg: 0, improvement: 0 };
    
    const bestHourIdx = metrics.reduce((maxIdx, m, idx, arr) => 
      arr[idx].avgViews > arr[maxIdx].avgViews ? idx : maxIdx, 0);
    
    const bestHour = metrics[bestHourIdx].hour;
    const bestHourAvg = metrics[bestHourIdx].avgViews;
    
    // Calculate overall average views across all hours
    const totalVideos = metrics.reduce((sum, hour) => sum + hour.count, 0);
    const totalViews = metrics.reduce((sum, hour) => sum + (hour.avgViews * hour.count), 0);
    const overallAvg = totalVideos > 0 ? totalViews / totalVideos : 0;
    
    const improvement = overallAvg > 0 ? ((bestHourAvg / overallAvg) - 1) * 100 : 0;
    
    return { hour: bestHour, avg: bestHourAvg, improvement };
  };

  const { hour, avg, improvement } = getBestHour();

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value);
  };

  const formatHour = (hour: number) => {
    const ampm = hour >= 12 ? 'PM' : 'AM';
    const formattedHour = hour % 12 === 0 ? 12 : hour % 12;
    return `${formattedHour} ${ampm}`;
  };

  // Get metrics data based on selected metric
  const getMetricData = () => {
    switch (metric) {
      case 'views':
        return metrics.map(h => h.avgViews);
      case 'likes':
        return metrics.map(h => h.avgLikes);
      case 'comments':
        return metrics.map(h => h.avgComments);
      case 'engagement':
        return metrics.map(h => h.avgEngagement);
      default:
        return metrics.map(h => h.avgViews);
    }
  };

  // Get y-axis title based on selected metric
  const getYAxisTitle = () => {
    switch (metric) {
      case 'views':
        return "Average Views";
      case 'likes':
        return "Average Likes";
      case 'comments':
        return "Average Comments";
      case 'engagement':
        return "Average Engagement Rate (%)";
      default:
        return "Average Views";
    }
  };

  // Define plotly props to avoid type errors
  const plotlyProps = {
    data: [{
      x: metrics.map(h => formatHour(h.hour)),
      y: getMetricData(),
      type: 'bar',
      marker: {
        color: metrics.map((h, i) => 
          i === metrics.reduce((maxIdx, m, idx, arr) => 
            arr[idx].avgViews > arr[maxIdx].avgViews ? idx : maxIdx, 0) 
            ? '#1890ff' : '#36cfc9'
        )
      },
      text: metrics.map(h => `Videos: ${h.count}<br>Avg Views: ${formatNumber(h.avgViews)}`),
      hoverinfo: 'text+x'
    }],
    layout: {
      title: {
        text: "Performance by Hour of Day",
        font: {
          size: 20,
          color: "black"
        }
      },
      xaxis: {
        title: {
          text: "Hour of Day",
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
          text: getYAxisTitle(),
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
      <Title level={4} style={{ color: 'black' }}>Time of Day Analysis</Title>
      <p style={{ color: 'black' }}>Find the most favorable time to post videos based on historical performance.</p>
      
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <RangePicker 
            onChange={handleDateChange}
            defaultValue={[null, null]}
          />
        </Col>
        <Col span={6}>
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
        <Col span={6}>
          <Select
            value={dayFilter}
            onChange={setDayFilter}
            style={{ width: '100%' }}
            options={dayOptions}
            placeholder="Filter by day"
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
              title={<span style={{ color: 'black' }}>Best Time to Post</span>}
              value={formatHour(hour)}
              prefix={<ClockCircleOutlined style={{ color: 'black' }} />}
              valueStyle={{ color: 'black' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card style={{ backgroundColor: 'white' }}>
            <Statistic
              title={<span style={{ color: 'black' }}>Average Views at Best Time</span>}
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