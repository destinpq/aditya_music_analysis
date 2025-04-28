'use client';

import { useEffect, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Spin, Select, DatePicker, Button, Card, Typography, Row, Col, Statistic, Table } from 'antd';
import { RangePickerProps } from 'antd/es/date-picker';
import { TagsOutlined } from '@ant-design/icons';
import { TagAnalysis } from '../models/types';
import { getTagAnalysis } from '../controllers/analysisController';

const { Title } = Typography;
const { RangePicker } = DatePicker;

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface TagAnalysisChartProps {
  datasetId: number;
}

export default function TagAnalysisChart({ datasetId }: TagAnalysisChartProps) {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<TagAnalysis[]>([]);
  const [dateRange, setDateRange] = useState<[string, string]>(['2001-01-01', new Date().toISOString().split('T')[0]]);
  const [metric, setMetric] = useState<'views' | 'likes' | 'comments' | 'engagement'>('views');
  const [sortField, setSortField] = useState<'count' | 'avgViews' | 'avgLikes' | 'avgComments' | 'avgEngagement'>('avgViews');
  const [tablePageSize, setTablePageSize] = useState(5);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getTagAnalysis(datasetId, dateRange[0], dateRange[1]);
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching tag analysis data:', error);
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

  // Find best performing tag
  const getBestTag = () => {
    if (metrics.length === 0) return { tag: '', avg: 0, improvement: 0 };
    
    const sortedTags = [...metrics].sort((a, b) => b.avgViews - a.avgViews);
    const bestTag = sortedTags[0];
    
    // Calculate overall average views across all tags
    const totalVideos = metrics.reduce((sum, tag) => sum + tag.count, 0);
    const totalViews = metrics.reduce((sum, tag) => sum + (tag.avgViews * tag.count), 0);
    const overallAvg = totalVideos > 0 ? totalViews / totalVideos : 0;
    
    const improvement = overallAvg > 0 ? ((bestTag.avgViews / overallAvg) - 1) * 100 : 0;
    
    return { tag: bestTag.name, avg: bestTag.avgViews, improvement };
  };

  const { tag, avg, improvement } = getBestTag();

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value);
  };

  // Sort metrics based on selected field
  const sortedMetrics = [...metrics].sort((a, b) => b[sortField] - a[sortField]);

  // Top performing tags for the chart (limit to top 10)
  const topTags = sortedMetrics.slice(0, 10);

  // Get metrics data based on selected metric
  const getMetricData = () => {
    switch (metric) {
      case 'views':
        return topTags.map(t => t.avgViews);
      case 'likes':
        return topTags.map(t => t.avgLikes);
      case 'comments':
        return topTags.map(t => t.avgComments);
      case 'engagement':
        return topTags.map(t => t.avgEngagement);
      default:
        return topTags.map(t => t.avgViews);
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
      x: topTags.map(t => t.name),
      y: getMetricData(),
      type: 'bar',
      marker: {
        color: topTags.map((t, i) => 
          i === 0 ? '#1890ff' : '#36cfc9'
        )
      },
      text: topTags.map(t => `Videos: ${t.count}<br>Avg Views: ${formatNumber(t.avgViews)}`),
      hoverinfo: 'text+x'
    }],
    layout: {
      title: {
        text: "Top Performing Tags",
        font: {
          size: 20,
          color: "black"
        }
      },
      xaxis: {
        title: {
          text: "Tag Name",
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
      margin: { t: 50, b: 100, l: 80, r: 30 },
      showlegend: false,
      autosize: true,
    },
    style: { width: '100%', height: 400 },
    config: { 
      responsive: true,
      displayModeBar: false
    }
  };

  // Table columns definition
  const columns = [
    {
      title: 'Tag',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Videos',
      dataIndex: 'count',
      key: 'count',
      render: (count: number) => formatNumber(count)
    },
    {
      title: 'Avg Views',
      dataIndex: 'avgViews',
      key: 'avgViews',
      render: (views: number) => formatNumber(views),
      sorter: (a: TagAnalysis, b: TagAnalysis) => a.avgViews - b.avgViews,
    },
    {
      title: 'Avg Likes',
      dataIndex: 'avgLikes',
      key: 'avgLikes',
      render: (likes: number) => formatNumber(likes),
      sorter: (a: TagAnalysis, b: TagAnalysis) => a.avgLikes - b.avgLikes,
    },
    {
      title: 'Avg Comments',
      dataIndex: 'avgComments',
      key: 'avgComments',
      render: (comments: number) => formatNumber(comments),
      sorter: (a: TagAnalysis, b: TagAnalysis) => a.avgComments - b.avgComments,
    },
    {
      title: 'Engagement',
      dataIndex: 'avgEngagement',
      key: 'avgEngagement',
      render: (engagement: number) => `${engagement.toFixed(2)}%`,
      sorter: (a: TagAnalysis, b: TagAnalysis) => a.avgEngagement - b.avgEngagement,
    },
  ];

  return (
    <Card style={{ backgroundColor: 'white' }}>
      <Title level={4} style={{ color: 'black' }}>Tag Analysis</Title>
      <p style={{ color: 'black' }}>Analyze the performance of different tags used in your videos.</p>
      
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={10}>
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
        <Col span={4}>
          <Select
            value={sortField}
            onChange={setSortField}
            style={{ width: '100%' }}
            options={[
              { value: 'count', label: 'Usage' },
              { value: 'avgViews', label: 'Views' },
              { value: 'avgLikes', label: 'Likes' },
              { value: 'avgComments', label: 'Comments' },
              { value: 'avgEngagement', label: 'Engagement' },
            ]}
            placeholder="Sort by"
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
      
      <Row gutter={16} style={{ marginTop: 24, marginBottom: 24 }}>
        <Col span={8}>
          <Card style={{ backgroundColor: 'white' }}>
            <Statistic
              title={<span style={{ color: 'black' }}>Best Performing Tag</span>}
              value={tag}
              prefix={<TagsOutlined style={{ color: 'black' }} />}
              valueStyle={{ color: 'black' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card style={{ backgroundColor: 'white' }}>
            <Statistic
              title={<span style={{ color: 'black' }}>Average Views with Best Tag</span>}
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
      
      <Title level={4} style={{ color: 'black', marginTop: 24 }}>Tag Performance Table</Title>
      <Table 
        dataSource={sortedMetrics} 
        columns={columns} 
        rowKey="name"
        pagination={{ pageSize: tablePageSize, showSizeChanger: true, pageSizeOptions: ['5', '10', '20'] }}
        onChange={(pagination) => setTablePageSize(pagination.pageSize || 5)}
        style={{ backgroundColor: 'white' }}
      />
    </Card>
  );
} 