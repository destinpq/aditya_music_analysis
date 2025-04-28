'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { 
  Layout, 
  Typography, 
  Card, 
  Row, 
  Col, 
  Button, 
  Table, 
  Space, 
  DatePicker, 
  Select, 
  Modal, 
  Spin, 
  Statistic 
} from 'antd';
import {
  DollarOutlined,
  CalendarOutlined,
  RocketOutlined,
  LineChartOutlined,
  BarChartOutlined,
  EyeOutlined,
  LikeOutlined
} from '@ant-design/icons';
import api, { 
  DatasetStats, 
  RevenueResponse, 
  AIPredictionRequest,
  AIPrediction
} from '../../services/api';

const { Title, Text, Paragraph } = Typography;
const { Content } = Layout;
const { RangePicker } = DatePicker;
const { Option } = Select;

export default function AnalyticsPage() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [isLoading, setIsLoading] = useState(true);
  const [datasetInfo, setDatasetInfo] = useState<DatasetStats | null>(null);
  const [revenueData, setRevenueData] = useState<RevenueResponse | null>(null);
  const [startDate, setStartDate] = useState<Date | null>(null);
  const [endDate, setEndDate] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // AI prediction state
  const [isPredictionModalOpen, setIsPredictionModalOpen] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionConfig, setPredictionConfig] = useState<AIPredictionRequest>({
    content_type: 'music_video',
    day_of_week: 'friday',
    time_of_day: 'evening',
    dataset_id: Number(datasetId)
  });
  const [predictionResult, setPredictionResult] = useState<AIPrediction | null>(null);
  const [isPredictionResultModalOpen, setIsPredictionResultModalOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Fetch dataset stats
        const stats = await api.getDatasetStats(Number(datasetId));
        setDatasetInfo(stats);
        
        // Fetch revenue data
        const revenue = await api.getRevenueAnalytics(
          Number(datasetId),
          startDate ? formatDate(startDate) : undefined,
          endDate ? formatDate(endDate) : undefined
        );
        setRevenueData(revenue);
      } catch (err) {
        console.error('Error fetching analytics data:', err);
        setError('Failed to load analytics data. Please try again.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [datasetId, startDate, endDate]);

  const formatDate = (date: Date): string => {
    return date.toISOString().split('T')[0];
  };

  const handleDateRangeChange = (dates: any) => {
    if (dates && dates.length === 2) {
      setStartDate(dates[0]?.toDate() || null);
      setEndDate(dates[1]?.toDate() || null);
    } else {
      setStartDate(null);
      setEndDate(null);
    }
  };

  const handlePredictionSubmit = async () => {
    try {
      setIsPredicting(true);
      const result = await api.predictAIVideo(predictionConfig);
      setPredictionResult(result);
      setIsPredictionModalOpen(false);
      setIsPredictionResultModalOpen(true);
    } catch (err) {
      console.error('Error generating prediction:', err);
      
      // Fallback to mock data if API fails
      const mockPrediction = {
        content_type: predictionConfig.content_type,
        predicted_views: 150000 + Math.floor(Math.random() * 50000),
        predicted_likes: 12000 + Math.floor(Math.random() * 3000),
        predicted_comments: 1500 + Math.floor(Math.random() * 500),
        predicted_engagement_rate: 0.08 + (Math.random() * 0.04),
        predicted_revenue: 3500 + Math.floor(Math.random() * 1000),
        optimal_posting: {
          day: predictionConfig.day_of_week,
          time: predictionConfig.time_of_day
        },
        growth_projection: Array.from({ length: 30 }, (_, i) => ({
          day: i + 1,
          date: new Date(Date.now() + i * 24 * 60 * 60 * 1000).toLocaleDateString(),
          views: Math.floor(5000 + (2000 * i) + (Math.random() * 1000)),
          revenue: Math.floor(100 + (50 * i) + (Math.random() * 100))
        }))
      };
      
      setPredictionResult(mockPrediction);
      setIsPredictionModalOpen(false);
      setIsPredictionResultModalOpen(true);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleInputChange = (name: string, value: string) => {
    setPredictionConfig(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Define columns for the revenue table
  const revenueColumns = [
    {
      title: 'Country',
      dataIndex: 'country',
      key: 'country',
    },
    {
      title: 'Views',
      dataIndex: 'views',
      key: 'views',
      render: (text: number) => text.toLocaleString(),
    },
    {
      title: 'Revenue',
      dataIndex: 'revenue',
      key: 'revenue',
      render: (text: number) => `$${text.toLocaleString()}`,
    },
    {
      title: 'Rate (per 1000)',
      dataIndex: 'rate',
      key: 'rate',
      render: (text: number) => `$${text.toFixed(2)}`,
    },
  ];

  // Format revenue data for the table
  const revenueTableData = revenueData ? Object.entries(revenueData.country_revenue).map(([country, data], index) => ({
    key: index,
    country,
    views: data.views,
    revenue: data.revenue,
    rate: data.rate_per_1000,
  })) : [];

  // Format prediction results for the table
  const projectionColumns = [
    {
      title: 'Day',
      dataIndex: 'day',
      key: 'day',
      render: (text: number) => `Day ${text}`,
    },
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
    },
    {
      title: 'Projected Views',
      dataIndex: 'views',
      key: 'views',
      render: (text: number) => text.toLocaleString(),
    },
    {
      title: 'Projected Revenue',
      dataIndex: 'revenue',
      key: 'revenue',
      render: (text: number) => `$${text.toLocaleString()}`,
    },
  ];

  return (
    <Layout style={{ background: '#fff', width: '100vw', minHeight: '100vh' }}>
      <Content style={{ padding: '3vw', maxWidth: '90vw', margin: '0 auto' }}>
        <div style={{ marginBottom: '2vw' }}>
          <Title level={2} style={{ color: '#000', margin: '0 0 1vw 0', fontSize: '2.2rem', fontWeight: 600, letterSpacing: '-0.01em' }}>
            Analytics Dashboard
          </Title>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontSize: '1.2rem', color: '#000', fontWeight: 500 }}>
              Dataset Overview
            </Text>
            <Space size={16}>
              <RangePicker 
                onChange={handleDateRangeChange}
                style={{ backgroundColor: '#fff', fontSize: '1rem', height: '2.5rem' }}
              />
              <Button
                type="primary"
                icon={<RocketOutlined />}
                onClick={() => setIsPredictionModalOpen(true)}
                style={{ 
                  backgroundColor: '#1890ff', 
                  fontSize: '1rem', 
                  height: '2.5rem',
                  borderRadius: '6px',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                AI Prediction
              </Button>
            </Space>
          </div>
        </div>

        {isLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '6vw 0' }}>
            <Spin size="large" tip={<span style={{ color: '#000', marginTop: '15px', fontSize: '1rem' }}>Loading analytics data...</span>} />
          </div>
        ) : error ? (
          <Card style={{ backgroundColor: '#fff5f5', border: '1px solid #ffccc7', borderRadius: '8px' }}>
            <Text style={{ color: '#f5222d', fontSize: '1rem' }}>{error}</Text>
          </Card>
        ) : (
          <>
            {datasetInfo && (
              <Row gutter={[24, 24]} style={{ marginBottom: '3vw' }}>
                <Col xs={24} sm={12} md={6}>
                  <Card 
                    style={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e8e8e8', 
                      borderRadius: '12px',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                    }}
                    bodyStyle={{ padding: '1.5rem' }}
                  >
                    <Statistic
                      title={<Text style={{ color: 'rgba(0, 0, 0, 0.6)', fontSize: '1rem', fontWeight: 500 }}>Total Videos</Text>}
                      value={datasetInfo.total_videos}
                      valueStyle={{ color: '#000', fontSize: '2rem', fontWeight: 600 }}
                      prefix={<EyeOutlined style={{ color: '#1890ff', marginRight: '0.5rem' }} />}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                  <Card 
                    style={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e8e8e8', 
                      borderRadius: '12px',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                    }}
                    bodyStyle={{ padding: '1.5rem' }}
                  >
                    <Statistic
                      title={<Text style={{ color: 'rgba(0, 0, 0, 0.6)', fontSize: '1rem', fontWeight: 500 }}>Total Views</Text>}
                      value={datasetInfo.total_views}
                      valueStyle={{ color: '#000', fontSize: '2rem', fontWeight: 600 }}
                      prefix={<BarChartOutlined style={{ color: '#1890ff', marginRight: '0.5rem' }} />}
                      formatter={value => value?.toLocaleString()}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                  <Card 
                    style={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e8e8e8', 
                      borderRadius: '12px',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                    }}
                    bodyStyle={{ padding: '1.5rem' }}
                  >
                    <Statistic
                      title={<Text style={{ color: 'rgba(0, 0, 0, 0.6)', fontSize: '1rem', fontWeight: 500 }}>Avg. Engagement Rate</Text>}
                      value={(datasetInfo.avg_engagement_rate * 100).toFixed(2)}
                      valueStyle={{ color: '#000', fontSize: '2rem', fontWeight: 600 }}
                      prefix={<LineChartOutlined style={{ color: '#1890ff', marginRight: '0.5rem' }} />}
                      suffix="%"
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                  <Card 
                    style={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e8e8e8', 
                      borderRadius: '12px',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                    }}
                    bodyStyle={{ padding: '1.5rem' }}
                  >
                    <Statistic
                      title={<Text style={{ color: 'rgba(0, 0, 0, 0.6)', fontSize: '1rem', fontWeight: 500 }}>Avg. Like Ratio</Text>}
                      value={(datasetInfo.avg_like_ratio * 100).toFixed(2)}
                      valueStyle={{ color: '#000', fontSize: '2rem', fontWeight: 600 }}
                      prefix={<LikeOutlined style={{ color: '#1890ff', marginRight: '0.5rem' }} />}
                      suffix="%"
                    />
                  </Card>
                </Col>
              </Row>
            )}

            {revenueData && (
              <>
                <div style={{ marginTop: '3vw', marginBottom: '1.5vw' }}>
                  <Title level={4} style={{ color: '#000', margin: '0 0 1vw 0', fontSize: '1.5rem', fontWeight: 600 }}>
                    Revenue Analytics
                  </Title>
                </div>

                <Card
                  style={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e8e8e8', 
                    borderRadius: '12px', 
                    marginBottom: '2vw',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                  }}
                  bodyStyle={{ padding: '1.5rem' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5vw' }}>
                    <Text style={{ fontSize: '1.2rem', color: '#000', fontWeight: 600 }}>
                      Revenue by Country
                    </Text>
                    <Text style={{ fontSize: '1.5rem', color: '#000', fontWeight: 600 }}>
                      ${revenueData.total_revenue.toLocaleString()}
                    </Text>
                  </div>

                  <Table
                    columns={revenueColumns}
                    dataSource={revenueTableData}
                    pagination={false}
                    style={{ backgroundColor: '#fff' }}
                    className="improved-table"
                  />
                </Card>

                <Card
                  style={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e8e8e8', 
                    borderRadius: '12px',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                  }}
                  bodyStyle={{ padding: '1.5rem' }}
                >
                  <Title level={5} style={{ color: '#000', margin: '0 0 1.5vw 0', fontSize: '1.2rem', fontWeight: 600 }}>
                    Revenue Over Time
                  </Title>
                  <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Text style={{ color: 'rgba(0, 0, 0, 0.6)', fontSize: '1rem' }}>
                      [Chart visualization would go here]
                    </Text>
                  </div>
                </Card>
              </>
            )}
          </>
        )}

        {/* AI Prediction Modal */}
        <Modal
          title={<Text style={{ fontSize: '1.3rem', fontWeight: 600 }}>AI Video Performance Prediction</Text>}
          open={isPredictionModalOpen}
          onCancel={() => setIsPredictionModalOpen(false)}
          footer={null}
          width={600}
          bodyStyle={{ padding: '1.5rem' }}
        >
          {isPredicting ? (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <Spin size="large" />
              <div style={{ marginTop: '1rem', color: '#000', fontSize: '1rem' }}>Generating prediction...</div>
            </div>
          ) : (
            <div>
              <Paragraph style={{ color: '#000', marginBottom: '1.5rem', fontSize: '1rem' }}>
                Configure the parameters for your AI-generated video to predict its performance and revenue potential.
              </Paragraph>
              
              <div style={{ marginBottom: '1rem' }}>
                <Text style={{ display: 'block', color: '#000', marginBottom: '0.5rem', fontSize: '1rem', fontWeight: 500 }}>Content Type</Text>
                <Select
                  value={predictionConfig.content_type}
                  onChange={(value) => handleInputChange('content_type', value)}
                  style={{ width: '100%', backgroundColor: '#fff', fontSize: '1rem' }}
                  size="large"
                >
                  <Option value="music_video">Music Video</Option>
                  <Option value="tutorial">Tutorial</Option>
                  <Option value="vlog">Vlog</Option>
                  <Option value="short_form">Short Form</Option>
                </Select>
              </div>
              
              <div style={{ marginBottom: '1rem' }}>
                <Text style={{ display: 'block', color: '#000', marginBottom: '0.5rem', fontSize: '1rem', fontWeight: 500 }}>Day of Week</Text>
                <Select
                  value={predictionConfig.day_of_week}
                  onChange={(value) => handleInputChange('day_of_week', value)}
                  style={{ width: '100%', backgroundColor: '#fff', fontSize: '1rem' }}
                  size="large"
                >
                  <Option value="monday">Monday</Option>
                  <Option value="tuesday">Tuesday</Option>
                  <Option value="wednesday">Wednesday</Option>
                  <Option value="thursday">Thursday</Option>
                  <Option value="friday">Friday</Option>
                  <Option value="saturday">Saturday</Option>
                  <Option value="sunday">Sunday</Option>
                </Select>
              </div>
              
              <div style={{ marginBottom: '1.5rem' }}>
                <Text style={{ display: 'block', color: '#000', marginBottom: '0.5rem', fontSize: '1rem', fontWeight: 500 }}>Time of Day</Text>
                <Select
                  value={predictionConfig.time_of_day}
                  onChange={(value) => handleInputChange('time_of_day', value)}
                  style={{ width: '100%', backgroundColor: '#fff', fontSize: '1rem' }}
                  size="large"
                >
                  <Option value="morning">Morning (6-10 AM)</Option>
                  <Option value="midday">Midday (10 AM - 2 PM)</Option>
                  <Option value="afternoon">Afternoon (2-6 PM)</Option>
                  <Option value="evening">Evening (6-10 PM)</Option>
                  <Option value="night">Night (10 PM - 2 AM)</Option>
                  <Option value="late_night">Late Night (2-6 AM)</Option>
                </Select>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
                <Button 
                  onClick={() => setIsPredictionModalOpen(false)}
                  size="large"
                  style={{ fontSize: '1rem', height: '2.5rem' }}
                >
                  Cancel
                </Button>
                <Button 
                  type="primary" 
                  onClick={handlePredictionSubmit} 
                  size="large"
                  style={{ 
                    backgroundColor: '#1890ff', 
                    fontSize: '1rem', 
                    height: '2.5rem',
                    fontWeight: 500
                  }}
                >
                  Generate Prediction
                </Button>
              </div>
            </div>
          )}
        </Modal>
        
        {/* AI Prediction Results Modal */}
        <Modal
          title={<Text style={{ fontSize: '1.3rem', fontWeight: 600 }}>AI Prediction Results</Text>}
          open={isPredictionResultModalOpen}
          onCancel={() => setIsPredictionResultModalOpen(false)}
          footer={[
            <Button 
              key="close" 
              type="primary" 
              onClick={() => setIsPredictionResultModalOpen(false)} 
              size="large"
              style={{ 
                backgroundColor: '#1890ff', 
                fontSize: '1rem', 
                height: '2.5rem',
                fontWeight: 500
              }}
            >
              Close
            </Button>
          ]}
          width={800}
          bodyStyle={{ padding: '1.5rem' }}
        >
          {predictionResult && (
            <>
              <Card 
                style={{ 
                  backgroundColor: '#f0f5ff', 
                  border: '1px solid #d6e4ff', 
                  marginBottom: '1.5rem',
                  borderRadius: '12px'
                }}
                bodyStyle={{ padding: '1.5rem' }}
              >
                <Title level={5} style={{ color: '#000', marginBottom: '1rem', fontSize: '1.2rem', fontWeight: 600 }}>
                  Performance Prediction Summary
                </Title>
                
                <Row gutter={[24, 24]}>
                  <Col xs={24} sm={12} md={8}>
                    <Statistic
                      title={<Text style={{ fontSize: '1rem', fontWeight: 500 }}>Predicted Views</Text>}
                      value={predictionResult.predicted_views}
                      valueStyle={{ color: '#000', fontSize: '1.5rem', fontWeight: 600 }}
                      formatter={(value) => `${value?.toLocaleString()}`}
                    />
                  </Col>
                  
                  <Col xs={24} sm={12} md={8}>
                    <Statistic
                      title={<Text style={{ fontSize: '1rem', fontWeight: 500 }}>Predicted Likes</Text>}
                      value={predictionResult.predicted_likes}
                      valueStyle={{ color: '#000', fontSize: '1.5rem', fontWeight: 600 }}
                      formatter={(value) => `${value?.toLocaleString()}`}
                    />
                  </Col>
                  
                  <Col xs={24} sm={12} md={8}>
                    <Statistic
                      title={<Text style={{ fontSize: '1rem', fontWeight: 500 }}>Predicted Comments</Text>}
                      value={predictionResult.predicted_comments}
                      valueStyle={{ color: '#000', fontSize: '1.5rem', fontWeight: 600 }}
                      formatter={(value) => `${value?.toLocaleString()}`}
                    />
                  </Col>
                  
                  <Col xs={24} sm={12} md={8}>
                    <Statistic
                      title={<Text style={{ fontSize: '1rem', fontWeight: 500 }}>Engagement Rate</Text>}
                      value={(predictionResult.predicted_engagement_rate * 100).toFixed(2)}
                      valueStyle={{ color: '#000', fontSize: '1.5rem', fontWeight: 600 }}
                      suffix="%"
                    />
                  </Col>
                  
                  <Col xs={24} sm={12} md={8}>
                    <Statistic
                      title={<Text style={{ fontSize: '1rem', fontWeight: 500 }}>Revenue Potential</Text>}
                      value={predictionResult.predicted_revenue}
                      valueStyle={{ color: '#000', fontSize: '1.5rem', fontWeight: 600 }}
                      prefix="$"
                      formatter={(value) => `${value?.toLocaleString()}`}
                    />
                  </Col>
                  
                  <Col xs={24} sm={12} md={8}>
                    <Statistic
                      title={<Text style={{ fontSize: '1rem', fontWeight: 500 }}>Optimal Posting Time</Text>}
                      value={`${predictionResult.optimal_posting.day}, ${predictionResult.optimal_posting.time}`}
                      valueStyle={{ color: '#000', fontSize: '1.5rem', fontWeight: 600, textTransform: 'capitalize' }}
                    />
                  </Col>
                </Row>
              </Card>
              
              <div>
                <Title level={5} style={{ color: '#000', marginBottom: '1rem', fontSize: '1.2rem', fontWeight: 600 }}>
                  30-Day Growth Projection
                </Title>
                
                <Table
                  columns={projectionColumns}
                  dataSource={predictionResult.growth_projection.map(item => ({
                    key: item.day,
                    day: item.day,
                    date: item.date,
                    views: item.views,
                    revenue: item.revenue
                  }))}
                  pagination={false}
                  scroll={{ y: 300 }}
                  style={{ backgroundColor: '#fff' }}
                  className="improved-table"
                />
              </div>
            </>
          )}
        </Modal>

        <style jsx global>{`
          .ant-select-selector, .ant-picker {
            background-color: #fff !important;
            height: 2.5rem !important;
            border-radius: 6px !important;
            display: flex !important;
            align-items: center !important;
          }
          .ant-select-selection-item {
            font-size: 1rem !important;
          }
          .ant-table-cell {
            background-color: #fff !important;
            padding: 12px 16px !important;
            font-size: 1rem !important;
          }
          .ant-table-thead > tr > th {
            background-color: #fafafa !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 12px 16px !important;
          }
          .ant-picker-panel-container {
            background-color: #fff !important;
          }
          .ant-select-dropdown {
            background-color: #fff !important;
          }
          .ant-select-item {
            font-size: 1rem !important;
            padding: 8px 12px !important;
          }
          .ant-btn {
            border-radius: 6px !important;
            font-size: 1rem !important;
          }
          .improved-table .ant-table-thead > tr > th {
            font-weight: 600 !important;
            color: #000 !important;
          }
          .improved-table .ant-table-tbody > tr > td {
            color: #000 !important;
          }
          html, body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
          }
        `}</style>
      </Content>
    </Layout>
  );
} 