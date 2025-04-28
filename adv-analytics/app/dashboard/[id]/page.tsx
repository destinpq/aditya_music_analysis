'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { Layout, Menu, Typography, Row, Col, Card, Statistic, Tabs, Spin, Alert } from 'antd';
import { BarChartOutlined, LineChartOutlined, TagsOutlined, DollarOutlined } from '@ant-design/icons';
import DayAnalysisChart from '../../components/DayAnalysisChart';
import TimeAnalysisChart from '../../components/TimeAnalysisChart';
import TagAnalysisChart from '../../components/TagAnalysisChart';
import RevenuePredictionChart from '../../components/RevenuePredictionChart';
import { AnalysisResult } from '../../models/types';

const { Header, Content, Sider } = Layout;
const { Title, Paragraph } = Typography;
const { TabPane } = Tabs;

// Mock data for demonstration
const mockAnalysisResult: AnalysisResult = {
  avgViews: 45678,
  avgLikes: 3245,
  plotJson: "{}",
  predictedAiViews: 65432,
  modelR2: 0.8521,
  featureImportance: {
    features: ['title_length', 'duration_seconds', 'day_of_week'],
    importance: [0.45, 0.32, 0.23]
  },
  revenueProjections: {
    totalMonthlyViews: 39259200,
    profitScenarios: {
      'Low Cost': {
        costPerVideo: 100,
        totalCost: 60000,
        profitsByCountry: {
          'India': {
            revenue: 1080628,
            cost: 60000,
            profit: 1020628,
            roi: 1701.05
          }
        }
      }
    },
    breakevenPoints: {
      'India': 90.05
    },
    chartData: {}
  },
  priceAnalysis: {},
  productionVolume: {}
};

export default function Dashboard() {
  const params = useParams();
  const { id } = params;
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisResult | null>(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        // In a real app, fetch from API:
        // const response = await fetch(`/api/analysis/${id}`);
        // if (!response.ok) throw new Error('Failed to fetch data');
        // const result = await response.json();
        
        // For demo, use mock data after delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        setData(mockAnalysisResult);
      } catch (err) {
        setError('Failed to load dashboard data');
        console.error('Error fetching dashboard data:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [id]);
  
  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" tip="Loading dashboard data..." />
      </div>
    );
  }
  
  if (error) {
    return (
      <div style={{ padding: '2rem' }}>
        <Alert
          message="Error Loading Dashboard"
          description={error}
          type="error"
          showIcon
        />
      </div>
    );
  }
  
  return (
    <Layout style={{ minHeight: '100vh', width: '100vw', backgroundColor: 'white' }} className="light-theme">
      <Header style={{ display: 'flex', alignItems: 'center', background: 'white', width: '100%', borderBottom: '1px solid #e8e8e8' }}>
        <div style={{ color: 'black', fontSize: '18px', fontWeight: 'bold' }}>
          Video Analytics Dashboard
        </div>
      </Header>
      
      <Layout style={{ backgroundColor: 'white' }}>
        <Content
          style={{
            padding: 24,
            margin: 0,
            width: '100%',
            background: 'white', 
            color: 'black'
          }}
        >
          <div className="analytics-header" style={{ marginBottom: '24px' }}>
            <Title level={3} style={{ color: 'black', margin: 0 }}>Video Performance Analytics</Title>
            <Paragraph style={{ color: 'rgba(0, 0, 0, 0.65)', margin: '8px 0 0 0' }}>
              Data insights and AI-powered analysis for dataset #{id}
            </Paragraph>
          </div>
          
          {/* Summary Cards */}
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <Card style={{ background: 'white', borderColor: '#e8e8e8' }}>
                <Statistic
                  title={<span style={{ color: 'rgba(0, 0, 0, 0.85)' }}>Average Views</span>}
                  value={data?.avgViews || 0}
                  formatter={value => value.toLocaleString()}
                  valueStyle={{ color: 'black' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card style={{ background: 'white', borderColor: '#e8e8e8' }}>
                <Statistic
                  title={<span style={{ color: 'rgba(0, 0, 0, 0.85)' }}>Average Likes</span>}
                  value={data?.avgLikes || 0}
                  formatter={value => value.toLocaleString()}
                  valueStyle={{ color: 'black' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card style={{ background: 'white', border: '1px solid #e8e8e8' }}>
                <Statistic
                  title={<span style={{ color: 'rgba(0, 0, 0, 0.85)' }}>Predicted AI Views</span>}
                  value={data?.predictedAiViews || 0}
                  formatter={value => value.toLocaleString()}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card style={{ background: 'white', borderColor: '#e8e8e8' }}>
                <Statistic
                  title={<span style={{ color: 'rgba(0, 0, 0, 0.85)' }}>Model Accuracy</span>}
                  value={data?.modelR2 || 0}
                  precision={2}
                  suffix="%"
                  formatter={value => (Number(value) * 100).toFixed(2)}
                  valueStyle={{ color: 'black' }}
                />
              </Card>
            </Col>
          </Row>
          
          {/* Analytics Charts */}
          <Tabs 
            defaultActiveKey="1" 
            style={{ color: 'black' }}
            className="custom-light-tabs"
          >
            <TabPane tab="Day Analysis" key="1">
              <DayAnalysisChart datasetId={Number(id)} />
            </TabPane>
            <TabPane tab="Time Analysis" key="2">
              <TimeAnalysisChart datasetId={Number(id)} />
            </TabPane>
            <TabPane tab="Tag Analysis" key="3">
              <TagAnalysisChart datasetId={Number(id)} />
            </TabPane>
            <TabPane tab="Revenue Projections" key="4">
              <RevenuePredictionChart datasetId={Number(id)} initialViewEstimate={data?.predictedAiViews || 50000} />
            </TabPane>
          </Tabs>
        </Content>
      </Layout>
    </Layout>
  );
} 