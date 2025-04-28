'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { 
  Layout,
  Typography, 
  Card, 
  Row, 
  Col, 
  Button, 
  Spin, 
  Empty, 
  Statistic, 
  Divider,
  Space
} from 'antd';
import {
  BarChartOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  PlusOutlined,
  EyeOutlined,
  UploadOutlined,
  RiseOutlined
} from '@ant-design/icons';

const { Title, Paragraph, Text } = Typography;
const { Content } = Layout;

interface Dataset {
  id: number;
  name: string;
  description: string;
  videoCount: number;
  createdAt: string;
  viewCount?: number;
  growth?: number;
}

export default function AnalyticsPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  useEffect(() => {
    // Simulate loading data
    const timer = setTimeout(() => {
      setIsLoading(false);
      
      // Simulate some sample data
      setDatasets([
        {
          id: 1,
          name: "ADITYA MUSIC.csv",
          description: "Main channel analytics dataset",
          videoCount: 20000,
          createdAt: "28/04/2025",
          viewCount: 5243789,
          growth: 12.4
        },
        {
          id: 2,
          name: "Backup Channel",
          description: "Secondary channel performance data",
          videoCount: 450,
          createdAt: "15/04/2025",
          viewCount: 891245,
          growth: 8.7
        }
      ]);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  const handleSelectDataset = (datasetId: number) => {
    router.push(`/analytics/${datasetId}`);
  };

  return (
    <Layout style={{ background: '#fff', minHeight: '100vh' }}>
      <Content style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <Title level={2} style={{ color: '#000', margin: '0 0 16px 0' }}>
            Analytics Dashboard
          </Title>
          <Paragraph style={{ color: '#000', fontSize: '16px', maxWidth: '600px', margin: '0 auto' }}>
            Select a dataset to view detailed analytics and insights for your YouTube channel performance
          </Paragraph>
        </div>

        {isLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '5rem 0' }}>
            <Spin size="large" tip={<span style={{ color: '#000', marginTop: '15px' }}>Loading datasets...</span>} />
          </div>
        ) : datasets && datasets.length > 0 ? (
          <>
            <Row gutter={[24, 24]}>
              <Col xs={24} sm={24} md={24} lg={24}>
                <Title level={4} style={{ color: '#000', margin: '0 0 16px 0' }}>
                  <DatabaseOutlined style={{ marginRight: '8px' }} />
                  Your Datasets
                </Title>
              </Col>
            </Row>
            
            <Row gutter={[24, 24]}>
              {datasets.map((dataset) => (
                <Col xs={24} sm={12} md={8} key={dataset.id}>
                  <Card 
                    hoverable
                    onClick={() => handleSelectDataset(dataset.id)}
                    style={{ 
                      background: '#fff',
                      border: '1px solid #e8e8e8',
                      borderRadius: '12px',
                      overflow: 'hidden',
                      height: '100%'
                    }}
                    bodyStyle={{ padding: '20px' }}
                  >
                    <Title level={4} style={{ color: '#000', margin: '0 0 8px 0' }}>
                      {dataset.name}
                    </Title>
                    <Paragraph style={{ color: 'rgba(0, 0, 0, 0.8)', marginBottom: '20px' }}>
                      {dataset.description}
                    </Paragraph>
                    
                    <Divider style={{ borderColor: 'rgba(0, 0, 0, 0.15)', margin: '15px 0' }} />
                    
                    <Row gutter={[16, 16]}>
                      <Col span={12}>
                        <Statistic 
                          title={<Text style={{ color: 'rgba(0, 0, 0, 0.7)' }}>Videos</Text>}
                          value={dataset.videoCount}
                          valueStyle={{ color: '#000', fontSize: '20px' }}
                          prefix={<EyeOutlined />}
                        />
                      </Col>
                      {dataset.viewCount && (
                        <Col span={12}>
                          <Statistic 
                            title={<Text style={{ color: 'rgba(0, 0, 0, 0.7)' }}>Views</Text>}
                            value={dataset.viewCount}
                            valueStyle={{ color: '#000', fontSize: '20px' }}
                            formatter={(value) => `${(value as number / 1000000).toFixed(1)}M`}
                          />
                        </Col>
                      )}
                    </Row>
                    
                    <Row style={{ marginTop: '16px' }}>
                      <Col span={24}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Text style={{ color: 'rgba(0, 0, 0, 0.7)' }}>
                            Created: {dataset.createdAt}
                          </Text>
                          {dataset.growth && (
                            <div style={{ 
                              display: 'flex', 
                              alignItems: 'center', 
                              background: 'rgba(0, 0, 0, 0.05)',
                              padding: '4px 8px',
                              borderRadius: '12px'
                            }}>
                              <RiseOutlined style={{ color: '#52c41a', marginRight: '4px' }} />
                              <Text style={{ color: '#000' }}>{dataset.growth}%</Text>
                            </div>
                          )}
                        </div>
                      </Col>
                    </Row>
                    
                    <Button 
                      type="default" 
                      style={{ 
                        marginTop: '20px',
                        border: '1px solid rgba(0, 0, 0, 0.2)',
                        color: '#000',
                        width: '100%',
                        height: '40px',
                        borderRadius: '8px',
                        background: 'rgba(0, 0, 0, 0.05)'
                      }}
                      icon={<BarChartOutlined />}
                    >
                      View Analytics
                    </Button>
                  </Card>
                </Col>
              ))}
              
              <Col xs={24} sm={12} md={8}>
                <Card 
                  hoverable
                  onClick={() => router.push('/dataset')}
                  style={{ 
                    background: '#fff',
                    border: '1px solid #e8e8e8',
                    borderRadius: '12px',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                  bodyStyle={{ 
                    padding: '20px', 
                    display: 'flex', 
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                    textAlign: 'center'
                  }}
                >
                  <div style={{ 
                    width: '60px', 
                    height: '60px', 
                    borderRadius: '50%', 
                    background: 'rgba(0, 0, 0, 0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '16px'
                  }}>
                    <PlusOutlined style={{ fontSize: '24px', color: '#000' }} />
                  </div>
                  
                  <Title level={5} style={{ color: '#000', margin: '0 0 8px 0' }}>
                    Add New Dataset
                  </Title>
                  
                  <Paragraph style={{ color: 'rgba(0, 0, 0, 0.7)', marginBottom: '20px' }}>
                    Upload a new YouTube dataset for analysis
                  </Paragraph>
                  
                  <Button
                    type="primary"
                    icon={<UploadOutlined />}
                    style={{ 
                      background: '#1890ff',
                      border: 'none'
                    }}
                  >
                    Upload Dataset
                  </Button>
                </Card>
              </Col>
            </Row>
            
            <Row gutter={[24, 24]} style={{ marginTop: '36px' }}>
              <Col xs={24}>
                <Card 
                  style={{ 
                    background: '#fff',
                    border: '1px solid #e8e8e8',
                    borderRadius: '12px'
                  }}
                  bodyStyle={{ padding: '24px' }}
                >
                  <Row gutter={[24, 24]} align="middle">
                    <Col xs={24} sm={16} md={18}>
                      <Title level={4} style={{ color: '#000', margin: '0 0 8px 0' }}>
                        Generate AI Insights
                      </Title>
                      <Paragraph style={{ color: 'rgba(0, 0, 0, 0.8)', margin: 0 }}>
                        Use machine learning to predict trends and optimize your content strategy
                      </Paragraph>
                    </Col>
                    <Col xs={24} sm={8} md={6} style={{ textAlign: 'right' }}>
                      <Button 
                        type="default" 
                        size="large"
                        icon={<LineChartOutlined />}
                        style={{ 
                          background: 'rgba(0, 0, 0, 0.05)',
                          borderColor: 'rgba(0, 0, 0, 0.2)',
                          color: '#000',
                          height: '48px',
                          borderRadius: '8px',
                        }}
                      >
                        Run Analysis
                      </Button>
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </>
        ) : (
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center',
            padding: '5rem 0' 
          }}>
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <span style={{ color: '#000' }}>No datasets available</span>
              }
            />
            <Button 
              type="primary" 
              icon={<UploadOutlined />} 
              size="large" 
              onClick={() => router.push('/dataset')}
              style={{ 
                marginTop: '24px',
                background: '#1890ff',
                border: 'none',
                height: '48px',
                paddingLeft: '24px',
                paddingRight: '24px',
                borderRadius: '8px',
                fontSize: '16px'
              }}
            >
              Upload Your First Dataset
            </Button>
          </div>
        )}
      </Content>
    </Layout>
  );
} 