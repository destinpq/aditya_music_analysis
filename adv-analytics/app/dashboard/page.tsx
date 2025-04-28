'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { 
  Layout,
  Typography, 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Button,
  Divider
} from 'antd';
import {
  DatabaseOutlined,
  VideoCameraOutlined,
  RocketOutlined,
  BarChartOutlined,
  BulbOutlined,
  ArrowUpOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Content } = Layout;

export default function Dashboard() {
  const router = useRouter();

  return (
    <Layout style={{ background: '#fff', minHeight: '100vh' }}>
      <Content style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
        <div>
          <Title level={2} style={{ color: '#000', margin: '0 0 8px 0' }}>
            Dashboard
          </Title>
          <Text style={{ color: '#000', fontSize: '16px' }}>
            Quick Overview
          </Text>
        </div>

        <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
          {/* Datasets Card */}
          <Col xs={24} sm={24} md={12} lg={6}>
            <Card 
              style={{ 
                background: '#fff',
                border: '1px solid #e8e8e8',
                borderRadius: '12px',
                height: '100%'
              }}
              bodyStyle={{ padding: '20px' }}
            >
              <Title level={3} style={{ color: '#000', margin: '0 0 8px 0' }}>
                1
              </Title>
              <Text style={{ color: 'rgba(0, 0, 0, 0.7)' }}>
                +1 this week
              </Text>
              
              <Divider style={{ borderColor: 'rgba(0, 0, 0, 0.15)', margin: '16px 0' }} />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Title level={5} style={{ color: '#000', margin: 0 }}>
                  Datasets
                </Title>
                <DatabaseOutlined style={{ color: '#000', fontSize: '20px' }} />
              </div>
              
              <div style={{ position: 'absolute', bottom: '12px', right: '12px' }}>
                <Button 
                  type="text" 
                  shape="circle" 
                  icon={<ArrowUpOutlined />} 
                  style={{ color: '#000', borderColor: 'rgba(0, 0, 0, 0.3)' }}
                />
              </div>
            </Card>
          </Col>
          
          {/* ML Models Card */}
          <Col xs={24} sm={24} md={12} lg={6}>
            <Card 
              style={{ 
                background: '#fff',
                border: '1px solid #e8e8e8',
                borderRadius: '12px',
                height: '100%'
              }}
              bodyStyle={{ padding: '20px' }}
            >
              <Title level={3} style={{ color: '#000', margin: '0 0 8px 0' }}>
                2
              </Title>
              <Text style={{ color: 'rgba(0, 0, 0, 0.7)' }}>
                2 new models
              </Text>
              
              <Divider style={{ borderColor: 'rgba(0, 0, 0, 0.15)', margin: '16px 0' }} />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Title level={5} style={{ color: '#000', margin: 0 }}>
                  ML Models
                </Title>
                <RocketOutlined style={{ color: '#000', fontSize: '20px' }} />
              </div>
              
              <div style={{ position: 'absolute', bottom: '12px', right: '12px' }}>
                <Button 
                  type="text" 
                  shape="circle" 
                  icon={<ArrowUpOutlined />} 
                  style={{ color: '#000', borderColor: 'rgba(0, 0, 0, 0.3)' }}
                />
              </div>
            </Card>
          </Col>
          
          {/* Videos Card */}
          <Col xs={24} sm={24} md={12} lg={6}>
            <Card 
              style={{ 
                background: '#fff',
                border: '1px solid #e8e8e8',
                borderRadius: '12px',
                height: '100%'
              }}
              bodyStyle={{ padding: '20px' }}
            >
              <Title level={3} style={{ color: '#000', margin: '0 0 8px 0' }}>
                10
              </Title>
              <Text style={{ color: 'rgba(0, 0, 0, 0.7)' }}>
                Unchanged
              </Text>
              
              <Divider style={{ borderColor: 'rgba(0, 0, 0, 0.15)', margin: '16px 0' }} />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Title level={5} style={{ color: '#000', margin: 0 }}>
                  Videos
                </Title>
                <VideoCameraOutlined style={{ color: '#000', fontSize: '20px' }} />
              </div>
              
              <div style={{ position: 'absolute', bottom: '12px', right: '12px' }}>
                <Button 
                  type="text" 
                  shape="circle" 
                  icon={<ArrowUpOutlined />} 
                  style={{ color: '#000', borderColor: 'rgba(0, 0, 0, 0.3)' }}
                />
              </div>
            </Card>
          </Col>
          
          {/* Predictions Card */}
          <Col xs={24} sm={24} md={12} lg={6}>
            <Card 
              style={{ 
                background: '#fff',
                border: '1px solid #e8e8e8',
                borderRadius: '12px',
                height: '100%'
              }}
              bodyStyle={{ padding: '20px' }}
            >
              <Title level={3} style={{ color: '#000', margin: '0 0 8px 0' }}>
                5
              </Title>
              <Text style={{ color: 'rgba(0, 0, 0, 0.7)' }}>
                Recent predictions
              </Text>
              
              <Divider style={{ borderColor: 'rgba(0, 0, 0, 0.15)', margin: '16px 0' }} />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Title level={5} style={{ color: '#000', margin: 0 }}>
                  Predictions
                </Title>
                <BulbOutlined style={{ color: '#000', fontSize: '20px' }} />
              </div>
              
              <div style={{ position: 'absolute', bottom: '12px', right: '12px' }}>
                <Button 
                  type="text" 
                  shape="circle" 
                  icon={<ArrowUpOutlined />} 
                  style={{ color: '#000', borderColor: 'rgba(0, 0, 0, 0.3)' }}
                />
              </div>
            </Card>
          </Col>
        </Row>
        
        <div style={{ marginTop: '32px' }}>
          <Title level={4} style={{ color: '#000', margin: '0 0 16px 0' }}>
            Your Datasets
          </Title>
        </div>
        
        <Row gutter={[24, 24]}>
          <Col xs={24}>
            <Card 
              hoverable
              style={{ 
                background: '#fff',
                border: '1px solid #e8e8e8',
                borderRadius: '12px'
              }}
              bodyStyle={{ padding: '24px' }}
            >
              <Row gutter={[24, 24]} align="middle">
                <Col xs={24} md={18}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                    <DatabaseOutlined style={{ color: '#000', fontSize: '18px', marginRight: '8px' }} />
                    <Title level={4} style={{ color: '#000', margin: 0 }}>
                      ADITYA MUSIC.csv
                    </Title>
                  </div>
                  
                  <div style={{ marginBottom: '16px', color: 'rgba(0, 0, 0, 0.7)' }}>
                    Uploaded: Unknown date | Videos: 20000
                  </div>
                  
                  <Button
                    type="default"
                    icon={<BarChartOutlined />}
                    style={{ 
                      background: 'rgba(0, 0, 0, 0.05)',
                      borderColor: 'rgba(0, 0, 0, 0.2)',
                      color: '#000'
                    }}
                    onClick={() => router.push('/analytics/1')}
                  >
                    View Analytics
                  </Button>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
        
        <div style={{ marginTop: '32px' }}>
          <Title level={4} style={{ color: '#000', margin: '0 0 16px 0' }}>
            Recent Activity
          </Title>
        </div>
        
        <Row gutter={[24, 24]}>
          <Col xs={24}>
            <Card 
              style={{ 
                background: '#fff',
                border: '1px solid #e8e8e8',
                borderRadius: '12px'
              }}
              bodyStyle={{ padding: '24px' }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '16px' }}>
                  <div style={{ 
                    width: '10px', 
                    height: '10px', 
                    borderRadius: '50%', 
                    background: '#000',
                    marginRight: '12px'
                  }} />
                  <div>
                    <div style={{ color: '#000', fontWeight: 500 }}>Dataset analyzed</div>
                    <div style={{ color: 'rgba(0, 0, 0, 0.7)', fontSize: '14px' }}>Today, 10:23 AM</div>
                  </div>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '16px' }}>
                  <div style={{ 
                    width: '10px', 
                    height: '10px', 
                    borderRadius: '50%', 
                    background: '#000',
                    marginRight: '12px'
                  }} />
                  <div>
                    <div style={{ color: '#000', fontWeight: 500 }}>New analytics report ready</div>
                    <div style={{ color: 'rgba(0, 0, 0, 0.7)', fontSize: '14px' }}>Yesterday, 4:12 PM</div>
                  </div>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <div style={{ 
                    width: '10px', 
                    height: '10px', 
                    borderRadius: '50%', 
                    background: '#000',
                    marginRight: '12px'
                  }} />
                  <div>
                    <div style={{ color: '#000', fontWeight: 500 }}>Video performance improved</div>
                    <div style={{ color: 'rgba(0, 0, 0, 0.7)', fontSize: '14px' }}>2 days ago</div>
                  </div>
                </div>
              </div>
            </Card>
          </Col>
        </Row>
      </Content>
    </Layout>
  );
} 