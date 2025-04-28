'use client';

import React, { useState } from 'react';
import { Layout, Menu, Typography, Card, Row, Col, Button } from 'antd';
import { 
  MenuUnfoldOutlined, 
  MenuFoldOutlined,
  BarChartOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  SettingOutlined,
  HomeOutlined
} from '@ant-design/icons';
import Image from 'next/image';
import Link from 'next/link';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

// Configure Ant Design theme to match your site's colors
const themeConfig = {
  token: {
    colorPrimary: '#4338ca',
    colorBgBase: '#121212',
    colorTextBase: '#ffffff',
    colorBorder: '#333333',
    borderRadius: 8,
  },
};

export default function ResponsiveLayout() {
  const [collapsed, setCollapsed] = useState(false);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  return (
    <Layout 
      className="layout-wrapper" 
      style={{ 
        minHeight: '100vh',
        background: '#121212'
      }}
    >
      <Sider 
        trigger={null} 
        collapsible 
        collapsed={collapsed}
        breakpoint="lg"
        collapsedWidth={0}
        onBreakpoint={(broken) => {
          if (broken) {
            setCollapsed(true);
          }
        }}
        style={{ 
          background: '#121212',
          borderRight: '1px solid #333333',
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          zIndex: 100
        }}
      >
        <div style={{ 
          height: 64, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          padding: '1rem'
        }}>
          {!collapsed && (
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#FFFFFF' }}>
              DestinPQ
            </div>
          )}
        </div>
        <Menu
          mode="inline"
          defaultSelectedKeys={['1']}
          style={{ 
            background: '#121212', 
            color: '#ffffff',
            borderRight: 'none'
          }}
          items={[
            {
              key: '1',
              icon: <HomeOutlined style={{ color: '#FFFFFF' }} />,
              label: <span style={{ color: '#FFFFFF' }}>Dashboard</span>,
            },
            {
              key: '2',
              icon: <BarChartOutlined style={{ color: '#FFFFFF' }} />,
              label: <span style={{ color: '#FFFFFF' }}>Analytics</span>,
            },
            {
              key: '3',
              icon: <DatabaseOutlined style={{ color: '#FFFFFF' }} />,
              label: <span style={{ color: '#FFFFFF' }}>Datasets</span>,
            },
            {
              key: '4',
              icon: <LineChartOutlined style={{ color: '#FFFFFF' }} />,
              label: <span style={{ color: '#FFFFFF' }}>ML Lab</span>,
            },
            {
              key: '5',
              icon: <SettingOutlined style={{ color: '#FFFFFF' }} />,
              label: <span style={{ color: '#FFFFFF' }}>Settings</span>,
            }
          ]}
        />
      </Sider>
      <Layout style={{ 
        marginLeft: collapsed ? 0 : 200,
        transition: 'margin-left 0.2s',
        background: '#121212'
      }}>
        <Header style={{ 
          padding: '0 16px', 
          background: '#121212',
          display: 'flex',
          alignItems: 'center',
          borderBottom: '1px solid #333333',
          position: 'sticky',
          top: 0,
          zIndex: 99,
          width: '100%'
        }}>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined style={{color: 'white'}} /> : <MenuFoldOutlined style={{color: 'white'}} />}
            onClick={toggleCollapsed}
            style={{
              fontSize: '16px',
              width: 64,
              height: 64,
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          />
          <Title level={4} style={{ color: 'white', margin: '0 0 0 16px' }}>
            YouTube Analytics Dashboard
          </Title>
        </Header>
        <Content style={{ 
          margin: '24px 16px', 
          padding: 24, 
          minHeight: 280,
          background: '#121212',
          borderRadius: '8px',
          overflow: 'initial'
        }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} md={6}>
              <Card 
                style={{ 
                  background: 'linear-gradient(135deg, #4338ca 0%, #818cf8 100%)',
                  color: '#ffffff',
                  borderRadius: '8px',
                  border: 'none',
                }}
                bodyStyle={{ padding: '16px' }}
              >
                <Text style={{ color: '#ffffff', fontSize: '16px', display: 'block' }}>
                  Datasets
                </Text>
                <Title level={2} style={{ color: '#ffffff', margin: '8px 0' }}>
                  1
                </Title>
                <Text style={{ color: '#ffffff', fontSize: '14px' }}>
                  +1 this week
                </Text>
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card 
                style={{ 
                  background: 'linear-gradient(135deg, #4338ca 0%, #818cf8 100%)',
                  color: '#ffffff',
                  borderRadius: '8px',
                  border: 'none'
                }}
                bodyStyle={{ padding: '16px' }}
              >
                <Text style={{ color: '#ffffff', fontSize: '16px', display: 'block' }}>
                  ML Models
                </Text>
                <Title level={2} style={{ color: '#ffffff', margin: '8px 0' }}>
                  2
                </Title>
                <Text style={{ color: '#ffffff', fontSize: '14px' }}>
                  2 new models
                </Text>
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card 
                style={{ 
                  background: 'linear-gradient(135deg, #4338ca 0%, #818cf8 100%)',
                  color: '#ffffff',
                  borderRadius: '8px',
                  border: 'none'
                }}
                bodyStyle={{ padding: '16px' }}
              >
                <Text style={{ color: '#ffffff', fontSize: '16px', display: 'block' }}>
                  Videos
                </Text>
                <Title level={2} style={{ color: '#ffffff', margin: '8px 0' }}>
                  10
                </Title>
                <Text style={{ color: '#ffffff', fontSize: '14px' }}>
                  Unchanged
                </Text>
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card 
                style={{ 
                  background: 'linear-gradient(135deg, #4338ca 0%, #818cf8 100%)',
                  color: '#ffffff',
                  borderRadius: '8px',
                  border: 'none'
                }}
                bodyStyle={{ padding: '16px' }}
              >
                <Text style={{ color: '#ffffff', fontSize: '16px', display: 'block' }}>
                  Predictions
                </Text>
                <Title level={2} style={{ color: '#ffffff', margin: '8px 0' }}>
                  5
                </Title>
                <Text style={{ color: '#ffffff', fontSize: '14px' }}>
                  Recent predictions
                </Text>
              </Card>
            </Col>
          </Row>
          
          <Title level={4} style={{ color: 'white', margin: '32px 0 16px 0' }}>
            Your Datasets
          </Title>
          
          <Card 
            style={{ 
              background: 'linear-gradient(135deg, #4338ca 0%, #818cf8 100%)',
              color: '#ffffff',
              borderRadius: '8px',
              border: 'none',
              marginBottom: '16px'
            }}
            bodyStyle={{ padding: '16px' }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
              <div>
                <Title level={5} style={{ color: '#ffffff', margin: '0 0 8px 0' }}>
                  ADITYA MUSIC.csv
                </Title>
                <Text style={{ color: '#ffffff', fontSize: '14px', display: 'block' }}>
                  Uploaded: Unknown date | Videos: 20000
                </Text>
              </div>
              <Button 
                type="default" 
                style={{ 
                  marginTop: '8px',
                  color: '#ffffff', 
                  borderColor: '#ffffff', 
                  background: 'rgba(255,255,255,0.1)',
                  fontWeight: 'bold'
                }}
              >
                View Analytics →
              </Button>
            </div>
          </Card>
          
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Title level={4} style={{ color: 'white', margin: '32px 0 16px 0' }}>
                Recent Activity
              </Title>
              
              <Card 
                style={{ 
                  background: 'linear-gradient(135deg, #4338ca 0%, #818cf8 100%)',
                  color: '#ffffff',
                  borderRadius: '8px',
                  border: 'none'
                }}
                bodyStyle={{ padding: '16px' }}
              >
                <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                  <li style={{ marginBottom: '16px', display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                    <div style={{ width: '24px', height: '24px', borderRadius: '50%', background: 'rgba(255,255,255,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>•</div>
                    <div>
                      <Text style={{ color: '#ffffff', fontSize: '16px', display: 'block', fontWeight: 'bold' }}>
                        Dataset analyzed
                      </Text>
                      <Text style={{ color: '#ffffff', fontSize: '14px', opacity: 0.8 }}>
                        Today, 10:23 AM
                      </Text>
                    </div>
                  </li>
                  <li style={{ marginBottom: '16px', display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                    <div style={{ width: '24px', height: '24px', borderRadius: '50%', background: 'rgba(255,255,255,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>•</div>
                    <div>
                      <Text style={{ color: '#ffffff', fontSize: '16px', display: 'block', fontWeight: 'bold' }}>
                        New analytics report ready
                      </Text>
                      <Text style={{ color: '#ffffff', fontSize: '14px', opacity: 0.8 }}>
                        Yesterday, 4:12 PM
                      </Text>
                    </div>
                  </li>
                  <li style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                    <div style={{ width: '24px', height: '24px', borderRadius: '50%', background: 'rgba(255,255,255,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>•</div>
                    <div>
                      <Text style={{ color: '#ffffff', fontSize: '16px', display: 'block', fontWeight: 'bold' }}>
                        Video performance improved
                      </Text>
                      <Text style={{ color: '#ffffff', fontSize: '14px', opacity: 0.8 }}>
                        2 days ago
                      </Text>
                    </div>
                  </li>
                </ul>
              </Card>
            </Col>
            
            <Col xs={24} md={12}>
              <Title level={4} style={{ color: 'white', margin: '32px 0 16px 0' }}>
                ML Models
              </Title>
              
              <Card 
                style={{ 
                  background: 'linear-gradient(135deg, #4338ca 0%, #818cf8 100%)',
                  color: '#ffffff',
                  borderRadius: '8px',
                  border: 'none',
                  height: 'calc(100% - 48px)'
                }}
                bodyStyle={{ padding: '16px' }}
              >
                <div style={{ 
                  display: 'flex', 
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  textAlign: 'center',
                  padding: '24px 0'
                }}>
                  <Text style={{ color: '#ffffff', fontSize: '18px', display: 'block', marginBottom: '16px' }}>
                    No ML models yet
                  </Text>
                  <Text style={{ color: '#ffffff', fontSize: '14px', opacity: 0.8, marginBottom: '24px' }}>
                    Train your first machine learning model to get predictive insights
                  </Text>
                  <Button 
                    type="default" 
                    style={{ 
                      color: '#ffffff', 
                      borderColor: '#ffffff', 
                      background: 'rgba(255,255,255,0.1)',
                      fontWeight: 'bold'
                    }}
                  >
                    Train Models
                  </Button>
                </div>
              </Card>
            </Col>
          </Row>
        </Content>
      </Layout>
      
      <style jsx global>{`
        .ant-menu-inline .ant-menu-item,
        .ant-menu-inline .ant-menu-submenu-title {
          color: white !important;
        }
        .ant-menu-item-selected {
          background-color: rgba(255, 255, 255, 0.1) !important;
        }
        .ant-menu-item:hover {
          background-color: rgba(255, 255, 255, 0.05) !important;
        }
        .ant-card {
          background-color: #1a1a1a;
        }
        .ant-typography {
          color: white;
        }
      `}</style>
    </Layout>
  );
} 