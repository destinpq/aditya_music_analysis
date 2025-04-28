'use client';

import { useEffect, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Spin, Select, Input, Button, Card, Typography, Row, Col, Statistic, InputNumber, Form, Tooltip } from 'antd';
import { DollarOutlined, LineChartOutlined, RiseOutlined, FallOutlined } from '@ant-design/icons';
import { calculateRevenueProjections } from '../controllers/analysisController';

const { Title, Text } = Typography;
const { Option } = Select;

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface RevenuePredictionChartProps {
  datasetId: number;
  initialViewEstimate?: number;
}

export default function RevenuePredictionChart({ datasetId, initialViewEstimate = 50000 }: RevenuePredictionChartProps) {
  const [loading, setLoading] = useState(false);
  const [viewEstimate, setViewEstimate] = useState(initialViewEstimate);
  const [revenueData, setRevenueData] = useState<any>(null);
  const [selectedCostScenario, setSelectedCostScenario] = useState('Low Cost');
  const [form] = Form.useForm();
  
  const calculateRevenue = useCallback(() => {
    setLoading(true);
    try {
      const data = calculateRevenueProjections(viewEstimate);
      setRevenueData(data);
    } catch (error) {
      console.error('Error calculating revenue projections:', error);
    } finally {
      setLoading(false);
    }
  }, [viewEstimate]);

  useEffect(() => {
    calculateRevenue();
  }, [calculateRevenue]);

  const handleViewsChange = (value: number | null) => {
    if (value !== null) {
      setViewEstimate(value);
    }
  };

  const handleRecalculate = () => {
    calculateRevenue();
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(value);
  };

  const formatCurrency = (value: number): string => {
    // Format as Indian currency (comma after every 2 digits except the first 3)
    return '₹' + value.toFixed(0).replace(/(\d)(?=(\d\d)+\d$)/g, "$1,");
  };

  // Generate fake time series data for the chart
  const generateTimeSeriesData = () => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const data = [];
    let accumulatedRevenue = 0;

    if (revenueData && revenueData.profitScenarios && revenueData.profitScenarios[selectedCostScenario]) {
      // Get revenue for India - already converted to INR in the controller now
      const monthlyRevenue = revenueData.profitScenarios[selectedCostScenario].profitsByCountry['India'].revenue / 12;
      
      for (let i = 0; i < 12; i++) {
        // Add some randomness to monthly figures
        const randomFactor = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
        const monthRevenue = monthlyRevenue * randomFactor; // Already in INR
        accumulatedRevenue += monthRevenue;
        
        data.push({
          month: months[i],
          revenue: monthRevenue,
          cumulativeRevenue: accumulatedRevenue
        });
      }
    }
    
    return data;
  };

  const timeSeriesData = generateTimeSeriesData();

  // Plotly chart configuration
  const plotlyProps = {
    data: [
      {
        x: timeSeriesData.map(d => d.month),
        y: timeSeriesData.map(d => d.revenue),
        type: 'bar',
        name: 'Monthly Revenue',
        marker: {
          color: '#36cfc9'
        }
      },
      {
        x: timeSeriesData.map(d => d.month),
        y: timeSeriesData.map(d => d.cumulativeRevenue),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Cumulative Revenue',
        marker: {
          color: '#1890ff'
        },
        line: {
          color: '#1890ff',
          width: 3
        }
      }
    ],
    layout: {
      title: {
        text: "Projected Revenue Over 12 Months",
        font: {
          size: 20,
          color: "white"
        }
      },
      xaxis: {
        title: {
          text: "Month",
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
          text: "Revenue (INR)",
          font: {
            size: 16,
            color: "white"
          }
        },
        tickfont: {
          color: "white"
        },
        tickprefix: '₹'
      },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      margin: { t: 50, b: 50, l: 80, r: 30 },
      legend: {
        font: {
          color: "white"
        },
        orientation: 'h'
      },
      autosize: true,
    },
    style: { width: '100%', height: 400 },
    config: { 
      responsive: true,
      displayModeBar: false
    }
  };

  // Get available cost scenarios
  const costScenarios = revenueData && revenueData.profitScenarios ? 
    Object.keys(revenueData.profitScenarios) : [];

  return (
    <Card style={{ backgroundColor: 'transparent' }}>
      <Title level={4} style={{ color: 'white' }}>Indian Market Revenue Projections</Title>
      <p style={{ color: 'white' }}>Estimate potential revenue based on projected views and various cost scenarios in the Indian market.</p>
      
      <Form form={form} layout="horizontal" initialValues={{ viewEstimate }}>
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={12}>
            <Form.Item 
              label={<span style={{ color: 'white' }}>Projected Views per Video</span>}
              name="viewEstimate"
            >
              <InputNumber
                style={{ width: '100%' }}
                min={1000}
                max={10000000}
                step={1000}
                formatter={value => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                parser={value => Number(value?.replace(/\$\s?|(,*)/g, '') || 0)}
                value={viewEstimate}
                onChange={handleViewsChange}
              />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label={<span style={{ color: 'white' }}>Cost Scenario</span>}>
              <Select 
                value={selectedCostScenario} 
                onChange={setSelectedCostScenario}
                style={{ width: '100%' }}
              >
                {costScenarios.map(scenario => (
                  <Option key={scenario} value={scenario}>{scenario}</Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
          <Col span={4} style={{ display: 'flex', alignItems: 'flex-end' }}>
            <Button type="primary" onClick={handleRecalculate}>Calculate</Button>
          </Col>
        </Row>
      </Form>
      
      <Spin spinning={loading}>
        {revenueData && revenueData.profitScenarios && revenueData.profitScenarios[selectedCostScenario] && (
          <>
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={8}>
                <Card style={{ backgroundColor: 'transparent' }}>
                  <Statistic
                    title={<span style={{ color: 'white' }}>Monthly Revenue</span>}
                    value={revenueData.profitScenarios[selectedCostScenario].profitsByCountry['India'].revenue / 12}
                    prefix={<span style={{ color: 'white' }}>₹</span>}
                    valueStyle={{ color: 'white' }}
                    formatter={(value: any) => formatCurrency(Number(value)).substring(1)}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card style={{ backgroundColor: 'transparent' }}>
                  <Statistic
                    title={<span style={{ color: 'white' }}>Annual Revenue</span>}
                    value={revenueData.profitScenarios[selectedCostScenario].profitsByCountry['India'].revenue}
                    prefix={<span style={{ color: 'white' }}>₹</span>}
                    valueStyle={{ color: 'white' }}
                    formatter={(value: any) => formatCurrency(Number(value)).substring(1)}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card style={{ backgroundColor: 'transparent' }}>
                  <Statistic
                    title={<span style={{ color: 'white' }}>CPM</span>}
                    value={revenueData.profitScenarios[selectedCostScenario].profitsByCountry['India'].cpm}
                    prefix={<span style={{ color: 'white' }}>₹</span>}
                    precision={2}
                    valueStyle={{ color: 'white' }}
                  />
                </Card>
              </Col>
            </Row>
          
            <div style={{ height: 400, width: '100%' }}>
              <Plot {...plotlyProps} />
            </div>
          
            <Row gutter={16} style={{ marginTop: 24 }}>
              <Col span={24}>
                <Card style={{ backgroundColor: 'transparent' }}>
                  <Title level={5} style={{ color: 'white' }}>Cost-Benefit Analysis (Indian Market)</Title>
                  <Row gutter={[16, 16]}>
                    <Col span={8}>
                      <Text style={{ color: 'white', fontWeight: 'bold' }}>Cost per Video:</Text>
                      <Text style={{ color: 'white', marginLeft: 8 }}>
                        {formatCurrency(revenueData.profitScenarios[selectedCostScenario].costPerVideo)}
                      </Text>
                    </Col>
                    <Col span={8}>
                      <Text style={{ color: 'white', fontWeight: 'bold' }}>Monthly Production Cost:</Text>
                      <Text style={{ color: 'white', marginLeft: 8 }}>
                        {formatCurrency(revenueData.profitScenarios[selectedCostScenario].totalCost / 12)}
                      </Text>
                    </Col>
                    <Col span={8}>
                      <Text style={{ color: 'white', fontWeight: 'bold' }}>Breakeven Point:</Text>
                      <Text style={{ color: 'white', marginLeft: 8 }}>
                        {formatNumber(Math.ceil(revenueData.breakevenPoints['India'] / revenueData.profitScenarios[selectedCostScenario].costPerVideo))} videos
                      </Text>
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </>
        )}
      </Spin>
    </Card>
  );
} 