'use client';

import React, { useState, useEffect } from 'react';
import { 
  Layout, 
  Typography, 
  Tabs, 
  Form, 
  Select, 
  Input, 
  Slider, 
  Checkbox, 
  Button, 
  Card, 
  Space,
  Row,
  Col,
  Upload,
  DatePicker,
  InputNumber,
  Table,
  Alert,
  Progress,
  Spin,
  Result,
  Steps,
  Tooltip,
  Statistic,
  Tag
} from 'antd';
import {
  DatabaseOutlined,
  AimOutlined,
  RocketOutlined,
  SettingOutlined,
  UploadOutlined,
  FileTextOutlined,
  BarChartOutlined,
  FilterOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  QuestionCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import api, { Dataset } from '../services/api';
import ModelTrainer from '../components/ModelTrainer';
import VideoPrediction from '../components/VideoPrediction';
import DatasetProcessor from '../components/DatasetProcessor';
import '../components/ml-components.css';

const { Title, Paragraph, Text } = Typography;
const { Content } = Layout;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;
const { Step } = Steps;

const modelTypes = [
  { id: 'xgboost', name: 'XGBoost' },
  { id: 'lightgbm', name: 'LightGBM' },
  { id: 'catboost', name: 'CatBoost' },
  { id: 'random_forest', name: 'Random Forest' },
  { id: 'linear_model', name: 'Linear Model' },
];

const targetOptions = [
  { id: 'view_count', name: 'View Count' },
  { id: 'like_count', name: 'Like Count' },
  { id: 'comment_count', name: 'Comment Count' },
  { id: 'engagement_rate', name: 'Engagement Rate' },
];

// Training steps for the progress indicator
const trainingSteps = [
  { title: 'Data Preparation', description: 'Loading and preparing data' },
  { title: 'Feature Engineering', description: 'Processing features for training' },
  { title: 'Model Training', description: 'Training the selected models' },
  { title: 'Evaluation', description: 'Evaluating model performance' },
];

// Estimated training times by model type (in seconds)
const estimatedTrainingTimes = {
  'XGBoost': 45,
  'LightGBM': 30,
  'CatBoost': 60,
  'Random Forest': 90,
  'Linear Model': 15,
};

// Update the prediction types and interfaces
// Add this near the top of the file after other type definitions
interface PredictionResult {
  key: string;
  title: string;
  tags: string;
  duration: string;
  predicted_views: string;
  predicted_likes: string;
  predicted_comments: string;
  engagement_rate: string;
  confidence_score: number;
  best_upload_time: string;
  growth_pattern: 'Viral' | 'Steady' | 'Slow' | 'Declining';
}

export default function MLLabPage() {
  const [activeTab, setActiveTab] = useState('1');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string>('view_count');
  const [testSize, setTestSize] = useState<number>(20);
  const [selectedModels, setSelectedModels] = useState<string[]>(['XGBoost', 'LightGBM']);
  const [mlModels, setMlModels] = useState<any[]>([]);
  
  // New state variables for enhanced UI
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentTrainingStep, setCurrentTrainingStep] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState(0);
  const [predictionInProgress, setPredictionInProgress] = useState(false);
  const [processingInProgress, setProcessingInProgress] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  
  // Add form state for video details
  const [videoTitle, setVideoTitle] = useState('');
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [uploadDay, setUploadDay] = useState<string>('');
  const [category, setCategory] = useState<string>('');
  const [videoTags, setVideoTags] = useState('');
  
  // Update the prediction state to use the new interface
  const [recentPredictions, setRecentPredictions] = useState<PredictionResult[]>([
    {
      key: '1',
      title: 'Summer Vacation - Beach Highlights',
      tags: 'summer, beach, vacation',
      duration: '12:45',
      predicted_views: '125K - 250K',
      predicted_likes: '8.5K - 15K',
      predicted_comments: '450 - 850',
      engagement_rate: '6.8%',
      confidence_score: 87,
      best_upload_time: 'Saturday, 10AM-12PM',
      growth_pattern: 'Steady'
    },
    {
      key: '2',
      title: 'How to Make Authentic Italian Pizza',
      tags: 'cooking, italian, recipe',
      duration: '08:30',
      predicted_views: '45K - 80K',
      predicted_likes: '3.2K - 5.5K',
      predicted_comments: '180 - 320',
      engagement_rate: '7.2%',
      confidence_score: 92,
      best_upload_time: 'Sunday, 4PM-6PM',
      growth_pattern: 'Slow'
    },
  ]);
  
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // Fetch datasets
        const datasetsResponse = await api.getDatasets();
        setDatasets(datasetsResponse);
        
        // Fetch ML models info
        try {
          const mlModelsResponse = await api.mlGetModelsInfo();
          setMlModels(mlModelsResponse);
        } catch (err) {
          console.error('Error fetching ML models:', err);
          // Non-critical error, don't show in UI
        }
      } catch (err) {
        setError(`Error loading data: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, []);

  // Update training progress at intervals
  useEffect(() => {
    let progressTimer: NodeJS.Timeout | null = null;
    let stepTimer: NodeJS.Timeout | null = null;
    
    if (trainingInProgress && trainingProgress < 100) {
      progressTimer = setInterval(() => {
        setTrainingProgress(prev => {
          const newProgress = prev + 1;
          // Update estimated time remaining
          const totalTime = calculateTotalTrainingTime();
          const elapsed = (newProgress / 100) * totalTime;
          setEstimatedTimeRemaining(Math.round(totalTime - elapsed));
          return newProgress;
        });
      }, 1000);
      
      // Update training steps
      stepTimer = setInterval(() => {
        setCurrentTrainingStep(prev => {
          if (trainingProgress > 25 && prev === 0) return 1;
          if (trainingProgress > 50 && prev === 1) return 2;
          if (trainingProgress > 75 && prev === 2) return 3;
          return prev;
        });
      }, 2000);
    }
    
    return () => {
      if (progressTimer) clearInterval(progressTimer);
      if (stepTimer) clearInterval(stepTimer);
    };
  }, [trainingInProgress, trainingProgress]);
  
  // Calculate total training time based on selected models
  const calculateTotalTrainingTime = () => {
    return selectedModels.reduce((total, model) => {
      return total + (estimatedTrainingTimes[model as keyof typeof estimatedTrainingTimes] || 30);
    }, 0);
  };
  
  const handleModelToggle = (modelName: string) => {
    if (selectedModels.includes(modelName)) {
      setSelectedModels(selectedModels.filter(name => name !== modelName));
    } else {
      setSelectedModels([...selectedModels, modelName]);
    }
  };
  
  const handleTrainModel = () => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }
    
    if (selectedModels.length === 0) {
      setError('Please select at least one model type');
      return;
    }
    
    setError(null);
    setTrainingInProgress(true);
    setTrainingProgress(0);
    setCurrentTrainingStep(0);
    setTrainingComplete(false);
    setEstimatedTimeRemaining(calculateTotalTrainingTime());
    
    // Simulate training process
    setTimeout(() => {
      setTrainingInProgress(false);
      setTrainingProgress(100);
      setCurrentTrainingStep(3);
      setTrainingComplete(true);
      
      // Refresh models list after training
      handleTrainingComplete({ success: true });
    }, calculateTotalTrainingTime() * 1000);
  };
  
  const handleHyperparameterTuning = () => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }
    
    if (selectedModels.length === 0) {
      setError('Please select at least one model type');
      return;
    }
    
    setError(null);
    setTrainingInProgress(true);
    setTrainingProgress(0);
    setCurrentTrainingStep(0);
    setTrainingComplete(false);
    // Hyperparameter tuning takes longer
    setEstimatedTimeRemaining(calculateTotalTrainingTime() * 2);
    
    // Simulate hyperparameter tuning process
    setTimeout(() => {
      setTrainingInProgress(false);
      setTrainingProgress(100);
      setCurrentTrainingStep(3);
      setTrainingComplete(true);
      
      // Refresh models list after training
      handleTrainingComplete({ success: true });
    }, calculateTotalTrainingTime() * 2000);
  };

  const handleMakePrediction = () => {
    // Validate required fields
    if (!videoTitle || !videoDuration || !uploadDay || !category) {
      setError('Please fill in all required fields');
      return;
    }
    
    setError(null);
    setPredictionInProgress(true);
    
    // Simulate prediction process
    setTimeout(() => {
      // Generate a predicted view range based on input factors
      // This is just for demonstration - in a real app this would come from the ML model
      const durationFactor = videoDuration ? Math.min(videoDuration / 5, 2) : 1;
      const categoryFactor = category === 'music' ? 1.5 : 
                           category === 'entertainment' ? 1.3 : 
                           category === 'education' ? 0.8 : 1;
      const dayFactor = uploadDay === 'saturday' || uploadDay === 'sunday' ? 1.4 : 1;
      
      const baseViews = Math.floor(Math.random() * 150) + 50;
      const minViews = Math.floor(baseViews * durationFactor * categoryFactor * dayFactor);
      const maxViews = Math.floor(minViews * (1 + Math.random()));
      
      // Calculate engagement metrics based on the predicted views
      const likeRate = (Math.random() * 0.03) + 0.04; // 4-7% of views convert to likes
      const commentRate = (Math.random() * 0.004) + 0.002; // 0.2-0.6% of views leave comments
      
      const minLikes = Math.floor(minViews * likeRate);
      const maxLikes = Math.floor(maxViews * likeRate);
      
      const minComments = Math.floor(minViews * commentRate);
      const maxComments = Math.floor(maxViews * commentRate);
      
      const engagementRate = ((minLikes + maxLikes) / 2 / ((minViews + maxViews) / 2) * 100).toFixed(1);
      
      // Determine best upload time based on selected day
      let bestTime = '';
      switch(uploadDay) {
        case 'monday':
        case 'tuesday':
        case 'wednesday':
        case 'thursday':
          bestTime = `${uploadDay.charAt(0).toUpperCase() + uploadDay.slice(1)}, 6PM-8PM`;
          break;
        case 'friday':
          bestTime = 'Friday, 3PM-5PM';
          break;
        case 'saturday':
          bestTime = 'Saturday, 10AM-12PM';
          break;
        case 'sunday':
          bestTime = 'Sunday, 2PM-4PM';
          break;
        default:
          bestTime = 'Weekday, 6PM-8PM';
      }
      
      // Determine growth pattern based on category and duration
      let growthPattern: 'Viral' | 'Steady' | 'Slow' | 'Declining' = 'Steady';
      if (category === 'music' && videoDuration && videoDuration < 4) {
        growthPattern = 'Viral';
      } else if (category === 'education' && videoDuration && videoDuration > 10) {
        growthPattern = 'Slow';
      } else if (Math.random() > 0.8) {
        growthPattern = 'Declining';
      }
      
      // Format numbers for display
      const viewFormat = (views: number) => {
        if (views >= 1000000) {
          return `${(views / 1000000).toFixed(1)}M`;
        } else if (views >= 1000) {
          return `${Math.floor(views / 1000)}K`;
        }
        return views.toString();
      };
      
      const likeFormat = (likes: number) => {
        if (likes >= 1000) {
          return `${(likes / 1000).toFixed(1)}K`;
        }
        return likes.toString();
      };
      
      // Calculate confidence score (70-95%)
      const confidenceScore = Math.floor(Math.random() * 25) + 70;
      
      // Create new prediction result
      const newPrediction: PredictionResult = {
        key: `prediction-${Date.now()}`,
        title: videoTitle,
        tags: videoTags || 'none',
        duration: videoDuration ? `${videoDuration}:00` : '0:00',
        predicted_views: `${viewFormat(minViews)} - ${viewFormat(maxViews)}`,
        predicted_likes: `${likeFormat(minLikes)} - ${likeFormat(maxLikes)}`,
        predicted_comments: `${minComments} - ${maxComments}`,
        engagement_rate: `${engagementRate}%`,
        confidence_score: confidenceScore,
        best_upload_time: bestTime,
        growth_pattern: growthPattern
      };
      
      // Add to recent predictions
      setRecentPredictions(prev => [newPrediction, ...prev.slice(0, 9)]);
      
      // Reset loading state
      setPredictionInProgress(false);
    }, 3000);
  };

  const handleProcessDataset = () => {
    setProcessingInProgress(true);
    setProcessingProgress(0);
    
    // Simulate processing with progress updates
    const processingTimer = setInterval(() => {
      setProcessingProgress(prev => {
        const newProgress = prev + 5;
        if (newProgress >= 100) {
          clearInterval(processingTimer);
          setTimeout(() => {
            setProcessingInProgress(false);
          }, 500);
          return 100;
        }
        return newProgress;
      });
    }, 500);
  };
  
  const handleTrainingComplete = (result: any) => {
    // Refresh models list after training
    api.mlGetModelsInfo()
      .then(response => {
        setMlModels(response);
      })
      .catch(err => console.error('Error fetching ML models after training:', err));
  };

  const resetTrainingState = () => {
    setTrainingInProgress(false);
    setTrainingProgress(0);
    setCurrentTrainingStep(0);
    setTrainingComplete(false);
  };

  // Sample processing columns
  const processingColumns = [
    {
      title: 'Column Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Data Type',
      dataIndex: 'type',
      key: 'type',
    },
    {
      title: 'Include',
      dataIndex: 'include',
      key: 'include',
      render: (_: any, record: any) => (
        <Checkbox defaultChecked />
      ),
    },
    {
      title: 'Missing Values Action',
      dataIndex: 'missing',
      key: 'missing',
      render: () => (
        <Select
          defaultValue="impute_mean"
          style={{ width: '100%', backgroundColor: '#fff' }}
          className="white-background-select"
        >
          <Option value="impute_mean">Impute (Mean)</Option>
          <Option value="impute_median">Impute (Median)</Option>
          <Option value="drop">Drop Rows</Option>
          <Option value="zero">Replace with Zero</Option>
        </Select>
      ),
    },
  ];

  // Sample data columns
  const sampleDataColumns = [
    { name: 'view_count', type: 'Numeric', key: '1' },
    { name: 'likes', type: 'Numeric', key: '2' },
    { name: 'comments', type: 'Numeric', key: '3' },
    { name: 'title', type: 'Text', key: '4' },
    { name: 'uploaded_date', type: 'Date', key: '5' },
  ];
  
  return (
    <Layout style={{ background: '#fff', minHeight: '100vh' }}>
      <Content style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <Title level={1} style={{ fontSize: '32px', color: '#000', margin: '0 0 16px 0' }}>
            ML Lab
          </Title>
          <Paragraph style={{ fontSize: '16px', color: '#000', maxWidth: '800px', margin: '0 auto' }}>
            Train, test, and deploy machine learning models for YouTube video analytics
          </Paragraph>
        </div>
        
        <Tabs 
          activeKey={activeTab} 
          onChange={(key) => {
            setActiveTab(key);
            // Reset training states when changing tabs
            resetTrainingState();
            setPredictionInProgress(false);
            setProcessingInProgress(false);
          }}
          style={{ marginBottom: '32px' }}
          items={[
            { key: '1', label: <span style={{ fontSize: '16px' }}>Train Models</span> },
            { key: '2', label: <span style={{ fontSize: '16px' }}>Make Predictions</span> },
            { key: '3', label: <span style={{ fontSize: '16px' }}>Process Datasets</span> }
          ]}
        />
        
        {/* Show error alert if there's an error */}
        {error && (
          <Alert
            message="Error"
            description={error}
            type="error"
            closable
            onClose={() => setError(null)}
            style={{ marginBottom: '24px' }}
          />
        )}
        
        {activeTab === '1' && (
          <Card 
            style={{ 
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              border: '1px solid #e8e8e8',
            }}
          >
            {trainingComplete ? (
              <Result
                status="success"
                title="Model Training Complete!"
                subTitle="Your models have been successfully trained and are now ready for making predictions."
                extra={[
                  <Button 
                    type="primary" 
                    key="make-predictions" 
                    onClick={() => setActiveTab('2')}
                    style={{ 
                      backgroundColor: '#1890ff',
                      borderColor: '#1890ff',
                    }}
                  >
                    Make Predictions
                  </Button>,
                  <Button 
                    key="train-more" 
                    onClick={resetTrainingState}
                  >
                    Train More Models
                  </Button>,
                ]}
              />
            ) : trainingInProgress ? (
              <div style={{ textAlign: 'center', padding: '32px 0' }}>
                <Title level={3} style={{ fontSize: '22px', color: '#000', marginBottom: '24px' }}>
                  Training Models
                </Title>
                
                <Progress 
                  type="circle" 
                  percent={trainingProgress} 
                  status={trainingProgress < 100 ? "active" : "success"}
                  style={{ marginBottom: '32px' }}
                />
                
                <div style={{ marginBottom: '24px' }}>
                  <Text style={{ fontSize: '16px', display: 'block', marginBottom: '8px' }}>
                    <ClockCircleOutlined style={{ marginRight: '8px' }} />
                    Estimated time remaining: {estimatedTimeRemaining} seconds
                  </Text>
                  <Text style={{ fontSize: '14px', color: '#666' }}>
                    Training {selectedModels.join(', ')} models for {selectedTarget} prediction
                  </Text>
                </div>
                
                <Steps 
                  current={currentTrainingStep}
                  style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'left' }}
                >
                  {trainingSteps.map(step => (
                    <Step 
                      key={step.title} 
                      title={step.title} 
                      description={step.description} 
                    />
                  ))}
                </Steps>
                
                <Alert
                  message="Training In Progress"
                  description="Model training takes time. You can continue working in other tabs while the training completes."
                  type="info"
                  showIcon
                  style={{ maxWidth: '800px', margin: '32px auto 0', textAlign: 'left' }}
                />
              </div>
            ) : (
              <Form layout="vertical">
                <Title level={3} style={{ fontSize: '22px', color: '#000', marginBottom: '24px' }}>
                  Train ML Models
                </Title>
                
                <Alert
                  message="Important"
                  description="Training ML models is required before making predictions. Select your dataset, target metric, and models to begin."
                  type="info"
                  showIcon
                  style={{ marginBottom: '24px' }}
                />
                
                <Form.Item 
                  label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Select Dataset</Text>} 
                  style={{ marginBottom: '24px' }}
                  required
                >
                  <Select
                    placeholder="Select a dataset..."
                    style={{ width: '100%', fontSize: '16px' }}
                    dropdownStyle={{ backgroundColor: '#fff' }}
                    className="white-background-select"
                    onChange={(value) => setSelectedDataset(value)}
                    value={selectedDataset}
                  >
                    {datasets.map(dataset => (
                      <Option key={dataset.id} value={dataset.id}>{dataset.filename || `Dataset #${dataset.id}`}</Option>
                    ))}
                  </Select>
                </Form.Item>
                
                <Form.Item 
                  label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Target Metric</Text>}
                  style={{ marginBottom: '24px' }}
                  required
                >
                  <Select
                    placeholder="View Count"
                    style={{ width: '100%', fontSize: '16px', backgroundColor: '#fff' }}
                    dropdownStyle={{ backgroundColor: '#fff' }}
                    className="white-background-select"
                    onChange={(value) => setSelectedTarget(value)}
                    value={selectedTarget}
                  >
                    {targetOptions.map(metric => (
                      <Option key={metric.id} value={metric.id}>{metric.name}</Option>
                    ))}
                  </Select>
                </Form.Item>
                
                <Form.Item 
                  label={
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <Text style={{ fontSize: '16px', fontWeight: 500, marginRight: '8px' }}>Test Set Size</Text>
                      <Tooltip title="The percentage of data used for testing your models. Higher values provide better evaluation but less training data.">
                        <QuestionCircleOutlined />
                      </Tooltip>
                    </div>
                  }
                  style={{ marginBottom: '24px' }}
                >
                  <Slider
                    min={10}
                    max={50}
                    value={testSize}
                    onChange={setTestSize}
                    tooltip={{ formatter: value => `${value}%` }}
                    style={{ backgroundColor: '#fff' }}
                  />
                </Form.Item>
                
                <Form.Item 
                  label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Select Model Types</Text>}
                  style={{ marginBottom: '32px' }}
                  required
                >
                  <Row gutter={[16, 16]}>
                    {modelTypes.map(model => (
                      <Col xs={24} sm={12} md={8} lg={6} key={model.id}>
                        <Card
                          style={{ 
                            border: '1px solid #e8e8e8',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            backgroundColor: selectedModels.includes(model.name) ? '#f0f7ff' : '#fff'
                          }}
                          bodyStyle={{ padding: '12px', display: 'flex', alignItems: 'center' }}
                          onClick={() => handleModelToggle(model.name)}
                        >
                          <Checkbox 
                            checked={selectedModels.includes(model.name)} 
                            style={{ marginRight: '8px' }} 
                          />
                          <span style={{ fontSize: '16px' }}>{model.name}</span>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                </Form.Item>
                
                <Space size={16}>
                  <Button 
                    type="primary" 
                    size="large"
                    onClick={handleTrainModel}
                    style={{ 
                      backgroundColor: '#1890ff',
                      borderColor: '#1890ff',
                      fontSize: '16px',
                      height: '44px',
                      paddingLeft: '24px',
                      paddingRight: '24px'
                    }}
                  >
                    Train Models
                  </Button>
                  
                  <Button 
                    size="large"
                    onClick={handleHyperparameterTuning}
                    style={{ 
                      fontSize: '16px',
                      height: '44px',
                      paddingLeft: '16px',
                      paddingRight: '16px',
                      backgroundColor: '#fff'
                    }}
                  >
                    Run Hyperparameter Tuning
                  </Button>
                  
                  <Tooltip title="Hyperparameter tuning takes longer but can improve model performance">
                    <QuestionCircleOutlined style={{ fontSize: '16px', color: '#666' }} />
                  </Tooltip>
                </Space>
                
                <div style={{ marginTop: '32px' }}>
                  <Alert
                    message="Training Time Estimates"
                    description={
                      <div>
                        <p>Approximate training times:</p>
                        <ul>
                          <li>XGBoost: 30-60 seconds</li>
                          <li>LightGBM: 20-40 seconds</li>
                          <li>CatBoost: 45-90 seconds</li>
                          <li>Random Forest: 60-120 seconds</li>
                          <li>Linear Model: 10-20 seconds</li>
                        </ul>
                        <p>Times vary based on dataset size and complexity.</p>
                      </div>
                    }
                    type="info"
                    showIcon
                  />
                </div>
              </Form>
            )}
          </Card>
        )}

        {activeTab === '2' && (
          <Card 
            style={{ 
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              border: '1px solid #e8e8e8',
            }}
          >
            <Form layout="vertical">
              <Title level={3} style={{ fontSize: '22px', color: '#000', marginBottom: '24px' }}>
                Make Predictions
              </Title>
              
              {mlModels.length === 0 && !predictionInProgress && (
                <Alert
                  message="No Trained Models Available"
                  description="You need to train models before making predictions. Please go to the 'Train Models' tab first."
                  type="warning"
                  showIcon
                  style={{ marginBottom: '24px' }}
                  action={
                    <Button type="primary" size="small" onClick={() => setActiveTab('1')}>
                      Train Models
                    </Button>
                  }
                />
              )}
              
              {predictionInProgress ? (
                <div style={{ textAlign: 'center', padding: '32px' }}>
                  <Spin 
                    indicator={<LoadingOutlined style={{ fontSize: 48 }} spin />} 
                    tip="Generating predictions... This will take a few seconds." 
                    size="large"
                  />
                  <Text style={{ display: 'block', marginTop: '16px', color: '#666' }}>
                    Analyzing video characteristics and applying ML models
                  </Text>
                </div>
              ) : (
                <>
                  <Row gutter={24}>
                    <Col xs={24} md={12}>
                      <Form.Item 
                        label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Select Model</Text>} 
                        style={{ marginBottom: '24px' }}
                        required
                      >
                        <Select
                          placeholder="Select a trained model..."
                          style={{ width: '100%', fontSize: '16px', backgroundColor: '#fff' }}
                          dropdownStyle={{ backgroundColor: '#fff' }}
                          className="white-background-select"
                        >
                          <Option value="1">ADITYA MUSIC.csv - View Count Predictor</Option>
                          <Option value="2">Backup Channel - Engagement Rate Predictor</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    
                    <Col xs={24} md={12}>
                      <Form.Item 
                        label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Input Method</Text>}
                        style={{ marginBottom: '24px' }}
                        tooltip="Choose 'Manual Entry' for single prediction or 'Upload File' for batch predictions"
                      >
                        <Select
                          defaultValue="manual"
                          style={{ width: '100%', fontSize: '16px', backgroundColor: '#fff' }}
                          dropdownStyle={{ backgroundColor: '#fff' }}
                          className="white-background-select"
                        >
                          <Option value="manual">Manual Entry</Option>
                          <Option value="file">Upload File</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>
                  
                  <Form.Item 
                    label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Video Details</Text>}
                    style={{ marginBottom: '24px' }}
                  >
                    <Card
                      style={{ 
                        backgroundColor: '#fff',
                        border: '1px solid #e8e8e8',
                      }}
                    >
                      <Form.Item 
                        label={<Text style={{ fontSize: '16px' }}>Video Title</Text>}
                        style={{ marginBottom: '16px' }}
                        required
                      >
                        <Input 
                          placeholder="Enter video title" 
                          style={{ fontSize: '16px', backgroundColor: '#fff' }}
                          value={videoTitle}
                          onChange={e => setVideoTitle(e.target.value)}
                        />
                      </Form.Item>
                      
                      <Row gutter={16}>
                        <Col xs={24} md={8}>
                          <Form.Item 
                            label={<Text style={{ fontSize: '16px' }}>Duration (minutes)</Text>}
                            required
                          >
                            <InputNumber
                              min={0}
                              placeholder="Duration"
                              style={{ width: '100%', fontSize: '16px', backgroundColor: '#fff' }}
                              value={videoDuration}
                              onChange={value => setVideoDuration(value)}
                            />
                          </Form.Item>
                        </Col>
                        
                        <Col xs={24} md={8}>
                          <Form.Item 
                            label={<Text style={{ fontSize: '16px' }}>Upload Day</Text>}
                            required
                          >
                            <Select
                              placeholder="Select day"
                              style={{ width: '100%', fontSize: '16px', backgroundColor: '#fff' }}
                              dropdownStyle={{ backgroundColor: '#fff' }}
                              className="white-background-select"
                              value={uploadDay}
                              onChange={value => setUploadDay(value)}
                            >
                              <Option value="monday">Monday</Option>
                              <Option value="tuesday">Tuesday</Option>
                              <Option value="wednesday">Wednesday</Option>
                              <Option value="thursday">Thursday</Option>
                              <Option value="friday">Friday</Option>
                              <Option value="saturday">Saturday</Option>
                              <Option value="sunday">Sunday</Option>
                            </Select>
                          </Form.Item>
                        </Col>
                        
                        <Col xs={24} md={8}>
                          <Form.Item 
                            label={<Text style={{ fontSize: '16px' }}>Category</Text>}
                            required
                          >
                            <Select
                              placeholder="Select category"
                              style={{ width: '100%', fontSize: '16px', backgroundColor: '#fff' }}
                              dropdownStyle={{ backgroundColor: '#fff' }}
                              className="white-background-select"
                              value={category}
                              onChange={value => setCategory(value)}
                            >
                              <Option value="music">Music</Option>
                              <Option value="entertainment">Entertainment</Option>
                              <Option value="education">Education</Option>
                              <Option value="howto">How-to & Style</Option>
                            </Select>
                          </Form.Item>
                        </Col>
                      </Row>
                      
                      <Form.Item 
                        label={<Text style={{ fontSize: '16px' }}>Tags (comma separated)</Text>}
                      >
                        <Input 
                          placeholder="music, latest, popular" 
                          style={{ fontSize: '16px', backgroundColor: '#fff' }}
                          value={videoTags}
                          onChange={e => setVideoTags(e.target.value)}
                        />
                      </Form.Item>
                    </Card>
                  </Form.Item>
                  
                  <Space size={16}>
                    <Button 
                      type="primary" 
                      size="large"
                      onClick={handleMakePrediction}
                      style={{ 
                        backgroundColor: '#1890ff',
                        borderColor: '#1890ff',
                        fontSize: '16px',
                        height: '44px',
                        paddingLeft: '24px',
                        paddingRight: '24px'
                      }}
                    >
                      Make Prediction
                    </Button>
                    
                    <Button 
                      size="large"
                      style={{ 
                        fontSize: '16px',
                        height: '44px',
                        paddingLeft: '16px',
                        paddingRight: '16px',
                        backgroundColor: '#fff'
                      }}
                    >
                      Batch Predict
                    </Button>
                    
                    <Tooltip title="Batch prediction lets you predict multiple videos at once by uploading a CSV file">
                      <QuestionCircleOutlined style={{ fontSize: '16px', color: '#666' }} />
                    </Tooltip>
                  </Space>
                </>
              )}
              
              <div style={{ marginTop: '32px' }}>
                <Title level={4} style={{ fontSize: '18px', color: '#000', marginBottom: '16px' }}>
                  Prediction Results
                </Title>
                
                <Alert
                  message="Prediction Information"
                  description={
                    <div>
                      <p><strong>Metrics Explanation:</strong></p>
                      <ul style={{ marginBottom: '8px' }}>
                        <li><strong>Views:</strong> Estimated range for the first 30 days after upload (K = thousand, M = million)</li>
                        <li><strong>Likes/Comments:</strong> Estimated engagement for the first 30 days</li>
                        <li><strong>Engagement Rate:</strong> Average percentage of viewers who interact with the video</li>
                        <li><strong>Confidence Score:</strong> Our model's confidence in this prediction (higher is better)</li>
                        <li><strong>Best Upload Time:</strong> Recommended day and time to maximize initial views</li>
                        <li><strong>Growth Pattern:</strong> Expected view accumulation pattern over time</li>
                      </ul>
                      <p><strong>Growth Pattern Types:</strong></p>
                      <ul>
                        <li><strong>Viral:</strong> Rapid initial growth with high sharing potential</li>
                        <li><strong>Steady:</strong> Consistent views over time with moderate sharing</li>
                        <li><strong>Slow:</strong> Gradual accumulation of views, typical for educational content</li>
                        <li><strong>Declining:</strong> Strong initial views followed by rapid dropoff</li>
                      </ul>
                    </div>
                  }
                  type="info"
                  showIcon
                  style={{ marginBottom: '24px' }}
                />
                
                {recentPredictions.length > 0 && (
                  <Card title="Latest Prediction Details" style={{ marginBottom: '24px' }}>
                    <Row gutter={[16, 16]}>
                      <Col xs={24} md={12}>
                        <Statistic 
                          title="Predicted 30-Day Views" 
                          value={recentPredictions[0].predicted_views} 
                          style={{ marginBottom: '16px' }}
                        />
                        <Statistic 
                          title="Predicted Likes" 
                          value={recentPredictions[0].predicted_likes}
                          style={{ marginBottom: '16px' }}
                        />
                        <Statistic 
                          title="Predicted Comments" 
                          value={recentPredictions[0].predicted_comments}
                        />
                      </Col>
                      <Col xs={24} md={12}>
                        <Statistic 
                          title="Engagement Rate" 
                          value={recentPredictions[0].engagement_rate}
                          style={{ marginBottom: '16px' }}
                        />
                        <Statistic 
                          title="Confidence Score" 
                          value={recentPredictions[0].confidence_score}
                          suffix="%"
                          style={{ marginBottom: '16px' }}
                        />
                        <Statistic 
                          title="Best Upload Time" 
                          value={recentPredictions[0].best_upload_time}
                        />
                      </Col>
                      <Col span={24}>
                        <div style={{ marginTop: '16px' }}>
                          <Text strong>Growth Pattern: </Text>
                          <Tag color={
                            recentPredictions[0].growth_pattern === 'Viral' ? 'green' :
                            recentPredictions[0].growth_pattern === 'Steady' ? 'blue' :
                            recentPredictions[0].growth_pattern === 'Slow' ? 'orange' : 'red'
                          }>
                            {recentPredictions[0].growth_pattern}
                          </Tag>
                        </div>
                        <div style={{ marginTop: '16px' }}>
                          <Text strong>Video: </Text>
                          <Text>{recentPredictions[0].title}</Text>
                          <br />
                          <Text strong>Tags: </Text>
                          <Text>{recentPredictions[0].tags}</Text>
                          <br />
                          <Text strong>Duration: </Text>
                          <Text>{recentPredictions[0].duration}</Text>
                        </div>
                      </Col>
                    </Row>
                  </Card>
                )}
                
                <Title level={5} style={{ fontSize: '16px', color: '#000', marginBottom: '16px', marginTop: '24px' }}>
                  Recent Predictions History
                </Title>
                
                <Table 
                  dataSource={recentPredictions} 
                  columns={[
                    {
                      title: 'Video Title',
                      dataIndex: 'title',
                      key: 'title',
                    },
                    {
                      title: 'Duration',
                      dataIndex: 'duration',
                      key: 'duration',
                    },
                    {
                      title: 'Views (30-day)',
                      dataIndex: 'predicted_views',
                      key: 'predicted_views',
                      render: (text) => <span style={{ fontWeight: 'bold', color: '#1890ff' }}>{text}</span>
                    },
                    {
                      title: 'Engagement',
                      dataIndex: 'engagement_rate',
                      key: 'engagement_rate',
                    },
                    {
                      title: 'Growth Pattern',
                      dataIndex: 'growth_pattern',
                      key: 'growth_pattern',
                      render: (text) => (
                        <Tag color={
                          text === 'Viral' ? 'green' :
                          text === 'Steady' ? 'blue' :
                          text === 'Slow' ? 'orange' : 'red'
                        }>
                          {text}
                        </Tag>
                      )
                    },
                  ]}
                  style={{ backgroundColor: '#fff' }}
                  loading={predictionInProgress}
                />
              </div>
            </Form>
          </Card>
        )}

        {activeTab === '3' && (
          <Card 
            style={{ 
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              border: '1px solid #e8e8e8',
            }}
          >
            <Form layout="vertical">
              <Title level={3} style={{ fontSize: '22px', color: '#000', marginBottom: '24px' }}>
                Process Datasets
              </Title>
              
              {processingInProgress ? (
                <div style={{ textAlign: 'center', padding: '32px 0' }}>
                  <Progress 
                    percent={processingProgress} 
                    status="active" 
                    style={{ marginBottom: '24px' }}
                  />
                  
                  <div style={{ marginBottom: '24px' }}>
                    <Text style={{ fontSize: '16px' }}>
                      Processing dataset... 
                      {processingProgress < 25 && "Loading data..."}
                      {processingProgress >= 25 && processingProgress < 50 && "Cleaning and preprocessing..."}
                      {processingProgress >= 50 && processingProgress < 75 && "Applying transformations..."}
                      {processingProgress >= 75 && "Finalizing and saving..."}
                    </Text>
                  </div>
                  
                  <Alert
                    message="Processing In Progress"
                    description="This may take a few moments depending on the dataset size. The processed dataset will be available for model training when complete."
                    type="info"
                    showIcon
                  />
                </div>
              ) : (
                <>
                  <Alert
                    message="About Dataset Processing"
                    description="Processing your dataset can improve model performance by handling missing values, removing duplicates, and normalizing data."
                    type="info"
                    showIcon
                    style={{ marginBottom: '24px' }}
                  />
                  
                  <Row gutter={24}>
                    <Col xs={24} md={12}>
                      <Form.Item 
                        label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Source Dataset</Text>} 
                        style={{ marginBottom: '24px' }}
                        required
                      >
                        <Select
                          placeholder="Select a dataset..."
                          style={{ width: '100%', fontSize: '16px', backgroundColor: '#fff' }}
                          dropdownStyle={{ backgroundColor: '#fff' }}
                          className="white-background-select"
                        >
                          {datasets.map(dataset => (
                            <Option key={dataset.id} value={dataset.id}>{dataset.filename || `Dataset #${dataset.id}`}</Option>
                          ))}
                        </Select>
                      </Form.Item>
                    </Col>
                    
                    <Col xs={24} md={12}>
                      <Form.Item 
                        label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>New Dataset Name</Text>}
                        style={{ marginBottom: '24px' }}
                        required
                      >
                        <Input 
                          placeholder="Enter name for processed dataset" 
                          style={{ fontSize: '16px', backgroundColor: '#fff' }}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                  
                  <Form.Item 
                    label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Date Range Filter</Text>}
                    style={{ marginBottom: '24px' }}
                    tooltip="Optional: Filter data to a specific date range"
                  >
                    <RangePicker
                      style={{ width: '100%', fontSize: '16px', backgroundColor: '#fff' }}
                    />
                  </Form.Item>
                  
                  <Form.Item 
                    label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Column Configuration</Text>}
                    style={{ marginBottom: '32px' }}
                  >
                    <Table 
                      dataSource={sampleDataColumns} 
                      columns={processingColumns}
                      pagination={false}
                      style={{ backgroundColor: '#fff' }}
                    />
                  </Form.Item>
                  
                  <Form.Item 
                    label={<Text style={{ fontSize: '16px', fontWeight: 500 }}>Processing Options</Text>}
                    style={{ marginBottom: '32px' }}
                  >
                    <Card
                      style={{ 
                        backgroundColor: '#fff',
                        border: '1px solid #e8e8e8',
                      }}
                    >
                      <Row gutter={16}>
                        <Col xs={24} md={8}>
                          <Form.Item 
                            valuePropName="checked"
                          >
                            <Checkbox style={{ fontSize: '16px' }}>
                              Remove Duplicates
                            </Checkbox>
                          </Form.Item>
                        </Col>
                        
                        <Col xs={24} md={8}>
                          <Form.Item 
                            valuePropName="checked"
                          >
                            <Checkbox style={{ fontSize: '16px' }}>
                              Normalize Numeric Values
                            </Checkbox>
                          </Form.Item>
                        </Col>
                        
                        <Col xs={24} md={8}>
                          <Form.Item 
                            valuePropName="checked"
                          >
                            <Checkbox style={{ fontSize: '16px' }}>
                              Generate Feature Statistics
                            </Checkbox>
                          </Form.Item>
                        </Col>
                      </Row>
                    </Card>
                  </Form.Item>
                  
                  <Space size={16}>
                    <Button 
                      type="primary" 
                      size="large"
                      onClick={handleProcessDataset}
                      style={{ 
                        backgroundColor: '#1890ff',
                        borderColor: '#1890ff',
                        fontSize: '16px',
                        height: '44px',
                        paddingLeft: '24px',
                        paddingRight: '24px'
                      }}
                    >
                      Process Dataset
                    </Button>
                    
                    <Button 
                      size="large"
                      style={{ 
                        fontSize: '16px',
                        height: '44px',
                        paddingLeft: '16px',
                        paddingRight: '16px',
                        backgroundColor: '#fff'
                      }}
                    >
                      Preview Changes
                    </Button>
                  </Space>
                </>
              )}
            </Form>
          </Card>
        )}
        
        <style jsx global>{`
          .white-background-select .ant-select-selector {
            background-color: #fff !important;
          }
          .ant-select-dropdown {
            background-color: #fff !important;
          }
          .ant-slider-track {
            background-color: #1890ff;
          }
          .ant-checkbox-checked .ant-checkbox-inner {
            background-color: #1890ff;
            border-color: #1890ff;
          }
          .ant-input {
            background-color: #fff;
          }
          .ant-picker {
            background-color: #fff;
          }
          .ant-table-cell {
            background-color: #fff;
          }
          .ant-table-thead > tr > th {
            background-color: #fafafa;
          }
        `}</style>
      </Content>
    </Layout>
  );
} 