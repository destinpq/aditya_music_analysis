import axios from 'axios';

// Use environment variable for API base URL, with fallback
// For Digital Ocean deployment, we'll use the Next.js API routes
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '/api';
const ML_API_URL = API_BASE_URL;
console.log('Using API base URL:', API_BASE_URL);

// Define types based on backend schemas
export interface Dataset {
  id: number;
  filename: string;
  uploadDate: string; // Frontend uses camelCase
  row_count?: number;
}

// Backend response uses snake_case
interface DatasetResponse {
  id: number;
  filename: string;
  upload_date: string;
  row_count?: number;
  detail?: string; // Error detail from backend
}

export interface VideoStats {
  view_count: number;
  like_count: number;
  dislike_count: number;
  favorite_count: number;
  comment_count: number;
}

export interface VideoEngagement {
  engagement_rate: number;
  like_ratio: number;
  comment_ratio: number;
}

export interface VideoMetaInfo {
  duration: number;
  channel_id: string | null;
  category_id: string | null;
  is_unlisted: boolean;
}

export interface Video {
  id: number;
  video_id: string;
  title: string;
  published_at: string;
  stats: VideoStats;
  engagement: VideoEngagement;
  meta_info: VideoMetaInfo;
  tags: string[];
}

export interface AnalysisDataPoint {
  value: number | string | object | (number | string | object)[];
}

export interface AnalysisItem {
  title: string;
  data: AnalysisDataPoint;
}

export interface AnalysisResults {
  results: AnalysisItem[];
}

export interface DatasetStats {
  total_videos: number;
  total_views: number;
  avg_engagement_rate: number;
  avg_like_ratio: number;
  top_videos: {
    video_id: string;
    title: string;
    views: number;
  }[];
}

export interface CountryRevenue {
  views: number;
  revenue: number;
  rate_per_1000: number;
}

export interface DailyRevenue {
  date: string;
  revenue: number;
}

export interface RevenueResponse {
  total_views: number;
  total_revenue: number;
  country_revenue: Record<string, CountryRevenue>;
  time_series: DailyRevenue[];
}

export interface AIPredictionRequest {
  content_type: string;
  day_of_week: string;
  time_of_day: string;
  dataset_id?: number;
}

export interface DailyProjection {
  day: number;
  date: string;
  views: number;
  revenue: number;
}

export interface AIPrediction {
  content_type: string;
  predicted_views: number;
  predicted_likes: number;
  predicted_comments: number;
  predicted_engagement_rate: number;
  predicted_revenue: number;
  optimal_posting: {
    day: string;
    time: string;
  };
  growth_projection: DailyProjection[];
}

export interface SuccessResponse {
  success: boolean;
  message: string;
}

export interface PredictionInput {
  title?: string;
  duration?: string;
  tags?: string;
  published_at?: string;
  category_id?: string;
}

export interface EarningsInput {
  view_count: number;
  custom_cpm?: number;
  country?: string;
  geography?: Record<string, number>;
  monetization_rate?: number;
  ad_impression_rate?: number;
}

export interface TrainModelInput {
  dataset_id: number;
  model_name: string;
  test_size?: number;
}

export interface CPMRate {
  min: number;
  avg: number;
  max: number;
}

export interface DatasetProcessRequest {
  dataset_id: number;
  test_size?: number;
  random_state?: number;
  dataset_name?: string;
}

export interface MLModelInfo {
  dataset_id: number;
  target_column: string;
  feature_columns: string[];
  models: string[];
  training_samples: number;
  test_samples: number;
  evaluation: Record<string, any>;
  has_tuning: boolean;
}

export interface MLTrainParams {
  dataset_id: number;
  model_types?: string[];
  target_column: string;
  feature_columns?: string[];
  test_size?: number;
}

export interface MLTuneParams {
  dataset_id: number;
  target_column: string;
  feature_columns?: string[];
  test_size?: number;
  n_splits?: number;
}

export interface MLPredictParams {
  dataset_id: number;
  target_column: string;
  feature_columns?: string[];
  model_type?: string;
  video_ids?: string[];
}

// Mock data for development and standalone use
const mockDatasets: Dataset[] = [
  { id: 1, filename: 'ADITYA MUSIC.csv', uploadDate: '2023-04-15', row_count: 1240 },
  { id: 2, filename: 'Backup Channel.csv', uploadDate: '2023-04-18', row_count: 890 },
];

const mockModels: MLModelInfo[] = [
  {
    dataset_id: 1,
    target_column: 'view_count',
    feature_columns: ['duration', 'category_id', 'tags'],
    models: ['XGBoost', 'RandomForest'],
    training_samples: 992,
    test_samples: 248,
    evaluation: {
      'XGBoost': { 'rmse': 12450, 'r2': 0.78 },
      'RandomForest': { 'rmse': 14320, 'r2': 0.72 }
    },
    has_tuning: true
  },
  {
    dataset_id: 1,
    target_column: 'like_count',
    feature_columns: ['duration', 'category_id', 'tags', 'view_count'],
    models: ['XGBoost'],
    training_samples: 992,
    test_samples: 248,
    evaluation: {
      'XGBoost': { 'rmse': 890, 'r2': 0.82 }
    },
    has_tuning: false
  }
];

// API client
const api = {
  // Get all datasets
  getDatasets: async (): Promise<Dataset[]> => {
    if (process.env.NODE_ENV === 'development') {
      // Return mock data in development
      return Promise.resolve(mockDatasets);
    }
    
    try {
      const response = await axios.get<DatasetResponse[]>(`${API_BASE_URL}/datasets`);
      // Convert snake_case to camelCase
      const datasets: Dataset[] = response.data.map(dataset => ({
        id: dataset.id,
        filename: dataset.filename,
        uploadDate: dataset.upload_date,
        row_count: dataset.row_count
      }));
      return datasets;
    } catch (error) {
      console.error("Error fetching datasets", error);
      // Return mock data as fallback
      return mockDatasets;
    }
  },

  // Upload a new dataset (CSV file)
  uploadDataset: async (file: File, onProgress?: (progress: number) => void): Promise<Dataset> => {
    if (process.env.NODE_ENV === 'development') {
      // Simulate upload in development
      return new Promise((resolve) => {
        let progress = 0;
        const interval = setInterval(() => {
          progress += 10;
          if (onProgress) onProgress(progress);
          if (progress >= 100) {
            clearInterval(interval);
            const newDataset = {
              id: mockDatasets.length + 1,
              filename: file.name,
              uploadDate: new Date().toISOString().split('T')[0],
              row_count: Math.floor(Math.random() * 1000) + 500
            };
            mockDatasets.push(newDataset);
            resolve(newDataset);
          }
        }, 300);
      });
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post<DatasetResponse>(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Accept': 'application/json',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            console.log(`Upload progress: ${percentCompleted}%`);
            if (onProgress) {
              onProgress(percentCompleted);
            }
          }
        },
        timeout: 300000, // 5 minutes
      });
      
      // Convert snake_case to camelCase
      const dataset: Dataset = {
        id: response.data.id,
        filename: response.data.filename,
        uploadDate: response.data.upload_date,
        row_count: response.data.row_count
      };
      
      return dataset;
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  },

  // Get videos for a specific dataset with sorting options
  getVideos: async (
    datasetId: number, 
    sortBy?: string,
    sortOrder?: 'asc' | 'desc',
    limit?: number
  ): Promise<Video[]> => {
    if (process.env.NODE_ENV === 'development') {
      // Return mock data in development
      return Promise.resolve([]);
    }
    
    const params = new URLSearchParams();
    
    if (sortBy) params.append('sort_by', sortBy);
    if (sortOrder) params.append('sort_order', sortOrder);
    if (limit) params.append('limit', limit.toString());
    
    const queryString = params.toString() ? `?${params.toString()}` : '';
    try {
      const response = await axios.get(`${API_BASE_URL}/datasets/${datasetId}/videos${queryString}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching videos", error);
      return [];
    }
  },

  // Get analysis for a dataset - try multiple endpoints
  getAnalysis: async (datasetId: number): Promise<AnalysisResults | any> => {
    if (process.env.NODE_ENV === 'development') {
      // Return mock data in development
      return Promise.resolve({
        results: [
          { title: 'Total Videos', data: { value: 350 } },
          { title: 'Total Views', data: { value: 12540000 } },
          { title: 'Average Views', data: { value: 35800 } },
          { title: 'Engagement Rate', data: { value: 6.8 } }
        ]
      });
    }
    
    try {
      const response = await axios.get(`${API_BASE_URL}/analysis/${datasetId}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching analysis", error);
      return null;
    }
  },
  
  // ML API endpoints
  mlGetModelsInfo: async (): Promise<MLModelInfo[]> => {
    if (process.env.NODE_ENV === 'development') {
      // Return mock data in development
      return Promise.resolve(mockModels);
    }
    
    try {
      const response = await axios.get(`${ML_API_URL}/ml/models`);
      return response.data.models;
    } catch (error) {
      console.error("Error fetching ML models", error);
      return [];
    }
  },

  mlTrainModels: async (params: MLTrainParams): Promise<any> => {
    if (process.env.NODE_ENV === 'development') {
      // Simulate training in development
      return new Promise(resolve => {
        setTimeout(() => {
          const newModel = {
            dataset_id: params.dataset_id,
            target_column: params.target_column,
            feature_columns: params.feature_columns || ['duration', 'category_id', 'tags'],
            models: params.model_types || ['XGBoost'],
            training_samples: 992,
            test_samples: 248,
            evaluation: {
              'XGBoost': { 'rmse': 12450, 'r2': 0.78 }
            },
            has_tuning: false
          };
          mockModels.push(newModel);
          resolve({
            success: true,
            message: 'Models trained successfully',
            model_info: newModel
          });
        }, 5000);
      });
    }
    
    const response = await axios.post(`${ML_API_URL}/ml/train`, params);
    return response.data;
  },

  mlPredictWithModels: async (params: MLPredictParams): Promise<any> => {
    if (process.env.NODE_ENV === 'development') {
      // Simulate prediction in development
      return new Promise(resolve => {
        setTimeout(() => {
          resolve({
            success: true,
            predictions: {
              video_id_1: {
                predicted_views: Math.floor(Math.random() * 100000) + 10000,
                confidence: 0.85
              }
            }
          });
        }, 2000);
      });
    }
    
    const response = await axios.post(`${ML_API_URL}/ml/predict`, params);
    return response.data;
  }
};

export default api;