// Dataset types
export interface Dataset {
  id: number;
  filename: string;
  uploadDate: string;
  rowCount: number;
}

// Video data types
export interface VideoData {
  id: number;
  videoId: string;
  title: string;
  publishedAt: string;
  datasetId: number;
}

// Video statistics
export interface VideoStats {
  id: number;
  videoId: number;
  viewCount: number;
}

// Video engagement
export interface VideoEngagement {
  id: number;
  videoId: number;
  likeCount: number;
  commentCount: number;
}

// Video metadata
export interface VideoMetaInfo {
  id: number;
  videoId: number;
  durationSeconds: number;
  thumbnailUrl: string;
  categoryId: string;
}

// Video tag
export interface VideoTag {
  id: number;
  videoId: number;
  tagName: string;
}

// Analysis results
export interface AnalysisResult {
  avgViews: number;
  avgLikes: number;
  plotJson: string;
  predictedAiViews: number;
  modelR2: number;
  featureImportance: {
    features: string[];
    importance: number[];
  };
  revenueProjections: {
    totalMonthlyViews: number;
    profitScenarios: Record<string, ProfitScenario>;
    breakevenPoints: Record<string, number>;
    chartData: Record<string, unknown>;
  };
  priceAnalysis: Record<string, unknown>;
  productionVolume: Record<string, unknown>;
}

// Profit scenario
export interface ProfitScenario {
  costPerVideo: number;
  totalCost: number;
  profitsByCountry: Record<string, CountryProfit>;
}

// Country profit
export interface CountryProfit {
  revenue: number;
  cost: number;
  profit: number;
  roi: number;
}

// Day analysis
export interface DayAnalysis {
  day: number;
  count: number;
  avgViews: number;
  avgLikes: number;
  avgComments: number;
  avgEngagement: number;
}

// Time analysis
export interface TimeAnalysis {
  hour: number;
  count: number;
  avgViews: number;
  avgLikes: number;
  avgComments: number;
  avgEngagement: number;
}

// Tag analysis
export interface TagAnalysis {
  name: string;
  count: number;
  avgViews: number;
  avgLikes: number;
  avgComments: number;
  avgEngagement: number;
} 