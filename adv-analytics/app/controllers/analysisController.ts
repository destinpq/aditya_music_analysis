import { DayAnalysis, TagAnalysis, TimeAnalysis } from "../models/types";

// Mock CPM rates (converted to INR - 83.5 INR per USD)
const CPM_RATES = {
  'India': 4175, // 50 USD * 83.5
  'US': 29225, // 350 USD * 83.5
  'UK': 25050, // 300 USD * 83.5
  'Canada': 20875, // 250 USD * 83.5
  'Australia': 22545, // 270 USD * 83.5
  'Germany': 18370, // 220 USD * 83.5
  'France': 16700, // 200 USD * 83.5
  'Brazil': 10020, // 120 USD * 83.5
  'Mexico': 8350, // 100 USD * 83.5
  'Japan': 23380, // 280 USD * 83.5
  'South Korea': 20875, // 250 USD * 83.5
  'Global Average': 15030 // 180 USD * 83.5
};

// Default revenue share (what creator gets after YouTube's cut)
const REVENUE_SHARE = 0.55;

// Get day of week analysis
export async function getDayAnalysis(datasetId: number, startDate?: string, endDate?: string): Promise<DayAnalysis[]> {
  console.log(`Fetching day analysis for dataset ${datasetId} from ${startDate} to ${endDate}`);
  // This would be replaced with an actual API call in production
  return Array(7).fill(null).map((_, i) => ({
    day: i,
    count: Math.floor(Math.random() * 100) + 20,
    avgViews: Math.floor(Math.random() * 10000) + 5000,
    avgLikes: Math.floor(Math.random() * 1000) + 200,
    avgComments: Math.floor(Math.random() * 100) + 10,
    avgEngagement: Math.random() * 5 + 1
  }));
}

// Get time of day analysis
export async function getTimeAnalysis(datasetId: number, startDate?: string, endDate?: string, dayOfWeek?: string): Promise<TimeAnalysis[]> {
  console.log(`Fetching time analysis for dataset ${datasetId} from ${startDate} to ${endDate} for day ${dayOfWeek}`);
  // This would be replaced with an actual API call in production
  return Array(24).fill(null).map((_, i) => ({
    hour: i,
    count: Math.floor(Math.random() * 80) + 10,
    avgViews: Math.floor(Math.random() * 10000) + 3000,
    avgLikes: Math.floor(Math.random() * 800) + 100,
    avgComments: Math.floor(Math.random() * 80) + 5,
    avgEngagement: Math.random() * 4 + 0.5
  }));
}

// Get tag analysis
export async function getTagAnalysis(datasetId: number, startDate?: string, endDate?: string): Promise<TagAnalysis[]> {
  console.log(`Fetching tag analysis for dataset ${datasetId} from ${startDate} to ${endDate}`);
  // This would be replaced with an actual API call in production
  const tags = [
    "music", "trending", "viral", "dance", "comedy", 
    "tutorial", "review", "gaming", "vlog", "reaction"
  ];
  
  return tags.map(tag => ({
    name: tag,
    count: Math.floor(Math.random() * 50) + 3,
    avgViews: Math.floor(Math.random() * 15000) + 2000,
    avgLikes: Math.floor(Math.random() * 1200) + 100,
    avgComments: Math.floor(Math.random() * 120) + 10,
    avgEngagement: Math.random() * 6 + 0.5
  }));
}

// Calculate revenue projections
export function calculateRevenueProjections(predictedViews: number) {
  // Number of AI videos per month
  const aiVideosPerMonth = 600;
  
  // Cost scenarios per video
  const costScenarios = {
    'Low Cost': 100,
    'Medium Cost': 200,
    'High Cost': 300
  };
  
  // Calculate total monthly views
  const totalMonthlyViews = predictedViews * aiVideosPerMonth;
  
  // Calculate revenue for different countries
  const revenuesByCountry: Record<string, number> = {};
  Object.entries(CPM_RATES).forEach(([country, cpmRate]) => {
    revenuesByCountry[country] = (totalMonthlyViews / 1000) * cpmRate * REVENUE_SHARE;
  });
  
  // Calculate profit for different scenarios
  const profitScenarios: Record<string, {
    costPerVideo: number;
    totalCost: number;
    profitsByCountry: Record<string, {
      revenue: number;
      cost: number;
      profit: number;
      roi: number;
    }>;
  }> = {};
  
  Object.entries(costScenarios).forEach(([costName, costPerVideo]) => {
    const totalMonthlyCost = costPerVideo * aiVideosPerMonth;
    const profitsByCountry: Record<string, {
      revenue: number;
      cost: number;
      profit: number;
      roi: number;
    }> = {};
    
    Object.entries(revenuesByCountry).forEach(([country, revenue]) => {
      const profit = revenue - totalMonthlyCost;
      const roi = (profit / totalMonthlyCost) * 100;
      
      profitsByCountry[country] = {
        revenue,
        cost: totalMonthlyCost,
        profit,
        roi
      };
    });
    
    profitScenarios[costName] = {
      costPerVideo,
      totalCost: totalMonthlyCost,
      profitsByCountry
    };
  });
  
  // Calculate breakeven points
  const breakevenPoints: Record<string, number> = {};
  Object.entries(CPM_RATES).forEach(([country, cpmRate]) => {
    const revenuePerVideo = (predictedViews / 1000) * cpmRate * REVENUE_SHARE;
    breakevenPoints[country] = revenuePerVideo;
  });
  
  return {
    totalMonthlyViews,
    profitScenarios,
    breakevenPoints,
    chartData: {} // Placeholder for chart data
  };
} 