/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
import { useState } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  TooltipItem
} from 'chart.js';

// Register required components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface UploadFrequencyOptimizerProps {
  genre: string;
  baseRevenue: number;
  baseViews: number;
}

// Constants for frequency impact
const FREQUENCY_LABELS = [
  '1 video/month',
  '2 videos/month',
  '3 videos/month',
  '1 video/week',
  '2-3 videos/week',
  '3-4 videos/week',
  '1 video/2 days',
  '1 video/day',
  '1+ videos/day'
];

// Define genre-specific upload impact factors
const GENRE_UPLOAD_IMPACT: Record<string, Record<string, number[]>> = {
  'Romantic': {
    viewsMultiplier: [1.0, 1.6, 2.0, 2.5, 2.8, 3.0, 2.7, 2.3, 1.8],
    engagementMultiplier: [1.0, 1.4, 1.8, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0],
    revenueMultiplier: [1.0, 1.7, 2.2, 2.8, 3.1, 3.3, 2.9, 2.4, 1.9],
    productionCost: [50000, 100000, 150000, 250000, 350000, 500000, 700000, 1050000, 1500000]
  },
  'Classical': {
    viewsMultiplier: [1.0, 1.4, 1.7, 2.0, 2.2, 2.3, 2.1, 1.8, 1.5],
    engagementMultiplier: [1.0, 1.3, 1.5, 1.7, 1.6, 1.5, 1.3, 1.1, 0.9],
    revenueMultiplier: [1.0, 1.5, 1.8, 2.1, 2.3, 2.4, 2.2, 1.9, 1.6],
    productionCost: [60000, 120000, 180000, 300000, 420000, 600000, 840000, 1260000, 1800000]
  },
  'Pop': {
    viewsMultiplier: [1.0, 1.8, 2.3, 3.0, 3.5, 3.8, 3.5, 3.0, 2.3],
    engagementMultiplier: [1.0, 1.5, 2.0, 2.5, 2.3, 2.0, 1.8, 1.5, 1.2],
    revenueMultiplier: [1.0, 1.9, 2.5, 3.2, 3.7, 4.0, 3.7, 3.2, 2.5],
    productionCost: [40000, 80000, 120000, 200000, 280000, 400000, 560000, 840000, 1200000]
  },
  'Default': {
    viewsMultiplier: [1.0, 1.6, 2.0, 2.4, 2.7, 2.9, 2.6, 2.2, 1.8],
    engagementMultiplier: [1.0, 1.4, 1.7, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0],
    revenueMultiplier: [1.0, 1.7, 2.1, 2.5, 2.8, 3.0, 2.7, 2.3, 1.9],
    productionCost: [50000, 100000, 150000, 250000, 350000, 500000, 700000, 1050000, 1500000]
  }
};

// Get upload impact data for a specific genre
function getUploadImpact(genre: string) {
  const genreData = GENRE_UPLOAD_IMPACT[genre] || GENRE_UPLOAD_IMPACT['Default'];
  return genreData;
}

interface RevenueChartProps {
  genre: string;
  baseRevenue: number;
}

function RevenueChart({ genre, baseRevenue }: RevenueChartProps) {
  const impactData = getUploadImpact(genre);
  
  // Calculate revenue and costs for each frequency
  const grossRevenue = impactData.revenueMultiplier.map(factor => Math.round(baseRevenue * factor));
  const costs = impactData.productionCost;
  const netRevenue = grossRevenue.map((rev, index) => Math.round(rev - costs[index]));
  
  // Find optimal upload frequency
  const maxNetRevenue = Math.max(...netRevenue);
  const optimalIndex = netRevenue.indexOf(maxNetRevenue);
  const optimalFrequency = FREQUENCY_LABELS[optimalIndex];
  
  const data = {
    labels: FREQUENCY_LABELS,
    datasets: [
      {
        type: 'line' as const,
        label: 'Net Profit',
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 2,
        fill: false,
        data: netRevenue,
      },
      {
        type: 'bar' as const,
        label: 'Gross Revenue',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        data: grossRevenue,
        borderColor: 'white',
        borderWidth: 2,
      },
      {
        type: 'bar' as const,
        label: 'Production Cost',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        data: costs,
      }
    ],
  };
  
  const options: ChartOptions<'bar'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Revenue vs Production Cost by Upload Frequency',
      },
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'bar'>) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += `₹${context.parsed.y.toLocaleString('en-IN')}`;
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        }
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Amount (₹)'
        },
        ticks: {
          callback: function(value) {
            if (Number(value) >= 1000000) {
              return `₹${(Number(value) / 1000000).toFixed(1)}M`;
            } else if (Number(value) >= 1000) {
              return `₹${(Number(value) / 1000).toFixed(0)}K`;
            }
            return `₹${value}`;
          }
        }
      }
    }
  };
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-md">
      {/* @ts-expect-error - Chart.js type incompatibility */}
      <Bar data={data} options={options} />
      <div className="mt-4 bg-blue-50 p-3 rounded-md border border-blue-100">
        <h4 className="font-medium text-blue-800">Optimal Upload Strategy</h4>
        <p className="text-sm text-blue-700 mt-1">
          For <strong>{genre}</strong> music, the optimal upload frequency is <strong>{optimalFrequency}</strong> with an estimated net profit of <strong>₹{maxNetRevenue.toLocaleString('en-IN')}</strong>.
        </p>
      </div>
    </div>
  );
}

interface EngagementChartProps {
  genre: string;
  baseViews: number;
}

function EngagementChart({ genre, baseViews }: EngagementChartProps) {
  const impactData = getUploadImpact(genre);
  
  // Engagement metrics
  const totalViews = impactData.viewsMultiplier.map(factor => Math.round(baseViews * factor));
  const engagementRate = impactData.engagementMultiplier.map(factor => Math.round(factor * 20)); // Scale to percentage
  const retentionData = impactData.engagementMultiplier.map((factor, index) => {
    // Retention drops as frequency increases beyond optimal point
    const dropOff = index > 5 ? (index - 5) * 5 : 0; 
    return Math.max(Math.round(factor * 15) - dropOff, 0); // Scale to percentage
  });
  
  const data = {
    labels: FREQUENCY_LABELS,
    datasets: [
      {
        label: 'Total Views',
        data: totalViews,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        yAxisID: 'y',
      },
      {
        label: 'Engagement Rate (%)',
        data: engagementRate,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderDash: [5, 5],
        yAxisID: 'y1',
      },
      {
        label: 'Audience Retention (%)',
        data: retentionData,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderDash: [5, 5],
        yAxisID: 'y1',
      }
    ],
  };
  
  const options: ChartOptions<'line'> = {
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      title: {
        display: true,
        text: 'Audience Engagement & Retention by Upload Frequency',
      },
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'line'>) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              if (context.dataset.yAxisID === 'y') {
                label += `${context.parsed.y.toLocaleString('en-IN')} views`;
              } else {
                label += `${context.parsed.y}%`;
              }
            }
            return label;
          }
        }
      }
    },
    scales: {
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        title: {
          display: true,
          text: 'Total Views'
        },
        ticks: {
          callback: function(value) {
            if (Number(value) >= 1000000) {
              return `${(Number(value) / 1000000).toFixed(1)}M`;
            } else if (Number(value) >= 1000) {
              return `${(Number(value) / 1000).toFixed(0)}K`;
            }
            return value;
          }
        }
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: {
          display: true,
          text: 'Percentage (%)'
        },
        min: 0,
        max: 50,
        grid: {
          drawOnChartArea: false,
        },
      },
      x: {
        grid: {
          display: false
        }
      }
    },
  };
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-md">
      <Line data={data} options={options} />
      <div className="mt-4">
        <p className="text-sm text-gray-600">
          <span className="inline-block w-3 h-3 bg-blue-500 rounded-full mr-2"></span>
          <strong>Total Views</strong>: Cumulative viewership across all uploaded videos
        </p>
        <p className="text-sm text-gray-600">
          <span className="inline-block w-3 h-3 bg-red-500 rounded-full mr-2"></span>
          <strong>Engagement Rate</strong>: Percentage of viewers interacting (likes, comments, shares)
        </p>
        <p className="text-sm text-gray-600">
          <span className="inline-block w-3 h-3 bg-teal-500 rounded-full mr-2"></span>
          <strong>Audience Retention</strong>: Percentage of viewers who return for future videos
        </p>
      </div>
    </div>
  );
}

interface FrequencyRecommendationProps {
  genre: string;
}

function FrequencyRecommendation({ genre }: FrequencyRecommendationProps) {
  const impactData = getUploadImpact(genre);
  
  // Find optimal frequency for different goals
  const maxRevenueIndex = impactData.revenueMultiplier.indexOf(Math.max(...impactData.revenueMultiplier));
  const maxViewsIndex = impactData.viewsMultiplier.indexOf(Math.max(...impactData.viewsMultiplier));
  const maxEngagementIndex = impactData.engagementMultiplier.indexOf(Math.max(...impactData.engagementMultiplier));
  
  const optimalFrequency = {
    revenue: FREQUENCY_LABELS[maxRevenueIndex],
    views: FREQUENCY_LABELS[maxViewsIndex],
    engagement: FREQUENCY_LABELS[maxEngagementIndex]
  };
  
  // Generate content strategies based on different frequencies
  const strategies = [
    {
      title: "Low Frequency (Quality Focus)",
      frequency: FREQUENCY_LABELS[1], // 2 videos/month
      pros: ["Higher production quality possible", "More time for marketing", "Lower production costs"],
      cons: ["Slower audience growth", "Less consistent presence", "Fewer monetization opportunities"],
      idealFor: "High-quality productions, classical music, devotional content"
    },
    {
      title: "Medium Frequency (Balanced)",
      frequency: FREQUENCY_LABELS[4], // 2-3 videos/week
      pros: ["Good engagement-to-effort ratio", "Steady audience growth", "Consistent presence"],
      cons: ["Moderate production quality trade-offs", "Requires consistent content pipeline"],
      idealFor: "Most music genres, especially Romantic, Pop, and Melody"
    },
    {
      title: "High Frequency (Quantity Focus)",
      frequency: FREQUENCY_LABELS[7], // 1 video/day
      pros: ["Maximizes algorithmic visibility", "Rapid audience growth potential", "More monetization opportunities"],
      cons: ["Quality may suffer", "Higher production costs", "Risk of audience fatigue"],
      idealFor: "Trending content, Dance, Item Songs, viral challenges"
    }
  ];
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Upload Strategy Recommendations</h3>
      
      <div className="mb-6">
        <h4 className="font-medium text-gray-700 mb-2">Optimal Frequency by Goal</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 p-3 rounded-md border border-blue-100">
            <div className="font-medium text-blue-800">Maximum Revenue</div>
            <div className="text-lg font-bold text-blue-900">{optimalFrequency.revenue}</div>
          </div>
          <div className="bg-purple-50 p-3 rounded-md border border-purple-100">
            <div className="font-medium text-purple-800">Maximum Views</div>
            <div className="text-lg font-bold text-purple-900">{optimalFrequency.views}</div>
          </div>
          <div className="bg-green-50 p-3 rounded-md border border-green-100">
            <div className="font-medium text-green-800">Best Engagement</div>
            <div className="text-lg font-bold text-green-900">{optimalFrequency.engagement}</div>
          </div>
        </div>
      </div>
      
      <div>
        <h4 className="font-medium text-gray-700 mb-2">Strategy Options</h4>
        <div className="space-y-4">
          {strategies.map((strategy, index) => (
            <div key={index} className="border rounded-md overflow-hidden">
              <div className="bg-gray-50 px-4 py-2 border-b font-medium">{strategy.title}</div>
              <div className="p-4">
                <div className="mb-2"><span className="font-medium">Frequency:</span> {strategy.frequency}</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
                  <div>
                    <div className="font-medium text-green-700 mb-1">Pros:</div>
                    <ul className="list-disc pl-5 text-sm space-y-1">
                      {strategy.pros.map((pro, i) => (
                        <li key={i}>{pro}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <div className="font-medium text-red-700 mb-1">Cons:</div>
                    <ul className="list-disc pl-5 text-sm space-y-1">
                      {strategy.cons.map((con, i) => (
                        <li key={i}>{con}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                <div><span className="font-medium">Ideal for:</span> {strategy.idealFor}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function UploadFrequencyOptimizer({ genre, baseRevenue, baseViews }: UploadFrequencyOptimizerProps) {
  const [activeTab, setActiveTab] = useState<'revenue' | 'engagement' | 'recommendation'>('revenue');
  
  const tabs = [
    { id: 'revenue', label: 'Revenue Optimization' },
    { id: 'engagement', label: 'Audience Engagement' },
    { id: 'recommendation', label: 'Strategy Recommendations' }
  ];
  
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="flex border-b">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex-1 py-3 px-4 text-center font-medium ${
                activeTab === tab.id
                  ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-500'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        
        <div className="p-4">
          <h2 className="text-xl font-bold text-gray-800 mb-4">Upload Frequency Optimizer</h2>
          
          {activeTab === 'revenue' && (
            <RevenueChart 
              genre={genre}
              baseRevenue={baseRevenue}
            />
          )}
          
          {activeTab === 'engagement' && (
            <EngagementChart 
              genre={genre}
              baseViews={baseViews}
            />
          )}
          
          {activeTab === 'recommendation' && (
            <FrequencyRecommendation 
              genre={genre}
            />
          )}
        </div>
      </div>
    </div>
  );
} 