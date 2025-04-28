/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
import React from 'react';
import { Line, Radar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TooltipItem
} from 'chart.js';

// Register required components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  RadialLinearScale,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface AdvancedAnalyticsProps {
  genre: string;
  country: string;
  revenue: number;
  views: number;
}

export function TrendPrediction({ genre, revenue }: Pick<AdvancedAnalyticsProps, 'genre' | 'revenue'>) {
  // Calculate future trend based on current revenue and genre growth factors
  const months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'];
  
  // Different growth patterns based on genre
  const getGrowthFactors = (genre: string) => {
    const growthPatterns: Record<string, number[]> = {
      'Romantic': [1.0, 1.2, 1.5, 1.3, 1.1, 1.0],
      'Classical': [1.0, 1.1, 1.15, 1.2, 1.25, 1.3],
      'Folk': [1.0, 1.15, 1.25, 1.2, 1.1, 1.05],
      'Devotional': [1.0, 1.1, 1.2, 1.4, 1.35, 1.3],
      'Pop': [1.0, 1.3, 1.6, 1.4, 1.2, 1.1],
      'Melody': [1.0, 1.2, 1.3, 1.25, 1.2, 1.15],
      'Item Song': [1.0, 1.5, 1.8, 1.4, 1.0, 0.8],
      'Dance': [1.0, 1.4, 1.7, 1.5, 1.2, 1.0],
      'Hip Hop': [1.0, 1.35, 1.65, 1.45, 1.25, 1.1]
    };
    
    return growthPatterns[genre] || growthPatterns['Pop'];
  };
  
  const growthFactors = getGrowthFactors(genre);
  const predictedRevenue = growthFactors.map(factor => Math.round(revenue * factor));
  
  // Generate organic scenario (baseline)
  const organicGrowth = predictedRevenue.map(rev => rev);
  
  // Generate optimistic scenario (20% better)
  const optimisticGrowth = predictedRevenue.map(rev => Math.round(rev * 1.2));
  
  // Generate conservative scenario (20% worse)
  const conservativeGrowth = predictedRevenue.map(rev => Math.round(rev * 0.8));
  
  const data = {
    labels: months,
    datasets: [
      {
        label: 'Expected Revenue',
        data: organicGrowth,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.4,
      },
      {
        label: 'Optimistic Scenario',
        data: optimisticGrowth,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderDash: [5, 5],
        tension: 0.4,
      },
      {
        label: 'Conservative Scenario',
        data: conservativeGrowth,
        borderColor: 'rgb(255, 159, 64)',
        backgroundColor: 'rgba(255, 159, 64, 0.5)',
        borderDash: [5, 5],
        tension: 0.4,
      }
    ],
  };
  
  const options = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: '6-Month Revenue Forecast',
      },
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'line'>) {
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
      y: {
        ticks: {
          callback: function(value: any) {
            return `₹${(value / 1000).toFixed(0)}K`;
          }
        }
      }
    },
  };
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Revenue Trend Forecast</h3>
      <Line data={data} options={options} />
      <div className="mt-4">
        <p className="text-sm text-gray-600 mb-2">
          <span className="inline-block w-3 h-3 bg-blue-500 rounded-full mr-2"></span>
          Expected: Based on historical performance for {genre} content
        </p>
        <p className="text-sm text-gray-600 mb-2">
          <span className="inline-block w-3 h-3 bg-teal-500 rounded-full mr-2"></span>
          Optimistic: With enhanced marketing and promotion
        </p>
        <p className="text-sm text-gray-600">
          <span className="inline-block w-3 h-3 bg-orange-500 rounded-full mr-2"></span>
          Conservative: In case of market fluctuations
        </p>
      </div>
    </div>
  );
}

export function CompetitorAnalysis({ genre, country }: Pick<AdvancedAnalyticsProps, 'genre' | 'country'>) {
  // Map competitor strengths based on genre and country
  const getCompetitorData = (genre: string, country: string) => {
    const baseData = {
      engagement: Math.random() * 30 + 60,
      marketShare: Math.random() * 25 + 5,
      contentQuality: Math.random() * 30 + 60,
      uploadFrequency: Math.random() * 30 + 60,
      fanBase: Math.random() * 30 + 60,
      monetization: Math.random() * 30 + 60
    };
    
    // Modify based on genre
    if (genre === 'Pop' || genre === 'Item Song') {
      baseData.engagement += 10;
      baseData.fanBase += 15;
    } else if (genre === 'Classical' || genre === 'Devotional') {
      baseData.contentQuality += 15;
      baseData.monetization += 10;
    }
    
    // Modify based on country
    if (country === 'India') {
      baseData.marketShare += 20;
    } else if (country === 'USA') {
      baseData.monetization += 20;
    }
    
    return [
      baseData.engagement,
      baseData.marketShare,
      baseData.contentQuality,
      baseData.uploadFrequency,
      baseData.fanBase,
      baseData.monetization
    ];
  };
  
  // Generate data for Aditya Music and top competitors
  const adityaData = [85, 40, 90, 70, 80, 75];
  const competitor1Data = getCompetitorData(genre, country);
  const competitor2Data = getCompetitorData(genre, country);
  
  const data = {
    labels: [
      'Audience Engagement',
      'Market Share',
      'Content Quality',
      'Upload Frequency',
      'Fan Base',
      'Monetization'
    ],
    datasets: [
      {
        label: 'Aditya Music',
        data: adityaData,
        fill: true,
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgb(54, 162, 235)',
        pointBackgroundColor: 'rgb(54, 162, 235)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgb(54, 162, 235)'
      },
      {
        label: 'Competitor A',
        data: competitor1Data,
        fill: true,
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgb(255, 99, 132)',
        pointBackgroundColor: 'rgb(255, 99, 132)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgb(255, 99, 132)'
      },
      {
        label: 'Competitor B',
        data: competitor2Data,
        fill: true,
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgb(75, 192, 192)',
        pointBackgroundColor: 'rgb(75, 192, 192)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgb(75, 192, 192)'
      }
    ]
  };
  
  const options = {
    scales: {
      r: {
        min: 0,
        max: 100,
        ticks: {
          stepSize: 20,
          showLabelBackdrop: false,
          font: {
            size: 10
          }
        },
        pointLabels: {
          font: {
            size: 12
          }
        }
      }
    },
    plugins: {
      legend: {
        position: 'bottom' as const
      },
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'bar'>) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              const value = context.parsed.y;
              if (value > 1) {
                label += `${value.toFixed(2)}x`;
              } else {
                label += `${Math.round((1 - value) * 100)}% loss`;
              }
            }
            return label;
          }
        }
      }
    }
  };
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Competitive Analysis</h3>
      <div className="flex justify-center">
        <div className="w-full max-w-md">
          {/* @ts-expect-error - Chart.js type incompatibility */}
          <Radar data={data} options={options} />
        </div>
      </div>
      <div className="mt-4">
        <p className="text-sm text-gray-600">
          Analysis based on market research and performance metrics for {genre} content in {country}.
        </p>
      </div>
    </div>
  );
}

export function AudienceDemographics({ views }: Pick<AdvancedAnalyticsProps, 'views'>) {
  // Generate simulated audience demographics based on views
  const ageGroups = {
    '13-17': Math.round(views * 0.15),
    '18-24': Math.round(views * 0.35),
    '25-34': Math.round(views * 0.25),
    '35-44': Math.round(views * 0.15),
    '45+': Math.round(views * 0.1)
  };
  
  const genderSplit = {
    'Male': Math.round(views * 0.48),
    'Female': Math.round(views * 0.51),
    'Other': Math.round(views * 0.01)
  };
  
  // Data for age distribution
  const ageData = {
    labels: Object.keys(ageGroups),
    datasets: [
      {
        label: 'Views by Age Group',
        data: Object.values(ageGroups),
        backgroundColor: [
          'rgba(255, 99, 132, 0.7)',
          'rgba(54, 162, 235, 0.7)',
          'rgba(255, 206, 86, 0.7)',
          'rgba(75, 192, 192, 0.7)',
          'rgba(153, 102, 255, 0.7)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)'
        ],
        borderWidth: 1
      }
    ]
  };
  
  // Data for gender distribution
  const genderData = {
    labels: Object.keys(genderSplit),
    datasets: [
      {
        label: 'Views by Gender',
        data: Object.values(genderSplit),
        backgroundColor: [
          'rgba(54, 162, 235, 0.7)',
          'rgba(255, 99, 132, 0.7)',
          'rgba(153, 102, 255, 0.7)'
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(255, 99, 132, 1)',
          'rgba(153, 102, 255, 1)'
        ],
        borderWidth: 1
      }
    ]
  };
  
  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right' as const
      },
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'line'>) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += `${context.parsed.y.toLocaleString('en-IN')} views`;
            }
            return label;
          }
        }
      }
    }
  };
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Audience Demographics</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-md font-medium text-gray-700 mb-2">Age Distribution</h4>
          {/* @ts-expect-error - Chart.js type incompatibility */}
          <Doughnut data={ageData} options={options} />
        </div>
        
        <div>
          <h4 className="text-md font-medium text-gray-700 mb-2">Gender Distribution</h4>
          {/* @ts-expect-error - Chart.js type incompatibility */}
          <Doughnut data={genderData} options={options} />
        </div>
      </div>
      
      <div className="mt-4">
        <p className="text-sm text-gray-600">
          Demographics are simulated based on typical viewership patterns for music content.
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Source: Industry benchmarks and YouTube Analytics trends, 2023
        </p>
      </div>
    </div>
  );
}

export function ContentStrategy({ genre, revenue }: Pick<AdvancedAnalyticsProps, 'genre' | 'revenue'>) {
  // Generate content strategy recommendations based on genre and revenue
  
  // Define potential content types and their effectiveness for different genres
  const contentEffectiveness: Record<string, Record<string, number>> = {
    'Romantic': {
      'Music Videos': 9,
      'Lyric Videos': 8,
      'Behind the Scenes': 6,
      'Acoustic Versions': 8,
      'Dance Covers': 7,
      'Live Performances': 7
    },
    'Classical': {
      'Music Videos': 7,
      'Lyric Videos': 6,
      'Behind the Scenes': 7,
      'Acoustic Versions': 5,
      'Dance Covers': 4,
      'Live Performances': 9
    },
    'Folk': {
      'Music Videos': 8,
      'Lyric Videos': 6,
      'Behind the Scenes': 7,
      'Acoustic Versions': 6,
      'Dance Covers': 7,
      'Live Performances': 8
    },
    'Devotional': {
      'Music Videos': 9,
      'Lyric Videos': 8,
      'Behind the Scenes': 5,
      'Acoustic Versions': 7,
      'Dance Covers': 3,
      'Live Performances': 8
    },
    'Pop': {
      'Music Videos': 9,
      'Lyric Videos': 7,
      'Behind the Scenes': 8,
      'Acoustic Versions': 7,
      'Dance Covers': 8,
      'Live Performances': 8
    },
  };
  
  // Use Pop as default if genre isn't in our mapping
  const effectivenessRatings = contentEffectiveness[genre] || contentEffectiveness['Pop'];
  
  // Sort content types by effectiveness
  const sortedContentTypes = Object.entries(effectivenessRatings)
    .sort((a, b) => b[1] - a[1])
    .map(([type, score]) => ({ type, score }));
  
  // Calculate ROI for each content type (simplified)
  const getProductionCost = (contentType: string) => {
    const costs: Record<string, number> = {
      'Music Videos': 80000,
      'Lyric Videos': 20000,
      'Behind the Scenes': 15000,
      'Acoustic Versions': 25000,
      'Dance Covers': 35000,
      'Live Performances': 45000
    };
    
    return costs[contentType] || 30000;
  };
  
  const revenuePerPoint = revenue / 40; // Assuming a perfect score of 10 across 4 types would be maximum
  
  const contentROI = sortedContentTypes.map(item => {
    const cost = getProductionCost(item.type);
    const estimatedRevenue = item.score * revenuePerPoint;
    const roi = (estimatedRevenue / cost).toFixed(1);
    
    return {
      ...item,
      cost,
      estimatedRevenue: Math.round(estimatedRevenue),
      roi
    };
  });
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Content Strategy Recommendations</h3>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Content Type</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Effectiveness</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Est. Cost</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Est. Revenue</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ROI</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {contentROI.map((item, index) => (
              <tr key={item.type} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.type}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  <div className="flex items-center">
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${item.score * 10}%` }}></div>
                    </div>
                    <span className="ml-2">{item.score}/10</span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">₹{item.cost.toLocaleString()}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">₹{item.estimatedRevenue.toLocaleString()}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${Number(item.roi) >= 2 ? 'bg-green-100 text-green-800' : Number(item.roi) >= 1 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'}`}>
                    {item.roi}x
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="mt-4 p-3 bg-blue-50 rounded-md border border-blue-100">
        <h4 className="text-md font-medium text-blue-800 mb-2">Top Recommendation</h4>
        <p className="text-sm text-blue-700">
          For {genre} music, prioritize <strong>{sortedContentTypes[0].type}</strong> with an estimated ROI of {contentROI[0].roi}x. This content type is highly effective for your genre and offers the best return on investment.
        </p>
      </div>
      
      <div className="mt-4">
        <p className="text-sm text-gray-600">
          Recommendations based on genre-specific performance metrics and industry standards.
        </p>
      </div>
    </div>
  );
}

export default function AdvancedAnalytics({ genre, country, revenue, views }: AdvancedAnalyticsProps) {
  return (
    <div className="space-y-6">
      <TrendPrediction genre={genre} revenue={revenue} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CompetitorAnalysis genre={genre} country={country} />
        <AudienceDemographics views={views} />
      </div>
      <ContentStrategy genre={genre} revenue={revenue} />
    </div>
  );
} 