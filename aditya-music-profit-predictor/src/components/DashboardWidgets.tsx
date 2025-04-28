import React from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

// Register required Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

interface MetricCardProps {
  title: string;
  value: string;
  icon: string;
  change?: string;
  positive?: boolean;
}

export function MetricCard({ title, value, icon, change, positive }: MetricCardProps) {
  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-4 shadow-lg border border-slate-700/40 transition-all duration-300 hover:shadow-xl hover:scale-[1.02]">
      <div className="flex justify-between items-start mb-2">
        <p className="text-gray-400 text-sm">{title}</p>
        <span className="text-2xl">{icon}</span>
      </div>
      <div className="flex items-end justify-between">
        <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 text-transparent bg-clip-text">{value}</h3>
        {change && (
          <div className={`flex items-center text-sm rounded-full px-2 py-0.5 ${positive ? 'text-green-400 bg-green-900/30' : 'text-red-400 bg-red-900/30'}`}>
            <span className="mr-1">{positive ? '↑' : '↓'}</span>
            {change}
          </div>
        )}
      </div>
    </div>
  );
}

interface StatsSummaryProps {
  engagement: string;
  views: number;
  revenue: number;
}

export function StatsSummary({ engagement, views, revenue }: StatsSummaryProps) {
  let engagementScore = 0;
  
  // Convert engagement level to numeric score
  switch (engagement.toLowerCase()) {
    case 'very high': 
      engagementScore = 90;
      break;
    case 'high': 
      engagementScore = 75;
      break;
    case 'moderate': 
      engagementScore = 50;
      break;
    case 'low': 
      engagementScore = 25;
      break;
    default: 
      engagementScore = 50;
  }
  
  // Determine color based on score
  const getScoreColor = () => {
    if (engagementScore >= 80) return ['#3b82f6', '#1d4ed8']; // blue gradient
    if (engagementScore >= 60) return ['#10b981', '#059669']; // green gradient
    if (engagementScore >= 40) return ['#f59e0b', '#d97706']; // yellow gradient
    return ['#ef4444', '#b91c1c']; // red gradient
  };
  
  const [primaryColor, secondaryColor] = getScoreColor();
  
  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-5 h-full shadow-lg border border-slate-700/40">
      <h3 className="text-lg font-semibold text-white mb-4">Engagement Score</h3>
      
      <div className="flex flex-col items-center">
        <div className="relative w-40 h-40 mb-3">
          <svg className="w-full h-full" viewBox="0 0 100 100">
            {/* Background circle */}
            <circle 
              cx="50" cy="50" r="40" fill="none" 
              stroke="#1e293b" strokeWidth="10" 
            />
            
            {/* Score circle with gradient */}
            <circle 
              cx="50" cy="50" r="40" fill="none" 
              stroke={`url(#scoreGradient)`} strokeWidth="10"
              strokeDasharray={`${engagementScore * 2.51} 251`}
              strokeDashoffset="0" 
              strokeLinecap="round"
              transform="rotate(-90 50 50)"
            />
            
            {/* Define gradient for the score circle */}
            <defs>
              <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor={primaryColor} />
                <stop offset="100%" stopColor={secondaryColor} />
              </linearGradient>
            </defs>
            
            {/* Score text */}
            <text 
              x="50" y="50" 
              textAnchor="middle" 
              dominantBaseline="middle"
              fill="white"
              fontSize="20"
              fontWeight="bold"
            >
              {engagementScore}
            </text>
            <text 
              x="50" y="65" 
              textAnchor="middle"
              fill="gray"
              fontSize="8"
            >
              out of 100
            </text>
          </svg>
        </div>
        
        <p className="text-sm text-gray-400 mt-2">
          Based on predicted engagement level: <span className="text-white font-medium">{engagement}</span>
        </p>
      </div>
      
      <div className="mt-4 space-y-2">
        <div className="flex justify-between items-center p-2 rounded-md bg-slate-700/20">
          <span className="text-gray-400">Predicted Views</span>
          <span className="font-medium">{views.toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center p-2 rounded-md bg-slate-700/20">
          <span className="text-gray-400">Est. Revenue</span>
          <span className="font-medium">₹{revenue.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}

interface PlatformBreakdownProps {
  views: number;
}

export function PlatformBreakdown({ views }: PlatformBreakdownProps) {
  // Distribute views across platforms based on market share
  const platformData = {
    YouTube: Math.round(views * 0.55),  // 55% 
    Instagram: Math.round(views * 0.25), // 25% 
    Facebook: Math.round(views * 0.15), // 15%
    Others: Math.round(views * 0.05),   // 5%
  };
  
  const platformColors = {
    YouTube: 'rgba(255, 0, 0, 0.8)',   // Red
    Instagram: 'rgba(188, 42, 141, 0.8)', // Pink purple
    Facebook: 'rgba(24, 119, 242, 0.8)', // Blue
    Others: 'rgba(75, 85, 99, 0.8)'    // Gray
  };
  
  const data = {
    labels: Object.keys(platformData),
    datasets: [
      {
        data: Object.values(platformData),
        backgroundColor: Object.keys(platformData).map(key => platformColors[key as keyof typeof platformColors]),
        borderColor: Object.keys(platformData).map(key => platformColors[key as keyof typeof platformColors].replace('0.8', '1')),
        borderWidth: 1,
        hoverOffset: 10
      }
    ]
  };
  
  const options = {
    responsive: true,
    cutout: '65%',
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          boxWidth: 12,
          padding: 15,
          color: '#f3f4f6'
        }
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const label = context.label || '';
            const value = context.raw;
            const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0);
            const percentage = Math.round((value / total) * 100);
            return `${label}: ${value.toLocaleString()} views (${percentage}%)`;
          }
        }
      }
    }
  };
  
  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-5 h-full shadow-lg border border-slate-700/40">
      <h3 className="text-lg font-semibold text-white mb-4">Platform Distribution</h3>
      
      <div className="chart-container" style={{ position: 'relative', width: '100%', height: '240px' }}>
        {/* @ts-ignore */}
        <Doughnut data={data} options={options} />
      </div>
      
      <div className="mt-4 space-y-2">
        {Object.entries(platformData).map(([platform, views]) => (
          <div key={platform} className="flex justify-between items-center text-sm">
            <div className="flex items-center">
              <span 
                className="inline-block w-3 h-3 rounded-full mr-2" 
                style={{ backgroundColor: platformColors[platform as keyof typeof platformColors] }}
              ></span>
              <span>{platform}</span>
            </div>
            <span className="font-medium">{views.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
} 