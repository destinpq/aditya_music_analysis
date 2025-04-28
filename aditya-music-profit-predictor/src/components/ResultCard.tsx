import React from 'react';

interface ResultCardProps {
  views: number;
  revenue: number;
  engagement: string;
  videoType: string;
}

export default function ResultCard({ views, revenue, engagement, videoType }: ResultCardProps) {
  // Format numbers for display
  const formatValue = (value: number) => {
    return value >= 1000000 
      ? `${(value / 1000000).toFixed(1)}M` 
      : value >= 1000 
        ? `${(value / 1000).toFixed(1)}K` 
        : value.toString();
  };

  // Get engagement color
  const getEngagementColor = () => {
    switch (engagement.toLowerCase()) {
      case 'very high': return 'bg-green-500';
      case 'high': return 'bg-emerald-500';
      case 'moderate': return 'bg-yellow-500';
      case 'low': return 'bg-red-500';
      default: return 'bg-blue-500';
    }
  };

  // Get video type icon
  const getVideoTypeIcon = () => {
    switch (videoType.toLowerCase()) {
      case 'music video': return '🎬';
      case 'lyric video': return '🎵';
      case 'dance cover': return '💃';
      case 'acoustic version': return '🎸';
      case 'behind the scenes': return '🎭';
      case 'live performance': return '🎤';
      default: return '📽️';
    }
  };

  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl overflow-hidden shadow-lg border border-slate-700/30">
      <div className="bg-gradient-to-r from-blue-600/20 to-indigo-600/20 px-6 py-4 border-b border-slate-700/30">
        <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-300 text-transparent bg-clip-text">Prediction Results</h2>
      </div>
      
      <div className="p-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="col-span-2 md:col-span-1">
            <div className="flex flex-col space-y-5">
              <div className="flex items-center gap-4">
                <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-3 rounded-lg shadow-lg">
                  <span className="text-2xl">👁️</span>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Predicted Views</p>
                  <p className="text-xl font-bold">{formatValue(views)}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="bg-gradient-to-br from-green-500 to-emerald-600 p-3 rounded-lg shadow-lg">
                  <span className="text-2xl">💰</span>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Estimated Revenue</p>
                  <p className="text-xl font-bold">₹{revenue.toLocaleString('en-IN')}</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="col-span-2 md:col-span-1">
            <div className="flex flex-col space-y-5">
              <div className="flex items-center gap-4">
                <div className={`bg-gradient-to-br from-${engagement.toLowerCase() === 'very high' ? 'green' : engagement.toLowerCase() === 'high' ? 'emerald' : engagement.toLowerCase() === 'moderate' ? 'yellow' : 'red'}-500 to-${engagement.toLowerCase() === 'very high' ? 'green' : engagement.toLowerCase() === 'high' ? 'emerald' : engagement.toLowerCase() === 'moderate' ? 'yellow' : 'red'}-600 p-3 rounded-lg shadow-lg`}>
                  <span className="text-2xl">📊</span>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Engagement Level</p>
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full mr-2 ${getEngagementColor()}`}></div>
                    <p className="text-xl font-bold">{engagement}</p>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="bg-gradient-to-br from-purple-500 to-pink-600 p-3 rounded-lg shadow-lg">
                  <span className="text-2xl">{getVideoTypeIcon()}</span>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Suggested Format</p>
                  <p className="text-xl font-bold">{videoType}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 p-4 rounded-lg bg-blue-600/10 border border-blue-500/20">
          <div className="flex items-start">
            <div className="p-2 rounded-full bg-blue-500/20 mr-3 mt-1">
              <span className="text-blue-400 text-lg">💡</span>
            </div>
            <div>
              <h3 className="font-semibold text-blue-400 mb-1">Recommendation</h3>
              <p className="text-sm text-gray-300">
                Based on our analysis, creating a <strong>{videoType.toLowerCase()}</strong> for this genre and market 
                would generate the best engagement and revenue potential. Consider incorporating trending elements 
                to maximize your reach.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 