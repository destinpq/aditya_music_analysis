'use client';

import { useState, useRef, DragEvent, useEffect } from 'react';
import GenreSelector from '@/components/GenreSelector';
import CountrySelector from '@/components/CountrySelector';
import ResultCard from '@/components/ResultCard';
import RevenueGraph from '@/components/RevenueGraph';
import { MetricCard, StatsSummary, PlatformBreakdown } from '@/components/DashboardWidgets';
import ProfitOptimizationChart from '@/components/ProfitOptimizationChart';
import AdvancedAnalytics from '@/components/AdvancedAnalytics';
import EnhancedUI, { 
  useCurrentTheme, 
  AnimatedCard, 
  GlassCard, 
  ShimmerButton, 
  ThemeSelector 
} from '@/components/EnhancedUI';

interface PredictionResult {
  views: number;
  revenue: number;
  engagement: string;
  videoType: string;
}

interface WidgetVisibility {
  revenueGraph: boolean;
  resultCard: boolean;
  statsSummary: boolean;
  platformBreakdown: boolean;
  profitOptimization: boolean;
}

// Add new interface for dashboard mode
interface DashboardMode {
  basic: boolean;
  advanced: boolean;
  export: boolean;
}

// Country CPM values
const COUNTRY_CPM: Record<string, number> = {
  'India': 120,
  'USA': 600,
  'UK': 500,
  'Canada': 450,
  'Australia': 450,
};

export default function Dashboard() {
  // Add EnhancedUI component
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    // Initialize enhanced UI and set mounted to true
    setMounted(true);
  }, []);

  // Get current theme
  const { themeConfig } = useCurrentTheme();

  // Add dashboard mode state
  const [dashboardMode, setDashboardMode] = useState<DashboardMode>({
    basic: true,
    advanced: false,
    export: false
  });
  
  const [genre, setGenre] = useState<string>('');
  const [country, setCountry] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [profit, setProfit] = useState<number | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [widgetVisibility, setWidgetVisibility] = useState<WidgetVisibility>({
    revenueGraph: true,
    resultCard: true,
    statsSummary: true,
    platformBreakdown: true,
    profitOptimization: false
  });
  
  // Initialize drag and drop refs
  const genreDropRef = useRef<HTMLDivElement>(null);
  const countryDropRef = useRef<HTMLDivElement>(null);

  // Add state to track dropped widgets
  const [droppedWidgets, setDroppedWidgets] = useState<{
    genreZone: string | null;
    countryZone: string | null;
  }>({
    genreZone: null,
    countryZone: null
  });

  // Add state for selected comparison countries
  const [selectedComparisonCountries, setSelectedComparisonCountries] = useState<string[]>([]);

  useEffect(() => {
    // Set up dragover event listeners only on client-side
    const genreDropElement = genreDropRef.current;
    const countryDropElement = countryDropRef.current;

    if (genreDropElement && countryDropElement) {
      // Initialize drag and drop on client only
      console.log('Drag and drop initialized on client side');
    }

    return () => {
      // Clean up any event listeners if needed
    };
  }, []);

  // For demonstration - dummy widgets that can be dragged
  const widgets = [
    { id: 'genre', label: 'Genre Selector' },
    { id: 'country', label: 'Country Selector' },
    { id: 'profit', label: 'Profit Indicator' },
    { id: 'revenue', label: 'Revenue Chart' },
    { id: 'analytics', label: 'Analytics' }
  ];

  const handleGenreChange = (selectedGenre: string) => {
    setGenre(selectedGenre);
  };

  const handleCountryChange = (selectedCountry: string) => {
    setCountry(selectedCountry);
  };

  const handleDragStart = (e: DragEvent<HTMLDivElement>, id: string) => {
    e.dataTransfer.setData('text/plain', id);
    if (e.currentTarget.classList) {
      e.currentTarget.classList.add('dragging');
    }
  };

  const handleDragEnd = (e: DragEvent<HTMLDivElement>) => {
    if (e.currentTarget.classList) {
      e.currentTarget.classList.remove('dragging');
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.currentTarget.classList) {
      e.currentTarget.classList.add('drag-over');
    }
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    if (e.currentTarget.classList) {
      e.currentTarget.classList.remove('drag-over');
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>, dropZoneId: string) => {
    e.preventDefault();
    const widgetId = e.dataTransfer.getData('text/plain');
    if (e.currentTarget.classList) {
      e.currentTarget.classList.remove('drag-over');
    }
    
    // Handle the dropped widget
    if (widgetId && dropZoneId) {
      console.log(`Dropped ${widgetId} on ${dropZoneId}`);
      
      // Update state to show the dropped widget in the zone
      if (dropZoneId === 'genreZone' || dropZoneId === 'countryZone') {
        setDroppedWidgets(prev => ({
          ...prev,
          [dropZoneId]: widgetId
        }));
      }
    }
  };

  const handlePredictClick = async () => {
    if (!genre || !country) {
      setError('Please select both genre and country');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ genre, country }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }

      const data = await response.json();
      setResult(data);
      setProfit(data.revenue);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to generate prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Format numbers for display
  const formatViews = (views: number) => {
    return views >= 1000000 
      ? `${(views / 1000000).toFixed(1)}M` 
      : views >= 1000 
        ? `${(views / 1000).toFixed(1)}K` 
        : views.toString();
  };

  const toggleWidget = (widgetName: keyof WidgetVisibility) => {
    setWidgetVisibility(prev => ({
      ...prev,
      [widgetName]: !prev[widgetName]
    }));
  };

  // Add function to toggle dashboard mode
  const toggleDashboardMode = (mode: keyof DashboardMode) => {
    setDashboardMode(prev => ({
      ...prev,
      [mode]: !prev[mode]
    }));
  };

  // Add handler for selecting countries for comparison
  const handleComparisonCountryToggle = (countryName: string) => {
    setSelectedComparisonCountries(prev => 
      prev.includes(countryName)
        ? prev.filter(c => c !== countryName)
        : [...prev, countryName]
    );
  };

  return (
    <>
      <EnhancedUI />
      <div className={`flex flex-col h-screen bg-gradient-to-b ${themeConfig.mainColor} ${themeConfig.textColor}`}>
        {/* Dashboard Header with Profit in top-right */}
        <header className="p-4 border-b border-gray-700 flex justify-between items-center backdrop-blur-md bg-slate-900/50">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold mr-4 bg-gradient-to-r from-blue-400 to-indigo-400 text-transparent bg-clip-text">
              Aditya Music AI Dashboard
            </h1>
            <ThemeSelector />
          </div>
          <div className="flex items-center gap-4">
            {profit !== null ? (
              <GlassCard className="py-2 px-4 rounded-lg profit-indicator">
                <p className="text-xs text-blue-300">Estimated Profit</p>
                <p className="text-2xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 text-transparent bg-clip-text">₹{profit.toLocaleString('en-IN')}</p>
              </GlassCard>
            ) : (
              <div className="bg-gray-800/50 py-2 px-4 rounded-lg border border-gray-700">
                <p className="text-xs text-blue-300">Estimated Profit</p>
                <p className="text-2xl font-bold text-gray-400">--</p>
              </div>
            )}
          </div>
        </header>
        
        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar with Widget Toggles */}
          <aside className={`w-64 ${themeConfig.cardBg} p-4 border-r border-gray-700 overflow-y-auto`}>
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <span className="inline-block w-1 h-5 bg-blue-500 rounded mr-2"></span>
              Dashboard Controls
            </h2>
            
            {/* Dashboard Mode Toggles */}
            <div className="mb-6 space-y-3">
              <h3 className="text-sm text-gray-400 uppercase flex items-center">
                <span className="inline-block w-3 h-3 bg-blue-500 rounded-full mr-2"></span>
                Dashboard Mode
              </h3>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Basic Analytics</span>
                <button 
                  onClick={() => toggleDashboardMode('basic')}
                  className={`w-10 h-5 relative rounded-full transition-colors duration-300 ease-in-out ${dashboardMode.basic ? 'bg-blue-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-300 ease-in-out ${dashboardMode.basic ? 'transform translate-x-5' : ''}`}></span>
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Advanced Analytics</span>
                <button 
                  onClick={() => toggleDashboardMode('advanced')}
                  className={`w-10 h-5 relative rounded-full transition-colors duration-300 ease-in-out ${dashboardMode.advanced ? 'bg-blue-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-300 ease-in-out ${dashboardMode.advanced ? 'transform translate-x-5' : ''}`}></span>
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Export Mode</span>
                <button 
                  onClick={() => toggleDashboardMode('export')}
                  className={`w-10 h-5 relative rounded-full transition-colors duration-300 ease-in-out ${dashboardMode.export ? 'bg-blue-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-300 ease-in-out ${dashboardMode.export ? 'transform translate-x-5' : ''}`}></span>
                </button>
              </div>
            </div>
            
            {/* Widget Visibility Controls */}
            <div className="mb-6 space-y-3">
              <h3 className="text-sm text-gray-400 uppercase flex items-center">
                <span className="inline-block w-3 h-3 bg-purple-500 rounded-full mr-2"></span>
                Visibility
              </h3>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Revenue Graph</span>
                <button 
                  onClick={() => toggleWidget('revenueGraph')}
                  className={`w-10 h-5 relative rounded-full transition-colors duration-300 ease-in-out ${widgetVisibility.revenueGraph ? 'bg-blue-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-300 ease-in-out ${widgetVisibility.revenueGraph ? 'transform translate-x-5' : ''}`}></span>
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Results Card</span>
                <button 
                  onClick={() => toggleWidget('resultCard')}
                  className={`w-10 h-5 relative rounded-full transition-colors duration-300 ease-in-out ${widgetVisibility.resultCard ? 'bg-blue-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-300 ease-in-out ${widgetVisibility.resultCard ? 'transform translate-x-5' : ''}`}></span>
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Engagement Score</span>
                <button 
                  onClick={() => toggleWidget('statsSummary')}
                  className={`w-10 h-5 relative rounded-full transition-colors duration-300 ease-in-out ${widgetVisibility.statsSummary ? 'bg-blue-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-300 ease-in-out ${widgetVisibility.statsSummary ? 'transform translate-x-5' : ''}`}></span>
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Platform Breakdown</span>
                <button 
                  onClick={() => toggleWidget('platformBreakdown')}
                  className={`w-10 h-5 relative rounded-full transition-colors duration-300 ease-in-out ${widgetVisibility.platformBreakdown ? 'bg-blue-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-300 ease-in-out ${widgetVisibility.platformBreakdown ? 'transform translate-x-5' : ''}`}></span>
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Profit Optimization</span>
                <button 
                  onClick={() => toggleWidget('profitOptimization')}
                  className={`w-10 h-5 relative rounded-full transition-colors duration-300 ease-in-out ${widgetVisibility.profitOptimization ? 'bg-blue-600' : 'bg-gray-600'}`}
                >
                  <span className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-300 ease-in-out ${widgetVisibility.profitOptimization ? 'transform translate-x-5' : ''}`}></span>
                </button>
              </div>
            </div>
            
            {/* Draggable Widgets section */}
            <div className="border-t border-gray-700 pt-4 mt-2">
              <h3 className="text-sm text-gray-400 uppercase mb-3 flex items-center">
                <span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                Draggable Widgets
              </h3>
              <div className="space-y-2">
                {mounted && widgets.map((widget) => (
                  <div
                    key={widget.id}
                    draggable
                    onDragStart={(e) => handleDragStart(e, widget.id)}
                    onDragEnd={handleDragEnd}
                    className={`${themeConfig.cardBg} p-3 rounded-lg cursor-move hover:bg-gray-700 transition-colors widget border border-gray-700/50 flex items-center`}
                  >
                    <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                    {widget.label}
                  </div>
                ))}
              </div>
            </div>
          </aside>
          
          {/* Main Dashboard Content */}
          <main className="flex-1 p-6 overflow-y-auto bg-gradient-to-br from-gray-900 to-slate-900">
            {/* Filter controls with enhanced UI */}
            <AnimatedCard className="mb-6" delay={100}>
              <GlassCard className="p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="lg:col-span-4 flex flex-wrap gap-6 justify-between items-end">
                    <div className="flex-1 min-w-[200px]">
                      <label className="block text-sm text-gray-400 mb-2">Select Genre</label>
                      <GenreSelector onSelect={handleGenreChange} />
                    </div>
                    <div className="flex-1 min-w-[200px]">
                      <label className="block text-sm text-gray-400 mb-2">Select Country</label>
                      <CountrySelector onSelect={handleCountryChange} />
                    </div>
                    <ShimmerButton
                      onClick={handlePredictClick}
                      disabled={isLoading || !genre || !country}
                      className="py-2"
                    >
                      {isLoading ? (
                        <span className="flex items-center">
                          <span className="w-4 h-4 mr-2 rounded-full border-2 border-gray-300 border-t-blue-600 animate-spin"></span>
                          Processing...
                        </span>
                      ) : (
                        'Predict Profit'
                      )}
                    </ShimmerButton>
                  </div>
                </div>
              </GlassCard>
            </AnimatedCard>
            
            {/* KPI Metrics with animation */}
            {result && (
              <AnimatedCard className="mb-6" delay={200}>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <MetricCard 
                    title="Predicted Views" 
                    value={formatViews(result.views)} 
                    icon="👁️" 
                    change="12%" 
                    positive={true}
                  />
                  <MetricCard 
                    title="Estimated Revenue" 
                    value={`₹${result.revenue.toLocaleString('en-IN')}`} 
                    icon="💰" 
                    change="8%" 
                    positive={true}
                  />
                  <MetricCard 
                    title="Engagement" 
                    value={result.engagement} 
                    icon="📊" 
                  />
                  <MetricCard 
                    title="Suggested Format" 
                    value={result.videoType} 
                    icon="🎥" 
                  />
                </div>
              </AnimatedCard>
            )}
            
            {/* Drop Zones with content based on what was dropped */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {mounted ? (
                <>
                  <div 
                    ref={genreDropRef}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={(e) => handleDrop(e, 'genreZone')}
                    className={`${themeConfig.cardBg} rounded-lg p-4 min-h-[200px] flex items-center justify-center border-2 border-dashed border-indigo-500/30 drop-zone transition-all duration-300 hover:border-indigo-500/50 shadow-lg`}
                  >
                    {droppedWidgets.genreZone ? (
                      <div className="w-full">
                        <h3 className="text-center font-medium mb-3 text-indigo-400">{widgets.find(w => w.id === droppedWidgets.genreZone)?.label}</h3>
                        {droppedWidgets.genreZone === 'genre' && (
                          <GenreSelector onSelect={handleGenreChange} />
                        )}
                        {droppedWidgets.genreZone === 'profit' && (
                          <div className="p-6 bg-gradient-to-br from-blue-900/30 to-indigo-900/30 rounded-lg text-center border border-blue-800/30">
                            <p className="text-sm text-blue-300">Estimated Profit</p>
                            <p className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 text-transparent bg-clip-text">{profit ? `₹${profit.toLocaleString('en-IN')}` : '--'}</p>
                          </div>
                        )}
                        {droppedWidgets.genreZone === 'revenue' && result && (
                          <div className="w-full">
                            <RevenueGraph country={country} revenue={result.revenue} selectedCountries={selectedComparisonCountries} />
                          </div>
                        )}
                        {droppedWidgets.genreZone === 'analytics' && result && (
                          <div className="p-4 bg-gradient-to-br from-purple-900/30 to-pink-900/30 rounded-lg border border-purple-800/30">
                            <p className="font-medium text-purple-300 mb-3">Analytics Summary</p>
                            <div className="space-y-2">
                              <div className="flex justify-between items-center p-2 rounded-md bg-slate-800/50">
                                <span className="text-gray-400">Views:</span>
                                <span className="font-medium">{formatViews(result.views)}</span>
                              </div>
                              <div className="flex justify-between items-center p-2 rounded-md bg-slate-800/50">
                                <span className="text-gray-400">Revenue:</span>
                                <span className="font-medium">₹{result.revenue.toLocaleString('en-IN')}</span>
                              </div>
                              <div className="flex justify-between items-center p-2 rounded-md bg-slate-800/50">
                                <span className="text-gray-400">Engagement:</span>
                                <span className="font-medium">{result.engagement}</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-indigo-300 bg-indigo-900/20 px-4 py-2 rounded-lg">Drop Widgets Here</p>
                    )}
                  </div>
                  
                  <div 
                    ref={countryDropRef}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={(e) => handleDrop(e, 'countryZone')}
                    className={`${themeConfig.cardBg} rounded-lg p-4 min-h-[200px] flex items-center justify-center border-2 border-dashed border-blue-500/30 drop-zone transition-all duration-300 hover:border-blue-500/50 shadow-lg`}
                  >
                    {droppedWidgets.countryZone ? (
                      <div className="w-full">
                        <h3 className="text-center font-medium mb-3 text-blue-400">{widgets.find(w => w.id === droppedWidgets.countryZone)?.label}</h3>
                        {droppedWidgets.countryZone === 'country' && (
                          <CountrySelector onSelect={handleCountryChange} />
                        )}
                        {droppedWidgets.countryZone === 'profit' && (
                          <div className="p-6 bg-gradient-to-br from-green-900/30 to-emerald-900/30 rounded-lg text-center border border-green-800/30">
                            <p className="text-sm text-green-300">Estimated Profit</p>
                            <p className="text-3xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 text-transparent bg-clip-text">{profit ? `₹${profit.toLocaleString('en-IN')}` : '--'}</p>
                          </div>
                        )}
                        {droppedWidgets.countryZone === 'revenue' && result && (
                          <div className="w-full">
                            <RevenueGraph country={country} revenue={result.revenue} selectedCountries={selectedComparisonCountries} />
                          </div>
                        )}
                        {droppedWidgets.countryZone === 'analytics' && result && (
                          <div className="p-4 bg-gradient-to-br from-cyan-900/30 to-blue-900/30 rounded-lg border border-cyan-800/30">
                            <p className="font-medium text-cyan-300 mb-3">Analytics Summary</p>
                            <div className="space-y-2">
                              <div className="flex justify-between items-center p-2 rounded-md bg-slate-800/50">
                                <span className="text-gray-400">Views:</span>
                                <span className="font-medium">{formatViews(result.views)}</span>
                              </div>
                              <div className="flex justify-between items-center p-2 rounded-md bg-slate-800/50">
                                <span className="text-gray-400">Revenue:</span>
                                <span className="font-medium">₹{result.revenue.toLocaleString('en-IN')}</span>
                              </div>
                              <div className="flex justify-between items-center p-2 rounded-md bg-slate-800/50">
                                <span className="text-gray-400">Engagement:</span>
                                <span className="font-medium">{result.engagement}</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-blue-300 bg-blue-900/20 px-4 py-2 rounded-lg">Drop Widgets Here</p>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className={`${themeConfig.cardBg} rounded-lg p-4 min-h-[200px] flex items-center justify-center border-2 border-dashed border-gray-600`}>
                    <p className="text-gray-400">Loading drop zone...</p>
                  </div>
                  <div className={`${themeConfig.cardBg} rounded-lg p-4 min-h-[200px] flex items-center justify-center border-2 border-dashed border-gray-600`}>
                    <p className="text-gray-400">Loading drop zone...</p>
                  </div>
                </>
              )}
            </div>
              
            {/* Error Message */}
            {error && (
              <div className="col-span-full bg-red-900/30 text-white p-4 rounded-lg mb-6 border border-red-800/50">
                <div className="flex items-center">
                  <span className="text-xl mr-2">⚠️</span>
                  <p>{error}</p>
                </div>
              </div>
            )}
            
            {/* Basic Analytics - Only show if selected */}
            {dashboardMode.basic && result && (
              <AnimatedCard delay={300} className="mb-6">
                <GlassCard>
                  <div className="flex items-center mb-4">
                    <span className="inline-block w-2 h-6 bg-blue-500 rounded mr-3"></span>
                    <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-300 text-transparent bg-clip-text">Basic Analytics</h2>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {widgetVisibility.revenueGraph && (
                      <div className="lg:col-span-2">
                        <div className="mb-3">
                          <h3 className="text-base font-medium mb-2 text-blue-400">Compare with:</h3>
                          <div className="flex flex-wrap gap-2">
                            {Object.keys(COUNTRY_CPM).filter(c => c !== country).map(countryName => {
                              // Get country flag emoji
                              const flagEmoji = {
                                'India': '🇮🇳',
                                'USA': '🇺🇸',
                                'UK': '🇬🇧',
                                'Canada': '🇨🇦',
                                'Australia': '🇦🇺'
                              }[countryName] || '🏳️';
                              
                              return (
                                <button
                                  key={countryName}
                                  onClick={() => handleComparisonCountryToggle(countryName)}
                                  className={`px-3 py-1 text-sm rounded-full transition-colors ${
                                    selectedComparisonCountries.includes(countryName)
                                      ? 'bg-gradient-to-r from-blue-600 to-indigo-700 text-white'
                                      : 'bg-slate-800 text-gray-300 hover:bg-slate-700'
                                  } border border-slate-700`}
                                >
                                  {flagEmoji} {countryName}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                        <RevenueGraph 
                          country={country}
                          revenue={result.revenue}
                          selectedCountries={selectedComparisonCountries}
                        />
                      </div>
                    )}
                    
                    {widgetVisibility.statsSummary && (
                      <div>
                        <StatsSummary 
                          engagement={result.engagement}
                          views={result.views}
                          revenue={result.revenue}
                        />
                      </div>
                    )}
                    
                    {widgetVisibility.platformBreakdown && (
                      <div>
                        <PlatformBreakdown views={result.views} />
                      </div>
                    )}
                    
                    {widgetVisibility.resultCard && (
                      <div className="lg:col-span-2">
                        <ResultCard 
                          views={result.views}
                          revenue={result.revenue}
                          engagement={result.engagement}
                          videoType={result.videoType}
                        />
                      </div>
                    )}

                    {/* Profit Optimization Chart */}
                    {widgetVisibility.profitOptimization && (
                      <div className="lg:col-span-3 mt-6">
                        <ProfitOptimizationChart 
                          revenue={result.revenue}
                          genre={genre}
                          country={country}
                        />
                      </div>
                    )}
                  </div>
                </GlassCard>
              </AnimatedCard>
            )}
            
            {/* Advanced Analytics - Only show if selected */}
            {dashboardMode.advanced && result && (
              <AnimatedCard delay={400}>
                <GlassCard>
                  <div className="flex items-center mb-4">
                    <span className="inline-block w-2 h-6 bg-purple-500 rounded mr-3"></span>
                    <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-300 text-transparent bg-clip-text">Advanced Analytics</h2>
                  </div>
                  <AdvancedAnalytics 
                    genre={genre}
                    country={country}
                    revenue={result.revenue}
                    views={result.views}
                  />
                </GlassCard>
              </AnimatedCard>
            )}
            
            {/* Export Options - Only show if selected */}
            {dashboardMode.export && result && (
              <AnimatedCard delay={500} className="mt-6">
                <GlassCard>
                  <div className="flex items-center mb-4">
                    <span className="inline-block w-2 h-6 bg-green-500 rounded mr-3"></span>
                    <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-300 text-transparent bg-clip-text">Export Options</h2>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <ShimmerButton 
                      className="p-4 flex flex-col items-center justify-center"
                      onClick={() => alert('Export as PDF feature coming soon')}
                    >
                      <span className="text-2xl mb-2">📊</span>
                      <span>Export as PDF</span>
                    </ShimmerButton>
                    
                    <ShimmerButton 
                      className="p-4 flex flex-col items-center justify-center"
                      onClick={() => alert('Export as Excel feature coming soon')}
                    >
                      <span className="text-2xl mb-2">📈</span>
                      <span>Export as Excel</span>
                    </ShimmerButton>
                    
                    <ShimmerButton 
                      className="p-4 flex flex-col items-center justify-center"
                      onClick={() => alert('Export as CSV feature coming soon')}
                    >
                      <span className="text-2xl mb-2">📋</span>
                      <span>Export as CSV</span>
                    </ShimmerButton>
                  </div>
                </GlassCard>
              </AnimatedCard>
            )}
          </main>
        </div>
      </div>
    </>
  );
}
