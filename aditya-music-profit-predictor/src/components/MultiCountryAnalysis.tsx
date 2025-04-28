import React, { useState } from 'react';
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
  RadarController,
  RadialLinearScale,
  Filler,
  ChartOptions,
  TooltipItem
} from 'chart.js';
import { Bar, Radar } from 'react-chartjs-2';
// Temporary comment out the icon imports until the package is installed
// import { FiDownload, FiFilter, FiGlobe, FiTrendingUp } from 'react-icons/fi';

// Register required ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadarController,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Define interfaces
interface CountryData {
  id: string;
  name: string;
  revenuePerView: number;
  avgEngagement: number;
  growthRate: number;
  marketSize: number;
  competitionLevel: number;
  contentPreference: {
    [key: string]: number;
  };
}

interface RevenueComparisonProps {
  selectedCountries: CountryData[];
  baseViews: number;
}

interface MetricsRadarProps {
  selectedCountries: CountryData[];
}

interface ContentPreferenceProps {
  selectedCountries: CountryData[];
}

interface OpportunityScoreProps {
  selectedCountries: CountryData[];
}

interface CountrySelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  availableCountries: CountryData[];
  selectedCountryIds: string[];
  onSelectionChange: (selectedIds: string[]) => void;
}

// Colors for charts
const CHART_COLORS = [
  'rgba(255, 99, 132, 0.7)',
  'rgba(54, 162, 235, 0.7)',
  'rgba(255, 206, 86, 0.7)',
  'rgba(75, 192, 192, 0.7)',
  'rgba(153, 102, 255, 0.7)',
  'rgba(255, 159, 64, 0.7)'
];

// Available countries with their metrics
const AVAILABLE_COUNTRIES: CountryData[] = [
  {
    id: 'in',
    name: 'India',
    revenuePerView: 0.75,
    avgEngagement: 65,
    growthRate: 12,
    marketSize: 750000000,
    competitionLevel: 75,
    contentPreference: {
      'Classical': 25,
      'Bollywood': 80,
      'Folk': 45,
      'Indie': 35,
      'Devotional': 60
    }
  },
  {
    id: 'us',
    name: 'United States',
    revenuePerView: 2.1,
    avgEngagement: 55,
    growthRate: 5,
    marketSize: 320000000,
    competitionLevel: 90,
    contentPreference: {
      'Classical': 15,
      'Bollywood': 10,
      'Folk': 15,
      'Indie': 60,
      'Devotional': 5
    }
  },
  {
    id: 'gb',
    name: 'United Kingdom',
    revenuePerView: 1.85,
    avgEngagement: 52,
    growthRate: 4,
    marketSize: 68000000,
    competitionLevel: 85,
    contentPreference: {
      'Classical': 25,
      'Bollywood': 15,
      'Folk': 20,
      'Indie': 55,
      'Devotional': 5
    }
  },
  {
    id: 'ca',
    name: 'Canada',
    revenuePerView: 1.65,
    avgEngagement: 50,
    growthRate: 6,
    marketSize: 38000000,
    competitionLevel: 80,
    contentPreference: {
      'Classical': 20,
      'Bollywood': 15,
      'Folk': 25,
      'Indie': 60,
      'Devotional': 10
    }
  },
  {
    id: 'au',
    name: 'Australia',
    revenuePerView: 1.55,
    avgEngagement: 48,
    growthRate: 5,
    marketSize: 25000000,
    competitionLevel: 70,
    contentPreference: {
      'Classical': 20,
      'Bollywood': 10,
      'Folk': 20,
      'Indie': 65,
      'Devotional': 5
    }
  },
  {
    id: 'ae',
    name: 'UAE',
    revenuePerView: 1.95,
    avgEngagement: 60,
    growthRate: 8,
    marketSize: 9800000,
    competitionLevel: 65,
    contentPreference: {
      'Classical': 20,
      'Bollywood': 50,
      'Folk': 25,
      'Indie': 30,
      'Devotional': 35
    }
  },
  {
    id: 'sg',
    name: 'Singapore',
    revenuePerView: 1.75,
    avgEngagement: 55,
    growthRate: 7,
    marketSize: 5700000,
    competitionLevel: 72,
    contentPreference: {
      'Classical': 25,
      'Bollywood': 35,
      'Folk': 15,
      'Indie': 45,
      'Devotional': 20
    }
  },
  {
    id: 'my',
    name: 'Malaysia',
    revenuePerView: 1.25,
    avgEngagement: 58,
    growthRate: 9,
    marketSize: 32000000,
    competitionLevel: 68,
    contentPreference: {
      'Classical': 15,
      'Bollywood': 40,
      'Folk': 35,
      'Indie': 30,
      'Devotional': 25
    }
  },
  {
    id: 'za',
    name: 'South Africa',
    revenuePerView: 0.95,
    avgEngagement: 62,
    growthRate: 10,
    marketSize: 58000000,
    competitionLevel: 60,
    contentPreference: {
      'Classical': 10,
      'Bollywood': 20,
      'Folk': 55,
      'Indie': 30,
      'Devotional': 15
    }
  },
  {
    id: 'ng',
    name: 'Nigeria',
    revenuePerView: 0.65,
    avgEngagement: 70,
    growthRate: 14,
    marketSize: 206000000,
    competitionLevel: 55,
    contentPreference: {
      'Classical': 5,
      'Bollywood': 15,
      'Folk': 70,
      'Indie': 25,
      'Devotional': 30
    }
  }
];

// Country Selection Modal Component
function CountrySelectionModal({ 
  isOpen, 
  onClose, 
  availableCountries, 
  selectedCountryIds, 
  onSelectionChange 
}: CountrySelectionModalProps) {
  // Local state to track selections before applying
  const [localSelectedIds, setLocalSelectedIds] = useState<string[]>([...selectedCountryIds]);
  
  // Toggle a country selection
  const toggleCountry = (countryId: string) => {
    if (localSelectedIds.includes(countryId)) {
      setLocalSelectedIds(prev => prev.filter(id => id !== countryId));
    } else {
      // Limit selection to 6 countries
      if (localSelectedIds.length < 6) {
        setLocalSelectedIds(prev => [...prev, countryId]);
      }
    }
  };
  
  // Apply selection and close modal
  const handleApply = () => {
    onSelectionChange(localSelectedIds);
    onClose();
  };
  
  // Reset and close modal
  const handleCancel = () => {
    setLocalSelectedIds([...selectedCountryIds]);
    onClose();
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="modal-overlay">
      <div className="modal-container">
        <div className="modal-header">
          <h3>Select Countries (max 6)</h3>
          <button className="close-button" onClick={handleCancel}>×</button>
        </div>
        
        <div className="modal-body">
          <div className="country-grid">
            {availableCountries.map(country => (
              <div
                key={country.id}
                className={`country-option ${localSelectedIds.includes(country.id) ? 'selected' : ''}`}
                onClick={() => toggleCountry(country.id)}
              >
                <div className="country-checkbox">
                  {localSelectedIds.includes(country.id) && <span>✓</span>}
                </div>
                <div className="country-name">{country.name}</div>
              </div>
            ))}
          </div>
          
          <div className="selection-count">
            Selected: {localSelectedIds.length}/6 countries
          </div>
        </div>
        
        <div className="modal-footer">
          <button className="cancel-button" onClick={handleCancel}>
            Cancel
          </button>
          <button className="apply-button" onClick={handleApply}>
            Apply
          </button>
        </div>
      </div>
      
      <style jsx>{`
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: rgba(0, 0, 0, 0.5);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 1000;
        }
        
        .modal-container {
          background-color: white;
          border-radius: 8px;
          width: 90%;
          max-width: 600px;
          max-height: 90vh;
          overflow-y: auto;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem 1.5rem;
          border-bottom: 1px solid #eee;
        }
        
        .modal-header h3 {
          margin: 0;
          font-size: 1.2rem;
          color: #333;
        }
        
        .close-button {
          background: none;
          border: none;
          font-size: 1.5rem;
          cursor: pointer;
          color: #666;
        }
        
        .modal-body {
          padding: 1.5rem;
        }
        
        .country-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 0.8rem;
        }
        
        .country-option {
          display: flex;
          align-items: center;
          padding: 0.8rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .country-option:hover {
          background-color: #f5f5f5;
        }
        
        .country-option.selected {
          background-color: #4a6fa5;
          color: white;
          border-color: #4a6fa5;
        }
        
        .country-checkbox {
          width: 18px;
          height: 18px;
          border: 2px solid #888;
          border-radius: 4px;
          margin-right: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: white;
          color: #4a6fa5;
        }
        
        .selected .country-checkbox {
          background-color: #4a6fa5;
          border-color: white;
          color: white;
        }
        
        .country-name {
          font-size: 0.95rem;
        }
        
        .selection-count {
          margin-top: 1.2rem;
          font-size: 0.9rem;
          color: #666;
          text-align: right;
        }
        
        .modal-footer {
          display: flex;
          justify-content: flex-end;
          gap: 1rem;
          padding: 1rem 1.5rem;
          border-top: 1px solid #eee;
        }
        
        .cancel-button {
          padding: 0.5rem 1rem;
          background-color: #f5f5f5;
          border: 1px solid #ddd;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .apply-button {
          padding: 0.5rem 1rem;
          background-color: #4a6fa5;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
      `}</style>
    </div>
  );
}

// Component for Revenue Comparison
function RevenueComparison({ selectedCountries, baseViews }: RevenueComparisonProps) {
  // Calculate projected revenue for each country based on views
  const data = {
    labels: selectedCountries.map(country => country.name),
    datasets: [
      {
        label: 'Projected Revenue (₹)',
        data: selectedCountries.map(country => Math.round(baseViews * country.revenuePerView)),
        backgroundColor: selectedCountries.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
        borderColor: 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Revenue Comparison by Country'
      },
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'bar'>) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += new Intl.NumberFormat('en-IN', { 
                style: 'currency', 
                currency: 'INR',
                maximumFractionDigits: 0
              }).format(context.parsed.y);
            }
            return label;
          }
        }
      }
    }
  };

  return (
    <div className="chart-container">
      <h3>Revenue Comparison</h3>
      <Bar data={data} options={options} />
    </div>
  );
}

// Component for Metrics Radar
function MetricsRadar({ selectedCountries }: MetricsRadarProps) {
  // Normalize metrics for radar chart
  const radarLabels = ['Revenue/View', 'Engagement', 'Growth', 'Market Size', 'Competition'];
  
  // Create datasets for each country
  const datasets = selectedCountries.map((country, index) => {
    // Normalize values to 0-100 scale for radar chart
    const normalizedData = [
      (country.revenuePerView / 2.5) * 100, // Max RPV is around 2.5
      country.avgEngagement,
      (country.growthRate / 15) * 100, // Max growth rate is around 15%
      (Math.log10(country.marketSize) / Math.log10(1000000000)) * 100, // Log scale for market size
      country.competitionLevel
    ];
    
    return {
      label: country.name,
      data: normalizedData,
      backgroundColor: `${CHART_COLORS[index % CHART_COLORS.length].replace('0.7', '0.2')}`,
      borderColor: CHART_COLORS[index % CHART_COLORS.length],
      borderWidth: 2,
      pointBackgroundColor: CHART_COLORS[index % CHART_COLORS.length],
      pointRadius: 4
    };
  });
  
  const data = {
    labels: radarLabels,
    datasets
  };
  
  const options: ChartOptions<'radar'> = {
    responsive: true,
    scales: {
      r: {
        min: 0,
        max: 100,
        ticks: {
          stepSize: 20,
          showLabelBackdrop: false,
          display: false
        },
        pointLabels: {
          font: {
            size: 12,
            weight: 600
          }
        },
        grid: {
          circular: true
        }
      }
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'radar'>) {
            if (context.datasetIndex === undefined) return '';
            
            const countryIndex = context.datasetIndex;
            const metricIndex = context.dataIndex;
            const country = selectedCountries[countryIndex];
            const originalValue = [
              country.revenuePerView,
              country.avgEngagement,
              country.growthRate,
              country.marketSize,
              country.competitionLevel
            ][metricIndex];
            
            const formattedValue = [
              `₹${originalValue.toFixed(2)}`,
              `${originalValue}%`,
              `${originalValue}%`,
              (originalValue / 1000000).toFixed(1) + 'M',
              `${originalValue}/100`
            ][metricIndex];
            
            return `${country.name}: ${formattedValue}`;
          }
        }
      }
    }
  };
  
  return (
    <div className="chart-container">
      <h3>Metrics Comparison</h3>
      <div className="radar-chart-container">
        <Radar data={data} options={options} />
      </div>
      <div className="metrics-legend">
        <p><strong>Revenue/View:</strong> Average revenue earned per view</p>
        <p><strong>Engagement:</strong> Average engagement rate (likes, comments, shares)</p>
        <p><strong>Growth:</strong> YoY growth rate of music content consumption</p>
        <p><strong>Market Size:</strong> Potential audience size</p>
        <p><strong>Competition:</strong> Level of competition (higher is more competitive)</p>
      </div>
    </div>
  );
}

// Component for Content Preference Analysis
function ContentPreference({ selectedCountries }: ContentPreferenceProps) {
  // Get all unique genres across all countries
  const allGenres = Array.from(
    new Set(
      selectedCountries.flatMap(country => Object.keys(country.contentPreference))
    )
  ).sort();
  
  // Create datasets for each genre
  const datasets = allGenres.map((genre, genreIndex) => {
    return {
      label: genre,
      data: selectedCountries.map(country => country.contentPreference[genre] || 0),
      backgroundColor: CHART_COLORS[genreIndex % CHART_COLORS.length],
      borderColor: 'rgba(0, 0, 0, 0.1)',
      borderWidth: 1
    };
  });
  
  const data = {
    labels: selectedCountries.map(country => country.name),
    datasets
  };
  
  const options = {
    responsive: true,
    scales: {
      x: {
        stacked: true,
      },
      y: {
        stacked: true,
        max: 100,
        title: {
          display: true,
          text: 'Preference Level (%)'
        }
      }
    },
    plugins: {
      title: {
        display: true,
        text: 'Content Preference by Country'
      },
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'bar'>) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y + '%';
            }
            return label;
          }
        }
      }
    }
  };
  
  return (
    <div className="chart-container">
      <h3>Content Preference Analysis</h3>
      <Bar data={data} options={options} />
      <div className="content-preference-legend">
        <p>This chart shows the popularity of different music genres in each country, 
        helping you tailor your content for specific markets.</p>
      </div>
    </div>
  );
}

// Component for Opportunity Score
function OpportunityScore({ selectedCountries }: OpportunityScoreProps) {
  // Calculate opportunity score for each country based on multiple factors
  const opportunityScores = selectedCountries.map(country => {
    // Formula weighs different factors:
    // - Higher revenue per view is better
    // - Higher engagement is better
    // - Higher growth rate is better
    // - Larger market size is better
    // - Lower competition is better (so we invert this)
    
    const revenueScore = (country.revenuePerView / 2.5) * 25; // Max 25 points
    const engagementScore = (country.avgEngagement / 100) * 20; // Max 20 points
    const growthScore = (country.growthRate / 15) * 25; // Max 25 points
    const marketSizeScore = (Math.log10(country.marketSize) / Math.log10(1000000000)) * 15; // Max 15 points
    const competitionScore = ((100 - country.competitionLevel) / 100) * 15; // Max 15 points
    
    return {
      country: country.name,
      score: Math.round(revenueScore + engagementScore + growthScore + marketSizeScore + competitionScore),
      revenueScore,
      engagementScore,
      growthScore,
      marketSizeScore,
      competitionScore
    };
  }).sort((a, b) => b.score - a.score);
  
  return (
    <div className="opportunity-score-container">
      <h3>Opportunity Score</h3>
      <div className="opportunity-score-table">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Country</th>
              <th>Score</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {opportunityScores.map((item, index) => (
              <tr key={item.country}>
                <td>#{index + 1}</td>
                <td>{item.country}</td>
                <td>
                  <div className="score-pill">
                    <span>{item.score}/100</span>
                  </div>
                </td>
                <td>
                  <div className="score-breakdown">
                    <div className="breakdown-item">
                      <span className="breakdown-label">Revenue:</span> 
                      <span className="breakdown-value">{Math.round(item.revenueScore)}</span>
                    </div>
                    <div className="breakdown-item">
                      <span className="breakdown-label">Engagement:</span> 
                      <span className="breakdown-value">{Math.round(item.engagementScore)}</span>
                    </div>
                    <div className="breakdown-item">
                      <span className="breakdown-label">Growth:</span> 
                      <span className="breakdown-value">{Math.round(item.growthScore)}</span>
                    </div>
                    <div className="breakdown-item">
                      <span className="breakdown-label">Market:</span> 
                      <span className="breakdown-value">{Math.round(item.marketSizeScore)}</span>
                    </div>
                    <div className="breakdown-item">
                      <span className="breakdown-label">Competition:</span> 
                      <span className="breakdown-value">{Math.round(item.competitionScore)}</span>
                    </div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="opportunity-score-explanation">
        <p>
          <strong>Opportunity Score</strong> is calculated as a weighted combination of:
        </p>
        <ul>
          <li>Revenue per view (25%)</li>
          <li>Engagement rate (20%)</li>
          <li>Growth rate (25%)</li>
          <li>Market size (15%)</li>
          <li>Competition level (15%)</li>
        </ul>
        <p>Higher scores suggest better opportunities for market entry and growth.</p>
      </div>
    </div>
  );
}

// Main component
export default function MultiCountryAnalysis() {
  const [selectedCountryIds, setSelectedCountryIds] = useState<string[]>([]);
  const [baseViews, setBaseViews] = useState<number>(100000);
  const [activeView, setActiveView] = useState<string>('revenue');
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  
  // Get selected country objects
  const selectedCountries = AVAILABLE_COUNTRIES.filter(country => 
    selectedCountryIds.includes(country.id)
  );
  
  // Handle country selection change
  const handleSelectionChange = (newSelectedIds: string[]) => {
    setSelectedCountryIds(newSelectedIds);
  };
  
  // Handle base views change
  const handleViewsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBaseViews(Number(e.target.value));
  };
  
  return (
    <div className="multi-country-analysis">
      <h2>Multi-Country Analysis</h2>
      
      <div className="control-panel">
        <div className="country-selector">
          <h3>Country Selection</h3>
          <div className="selected-countries-display">
            {selectedCountries.length > 0 ? (
              <div className="selected-countries-grid">
                {selectedCountries.map(country => (
                  <div key={country.id} className="selected-country-pill">
                    {country.name}
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-countries-selected">
                No countries selected
              </div>
            )}
          </div>
          <button 
            className="select-countries-button" 
            onClick={() => setIsModalOpen(true)}
          >
            Select Countries
          </button>
          
          {/* Country Selection Modal */}
          <CountrySelectionModal 
            isOpen={isModalOpen}
            onClose={() => setIsModalOpen(false)}
            availableCountries={AVAILABLE_COUNTRIES}
            selectedCountryIds={selectedCountryIds}
            onSelectionChange={handleSelectionChange}
          />
        </div>
        
        <div className="base-views-control">
          <h3>Base Views</h3>
          <input
            type="range"
            min="10000"
            max="1000000"
            step="10000"
            value={baseViews}
            onChange={handleViewsChange}
          />
          <span>{baseViews.toLocaleString()} views</span>
        </div>
      </div>
      
      {selectedCountries.length > 0 ? (
        <div className="analysis-view">
          <div className="view-selector">
            <button 
              className={activeView === 'revenue' ? 'active' : ''}
              onClick={() => setActiveView('revenue')}
            >
              Revenue Comparison
            </button>
            <button 
              className={activeView === 'metrics' ? 'active' : ''}
              onClick={() => setActiveView('metrics')}
            >
              Metrics Radar
            </button>
            <button 
              className={activeView === 'content' ? 'active' : ''}
              onClick={() => setActiveView('content')}
            >
              Content Preference
            </button>
            <button 
              className={activeView === 'opportunity' ? 'active' : ''}
              onClick={() => setActiveView('opportunity')}
            >
              Opportunity Score
            </button>
          </div>
          
          <div className="analysis-container">
            {activeView === 'revenue' && <RevenueComparison selectedCountries={selectedCountries} baseViews={baseViews} />}
            {activeView === 'metrics' && <MetricsRadar selectedCountries={selectedCountries} />}
            {activeView === 'content' && <ContentPreference selectedCountries={selectedCountries} />}
            {activeView === 'opportunity' && <OpportunityScore selectedCountries={selectedCountries} />}
          </div>
        </div>
      ) : (
        <div className="no-selection-message">
          <p>Please select at least one country to see the analysis.</p>
        </div>
      )}
      
      <style jsx>{`
        .multi-country-analysis {
          padding: 1.5rem;
          background-color: #f9f9f9;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        h2 {
          color: #333;
          margin-bottom: 1.5rem;
          font-size: 1.8rem;
        }
        
        .control-panel {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          margin-bottom: 2rem;
          padding: 1rem;
          background-color: white;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .country-selector h3, .base-views-control h3 {
          margin-bottom: 0.8rem;
          font-size: 1.2rem;
          color: #444;
        }
        
        .selected-countries-display {
          min-height: 60px;
          border: 1px dashed #ccc;
          border-radius: 4px;
          padding: 0.8rem;
          margin-bottom: 0.8rem;
          background-color: #fafafa;
        }
        
        .selected-countries-grid {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }
        
        .selected-country-pill {
          background-color: #4a6fa5;
          color: white;
          padding: 0.4rem 0.8rem;
          border-radius: 20px;
          font-size: 0.9rem;
        }
        
        .no-countries-selected {
          color: #888;
          font-style: italic;
          font-size: 0.9rem;
          display: flex;
          align-items: center;
          justify-content: center;
          height: 40px;
        }
        
        .select-countries-button {
          width: 100%;
          padding: 0.7rem;
          background-color: #4a6fa5;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-weight: 500;
        }
        
        .select-countries-button:hover {
          background-color: #3d5d8a;
        }
        
        .base-views-control {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
        }
        
        .base-views-control input {
          width: 100%;
          margin: 0.5rem 0;
        }
        
        .view-selector {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
          overflow-x: auto;
          padding-bottom: 0.5rem;
        }
        
        .view-selector button {
          padding: 0.7rem 1.2rem;
          border: none;
          border-radius: 4px;
          background-color: #eef2f7;
          cursor: pointer;
          white-space: nowrap;
          transition: all 0.2s;
        }
        
        .view-selector button:hover {
          background-color: #dbe4f0;
        }
        
        .view-selector button.active {
          background-color: #4a6fa5;
          color: white;
        }
        
        .no-selection-message {
          background-color: white;
          padding: 2rem;
          text-align: center;
          border-radius: 8px;
          color: #666;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        @media (min-width: 768px) {
          .control-panel {
            flex-direction: row;
          }
          
          .country-selector {
            flex: 3;
          }
          
          .base-views-control {
            flex: 1;
            min-width: 200px;
          }
        }
        
        .chart-container {
          background-color: white;
          padding: 1.5rem;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .chart-container h3 {
          margin-bottom: 1rem;
          font-size: 1.3rem;
          color: #333;
        }
        
        .radar-chart-container {
          max-width: 500px;
          margin: 0 auto;
        }
        
        .metrics-legend, .content-preference-legend, .opportunity-score-explanation {
          margin-top: 1.5rem;
          padding-top: 1rem;
          border-top: 1px solid #eee;
          font-size: 0.9rem;
          color: #666;
        }
        
        .metrics-legend p {
          margin: 0.3rem 0;
        }
        
        .opportunity-score-container {
          background-color: white;
          padding: 1.5rem;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .opportunity-score-table table {
          width: 100%;
          border-collapse: collapse;
          margin: 1rem 0 1.5rem;
        }
        
        .opportunity-score-table th, .opportunity-score-table td {
          padding: 0.8rem;
          text-align: left;
          border-bottom: 1px solid #eee;
        }
        
        .opportunity-score-table th {
          font-weight: 600;
          color: #555;
        }
        
        .score-pill {
          background-color: #4a6fa5;
          color: white;
          padding: 0.3rem 0.6rem;
          border-radius: 12px;
          font-weight: 600;
          display: inline-block;
        }
        
        .score-breakdown {
          display: flex;
          gap: 0.8rem;
          flex-wrap: wrap;
        }
        
        .breakdown-item {
          font-size: 0.85rem;
          background-color: #f5f7fa;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
        }
        
        .breakdown-label {
          font-weight: 600;
          color: #666;
        }
        
        .breakdown-value {
          margin-left: 0.3rem;
        }
      `}</style>
    </div>
  );
} 