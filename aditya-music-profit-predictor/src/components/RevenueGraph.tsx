import React from 'react';
import { Bar } from 'react-chartjs-2';
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
  ChartData,
  ChartOptions,
  TooltipItem
} from 'chart.js';

// Register ChartJS components
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

interface RevenueGraphProps {
  country: string;
  revenue: number;
  selectedCountries?: string[]; // Add optional parameter for selected countries
}

export default function RevenueGraph({ country, revenue, selectedCountries = [] }: RevenueGraphProps) {
  // Generate comparison data for other countries based on CPM ratios
  const COUNTRY_CPM: Record<string, number> = {
    'India': 120,
    'USA': 600,
    'UK': 500,
    'Canada': 450,
    'Australia': 450,
  };

  // Include the current country in the selection if not already included
  const allSelectedCountries = [...new Set([country, ...selectedCountries])];

  // Generate comparison data for all countries
  const graphData = Object.entries(COUNTRY_CPM)
    .filter(([countryName]) => 
      // If there are selected countries, only include those, otherwise include all
      selectedCountries.length === 0 || allSelectedCountries.includes(countryName)
    )
    .map(([countryName, cpm]) => {
      const ratio = cpm / COUNTRY_CPM[country];
      return {
        country: countryName,
        revenue: countryName === country ? revenue : Math.round(revenue * ratio)
      };
    });

  // Sort by revenue descending
  graphData.sort((a, b) => b.revenue - a.revenue);

  const data: ChartData<'bar'> = {
    labels: graphData.map(item => item.country),
    datasets: [
      {
        label: 'Estimated Revenue (₹)',
        data: graphData.map(item => item.revenue),
        backgroundColor: graphData.map(item => 
          item.country === country ? 'rgba(53, 162, 235, 0.8)' : 'rgba(75, 192, 192, 0.6)'
        ),
        borderColor: graphData.map(item => 
          item.country === country ? 'rgb(53, 162, 235)' : 'rgb(75, 192, 192)'
        ),
        borderWidth: 1,
      },
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
        text: selectedCountries.length > 0 
          ? 'Revenue Comparison - Selected Countries' 
          : 'Revenue Comparison by Country',
      },
      tooltip: {
        callbacks: {
          label: function(context: TooltipItem<'bar'>) {
            const label = context.dataset.label || '';
            const value = context.raw as number;
            return `${label}: ₹${value.toLocaleString('en-IN')}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(
            tickValue: string | number
          ): string {
            const value = Number(tickValue);
            if (value >= 100000) {
              return `₹${(value / 100000).toFixed(1)}L`;
            }
            if (value >= 1000) {
              return `₹${(value / 1000).toFixed(1)}K`;
            }
            return `₹${value}`;
          }
        }
      }
    },
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-2xl">
      <Bar data={data} options={options} />
    </div>
  );
} 