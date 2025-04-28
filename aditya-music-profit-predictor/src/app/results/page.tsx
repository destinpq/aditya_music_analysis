'use client';

import { useEffect, useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import ResultCard from '@/components/ResultCard';
import RevenueGraph from '@/components/RevenueGraph';

interface PredictionResult {
  views: number;
  revenue: number;
  engagement: string;
  videoType: string;
}

function ResultsContent() {
  const searchParams = useSearchParams();
  const genre = searchParams.get('genre') || '';
  const country = searchParams.get('country') || '';
  
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    async function fetchPrediction() {
      if (!genre || !country) {
        setError('Missing required parameters');
        setLoading(false);
        return;
      }

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
      } catch (err) {
        console.error('Error:', err);
        setError('Failed to generate prediction. Please try again.');
      } finally {
        setLoading(false);
      }
    }

    fetchPrediction();
  }, [genre, country]);

  if (loading) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-b from-gray-900 to-gray-800 text-white">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4">Analyzing Music Data...</h1>
          <p className="text-lg mb-4">Predicting revenue for {genre} music in {country}</p>
          <div className="w-16 h-16 border-t-4 border-blue-500 border-solid rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-b from-gray-900 to-gray-800 text-white">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4">Error</h1>
          <p className="text-lg mb-8">{error || 'Something went wrong'}</p>
          <Link href="/predict" className="inline-block px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition duration-300">
            Try Again
          </Link>
        </div>
      </div>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-8 md:p-24 bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      <div className="w-full max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">
          Profit Prediction Results
        </h1>
        
        <div className="mb-4 text-center">
          <div className="inline-block px-4 py-2 bg-gray-800 rounded-lg mb-8">
            <span className="font-medium mr-2">Genre:</span> {genre}
            <span className="mx-4">|</span>
            <span className="font-medium mr-2">Country:</span> {country}
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="flex justify-center">
            <ResultCard 
              views={result.views}
              revenue={result.revenue}
              engagement={result.engagement}
              videoType={result.videoType}
            />
          </div>
          
          <div className="flex justify-center">
            <RevenueGraph 
              country={country}
              revenue={result.revenue}
            />
          </div>
        </div>
        
        <div className="mt-12 text-center">
          <Link href="/predict" className="inline-block px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition duration-300">
            Make Another Prediction
          </Link>
        </div>
      </div>
    </main>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-b from-gray-900 to-gray-800 text-white">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4">Loading Results...</h1>
          <div className="w-16 h-16 border-t-4 border-blue-500 border-solid rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    }>
      <ResultsContent />
    </Suspense>
  );
} 