'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import GenreSelector from '@/components/GenreSelector';
import CountrySelector from '@/components/CountrySelector';

export default function PredictPage() {
  const router = useRouter();
  const [genre, setGenre] = useState<string>('');
  const [country, setCountry] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  const handleGenreChange = (selectedGenre: string) => {
    setGenre(selectedGenre);
  };

  const handleCountryChange = (selectedCountry: string) => {
    setCountry(selectedCountry);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!genre || !country) {
      alert('Please select both a genre and a country.');
      return;
    }

    setLoading(true);
    
    try {
      // Redirect to results page with query parameters
      router.push(`/results?genre=${encodeURIComponent(genre)}&country=${encodeURIComponent(country)}`);
    } catch (error) {
      console.error('Error:', error);
      setLoading(false);
      alert('An error occurred. Please try again.');
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      <div className="w-full max-w-md bg-gray-800 rounded-lg shadow-lg p-8">
        <h1 className="text-3xl font-bold text-center mb-8">
          Predict Your Music Profit
        </h1>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-4">
            <GenreSelector onSelect={handleGenreChange} />
            <CountrySelector onSelect={handleCountryChange} />
          </div>
          
          <button 
            type="submit"
            disabled={loading || !genre || !country}
            className="w-full py-3 mt-6 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Processing...' : 'Predict Profit'}
          </button>
        </form>
      </div>
    </main>
  );
} 