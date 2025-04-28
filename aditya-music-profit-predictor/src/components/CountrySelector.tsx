import { useState } from 'react';

interface CountrySelectorProps {
  onSelect: (country: string) => void;
}

const COUNTRIES = [
  { name: 'India', cpm: 120 },
  { name: 'USA', cpm: 600 },
  { name: 'UK', cpm: 500 },
  { name: 'Canada', cpm: 450 },
  { name: 'Australia', cpm: 450 }
];

export default function CountrySelector({ onSelect }: CountrySelectorProps) {
  const [selectedCountry, setSelectedCountry] = useState<string>('');

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const country = e.target.value;
    setSelectedCountry(country);
    onSelect(country);
  };

  return (
    <div className="w-full max-w-xs">
      <label htmlFor="country-select" className="block text-sm font-medium mb-2">
        Select Target Country
      </label>
      <select
        id="country-select"
        value={selectedCountry}
        onChange={handleChange}
        className="block w-full p-2 border border-gray-300 rounded-md shadow-sm bg-white text-gray-900 focus:ring-blue-500 focus:border-blue-500"
      >
        <option value="" disabled>Choose a country</option>
        {COUNTRIES.map(country => (
          <option key={country.name} value={country.name}>
            {country.name} (CPM: ₹{country.cpm})
          </option>
        ))}
      </select>
    </div>
  );
} 