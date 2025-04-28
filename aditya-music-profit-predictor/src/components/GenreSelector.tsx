import { useState } from 'react';

interface GenreSelectorProps {
  onSelect: (genre: string) => void;
}

const GENRES = [
  'Romantic',
  'Classical',
  'Folk',
  'Devotional',
  'Pop',
  'Melody',
  'Item Song',
  'Dance',
  'Hip Hop'
];

export default function GenreSelector({ onSelect }: GenreSelectorProps) {
  const [selectedGenre, setSelectedGenre] = useState<string>('');

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const genre = e.target.value;
    setSelectedGenre(genre);
    onSelect(genre);
  };

  return (
    <div className="w-full max-w-xs">
      <label htmlFor="genre-select" className="block text-sm font-medium mb-2">
        Select Genre
      </label>
      <select
        id="genre-select"
        value={selectedGenre}
        onChange={handleChange}
        className="block w-full p-2 border border-gray-300 rounded-md shadow-sm bg-white text-gray-900 focus:ring-blue-500 focus:border-blue-500"
      >
        <option value="" disabled>Choose a genre</option>
        {GENRES.map(genre => (
          <option key={genre} value={genre}>
            {genre}
          </option>
        ))}
      </select>
    </div>
  );
} 