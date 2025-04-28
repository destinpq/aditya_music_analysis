'use client';

import { useState, useEffect, useRef } from 'react';

// Define a generic type for the data items
interface SearchableItem {
  [key: string]: string | number | boolean | null | undefined;
}

interface AutocompleteSearchProps<T extends SearchableItem> {
  data: T[];
  searchKeys: string[];
  placeholder?: string;
  onSelect: (item: T) => void;
  renderItem?: (item: T) => React.ReactNode;
  minSearchLength?: number;
  maxResults?: number;
  className?: string;
}

export default function AutocompleteSearch<T extends SearchableItem>({
  data,
  searchKeys,
  placeholder = 'Search...',
  onSelect,
  renderItem,
  minSearchLength = 2,
  maxResults = 10,
  className = ''
}: AutocompleteSearchProps<T>) {
  const [searchTerm, setSearchTerm] = useState('');
  const [suggestions, setSuggestions] = useState<T[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (searchTerm.length < minSearchLength) {
      setSuggestions([]);
      setIsOpen(false);
      return;
    }

    const normalizedSearchTerm = searchTerm.toLowerCase();
    const filteredData = data.filter(item => {
      return searchKeys.some(key => {
        const value = item[key];
        return value !== null && value !== undefined && value.toString().toLowerCase().includes(normalizedSearchTerm);
      });
    }).slice(0, maxResults);

    setSuggestions(filteredData);
    setIsOpen(filteredData.length > 0);
    setSelectedIndex(-1);
  }, [searchTerm, data, searchKeys, minSearchLength, maxResults]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  const handleSelect = (item: T) => {
    onSelect(item);
    setSearchTerm('');
    setIsOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) return;

    // Arrow down
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => (
        prev < suggestions.length - 1 ? prev + 1 : prev
      ));
    }
    // Arrow up
    else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => (prev > 0 ? prev - 1 : prev));
    }
    // Enter
    else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault();
      handleSelect(suggestions[selectedIndex]);
    }
    // Escape
    else if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        suggestionsRef.current && 
        !suggestionsRef.current.contains(e.target as Node) && 
        inputRef.current && 
        !inputRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Default item renderer
  const defaultRenderItem = (item: T) => {
    const firstKey = searchKeys[0];
    return <span>{String(item[firstKey] || '')}</span>;
  };

  return (
    <div className={`autocomplete-search ${className}`}>
      <input
        ref={inputRef}
        type="text"
        value={searchTerm}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="autocomplete-input"
        onFocus={() => searchTerm.length >= minSearchLength && suggestions.length > 0 && setIsOpen(true)}
      />
      
      {isOpen && (
        <div className="autocomplete-suggestions" ref={suggestionsRef}>
          {suggestions.length > 0 ? (
            <ul className="suggestions-list">
              {suggestions.map((item, index) => (
                <li
                  key={index}
                  className={`suggestion-item ${selectedIndex === index ? 'selected' : ''}`}
                  onClick={() => handleSelect(item)}
                  onMouseEnter={() => setSelectedIndex(index)}
                >
                  {renderItem ? renderItem(item) : defaultRenderItem(item)}
                </li>
              ))}
            </ul>
          ) : (
            <div className="no-suggestions">No results found</div>
          )}
        </div>
      )}
    </div>
  );
} 