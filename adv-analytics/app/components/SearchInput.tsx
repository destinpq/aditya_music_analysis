'use client';

import React, { useState, useEffect, useRef } from 'react';
// We'll define our own search icon to avoid dependency on lucide-react

interface SearchInputProps {
  placeholder?: string;
  onSearch: (value: string) => void;
  suggestions?: string[];
  className?: string;
  initialValue?: string;
}

export default function SearchInput({
  placeholder = 'Search...',
  onSearch,
  suggestions = [],
  className = '',
  initialValue = '',
}: SearchInputProps) {
  const [searchTerm, setSearchTerm] = useState(initialValue);
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Filter suggestions based on search term
    if (searchTerm) {
      const filtered = suggestions.filter(suggestion =>
        suggestion.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredSuggestions(filtered);
    } else {
      setFilteredSuggestions([]);
    }
  }, [searchTerm, suggestions]);

  useEffect(() => {
    // Handle clicks outside of the component to close suggestions
    function handleClickOutside(event: MouseEvent) {
      if (
        inputRef.current &&
        !inputRef.current.contains(event.target as Node) &&
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchTerm(value);
    setShowSuggestions(true);
    setActiveIndex(-1);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Handle keyboard navigation
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex(prev => 
        prev < filteredSuggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex(prev => (prev > 0 ? prev - 1 : prev));
    } else if (e.key === 'Enter') {
      if (activeIndex >= 0 && filteredSuggestions[activeIndex]) {
        selectSuggestion(filteredSuggestions[activeIndex]);
      } else {
        handleSearch();
      }
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  const selectSuggestion = (suggestion: string) => {
    setSearchTerm(suggestion);
    setShowSuggestions(false);
    onSearch(suggestion);
  };

  const handleSearch = () => {
    onSearch(searchTerm);
    setShowSuggestions(false);
  };

  return (
    <div className={`autocomplete-search ${className}`}>
      <div className="search-form">
        <input
          ref={inputRef}
          type="text"
          className="autocomplete-input"
          placeholder={placeholder}
          value={searchTerm}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(true)}
        />
        <button 
          type="button" 
          className="search-button"
          onClick={handleSearch}
          aria-label="Search"
        >
          {/* Custom search icon SVG */}
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </button>
      </div>

      {showSuggestions && filteredSuggestions.length > 0 && (
        <div 
          ref={suggestionsRef} 
          className="autocomplete-suggestions"
        >
          <ul className="suggestions-list">
            {filteredSuggestions.map((suggestion, index) => (
              <li
                key={index}
                className={`suggestion-item ${index === activeIndex ? 'selected' : ''}`}
                onClick={() => selectSuggestion(suggestion)}
                onMouseEnter={() => setActiveIndex(index)}
              >
                {suggestion}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
} 