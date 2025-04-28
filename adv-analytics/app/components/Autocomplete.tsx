'use client';

import React, { useState, useEffect, useRef } from 'react';

interface AutocompleteProps {
  suggestions: string[];
  placeholder?: string;
  onSelect: (selection: string) => void;
  className?: string;
}

const Autocomplete: React.FC<AutocompleteProps> = ({
  suggestions,
  placeholder = 'Search...',
  onSelect,
  className = '',
}) => {
  const [inputValue, setInputValue] = useState('');
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLUListElement>(null);

  // Filter suggestions based on input
  useEffect(() => {
    if (inputValue) {
      const filtered = suggestions.filter(
        suggestion => suggestion.toLowerCase().includes(inputValue.toLowerCase())
      );
      setFilteredSuggestions(filtered);
    } else {
      setFilteredSuggestions([]);
    }
    setActiveSuggestionIndex(0);
  }, [inputValue, suggestions]);

  // Handle clicks outside the component
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        inputRef.current && 
        suggestionsRef.current && 
        !inputRef.current.contains(event.target as Node) && 
        !suggestionsRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
    setShowSuggestions(true);
  };

  const handleSelect = (suggestion: string) => {
    setInputValue(suggestion);
    setShowSuggestions(false);
    onSelect(suggestion);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Enter key
    if (e.key === 'Enter') {
      if (filteredSuggestions.length > 0 && showSuggestions) {
        handleSelect(filteredSuggestions[activeSuggestionIndex]);
      } else {
        onSelect(inputValue);
      }
    }
    // Arrow down
    else if (e.key === 'ArrowDown') {
      if (filteredSuggestions.length > 0) {
        setActiveSuggestionIndex(prevIndex => 
          prevIndex < filteredSuggestions.length - 1 ? prevIndex + 1 : prevIndex
        );
      }
    }
    // Arrow up
    else if (e.key === 'ArrowUp') {
      if (filteredSuggestions.length > 0) {
        setActiveSuggestionIndex(prevIndex => 
          prevIndex > 0 ? prevIndex - 1 : 0
        );
      }
    }
    // Escape
    else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  return (
    <div className={`autocomplete ${className}`}>
      <div className="autocomplete-input-container">
        <input
          type="text"
          className="autocomplete-input"
          value={inputValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(true)}
          placeholder={placeholder}
          ref={inputRef}
        />
        {inputValue && (
          <button 
            className="autocomplete-clear"
            onClick={() => {
              setInputValue('');
              setShowSuggestions(false);
              inputRef.current?.focus();
            }}
            aria-label="Clear input"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18M6 6l12 12"></path>
            </svg>
          </button>
        )}
      </div>
      
      {showSuggestions && filteredSuggestions.length > 0 && (
        <ul className="autocomplete-suggestions" ref={suggestionsRef}>
          {filteredSuggestions.map((suggestion, index) => (
            <li
              key={suggestion}
              className={`autocomplete-suggestion ${index === activeSuggestionIndex ? 'active' : ''}`}
              onClick={() => handleSelect(suggestion)}
              onMouseEnter={() => setActiveSuggestionIndex(index)}
            >
              {suggestion}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Autocomplete; 