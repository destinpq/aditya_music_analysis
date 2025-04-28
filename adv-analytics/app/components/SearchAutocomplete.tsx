'use client';

import React, { useState, useEffect, useRef } from 'react';

interface SearchOption {
  id: string | number;
  label: string;
}

interface SearchAutocompleteProps {
  options: SearchOption[];
  onSelect: (option: SearchOption) => void;
  placeholder?: string;
  className?: string;
}

export default function SearchAutocomplete({
  options,
  onSelect,
  placeholder = 'Search...',
  className = '',
}: SearchAutocompleteProps) {
  const [inputValue, setInputValue] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [filteredOptions, setFilteredOptions] = useState<SearchOption[]>([]);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Filter options based on input value
    const filtered = options.filter(option =>
      option.label.toLowerCase().includes(inputValue.toLowerCase())
    );
    setFilteredOptions(filtered);
  }, [inputValue, options]);

  useEffect(() => {
    // Handle clicks outside of the component to close dropdown
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInputValue(value);
    if (value) {
      setIsOpen(true);
    } else {
      setIsOpen(false);
    }
  };

  const handleOptionClick = (option: SearchOption) => {
    onSelect(option);
    setInputValue(option.label);
    setIsOpen(false);
  };

  return (
    <div className={`search-autocomplete-wrapper ${className}`} ref={wrapperRef}>
      <div className="search-input-container">
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onClick={() => inputValue && setIsOpen(true)}
          placeholder={placeholder}
          className="search-input"
        />
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="search-icon"
        >
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      </div>
      
      {isOpen && filteredOptions.length > 0 && (
        <ul className="options-dropdown">
          {filteredOptions.map((option) => (
            <li
              key={option.id}
              onClick={() => handleOptionClick(option)}
              className="option-item"
            >
              {option.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
} 