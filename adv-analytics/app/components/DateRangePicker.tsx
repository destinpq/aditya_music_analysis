'use client';

import React, { useState } from 'react';

interface DateRangePickerProps {
  onDateRangeChange: (startDate: Date | null, endDate: Date | null) => void;
  className?: string;
}

export default function DateRangePicker({ onDateRangeChange, className = '' }: DateRangePickerProps) {
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');

  const handleStartDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setStartDate(e.target.value);
    onDateRangeChange(
      e.target.value ? new Date(e.target.value) : null,
      endDate ? new Date(endDate) : null
    );
  };

  const handleEndDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEndDate(e.target.value);
    onDateRangeChange(
      startDate ? new Date(startDate) : null,
      e.target.value ? new Date(e.target.value) : null
    );
  };

  const clearDates = () => {
    setStartDate('');
    setEndDate('');
    onDateRangeChange(null, null);
  };

  return (
    <div className={`date-range-picker ${className}`}>
      <div className="date-range-field">
        <label htmlFor="start-date" className="date-range-label">
          Start Date
        </label>
        <input
          type="date"
          id="start-date"
          value={startDate}
          onChange={handleStartDateChange}
          className="date-range-input"
        />
      </div>
      
      <div className="date-range-field">
        <label htmlFor="end-date" className="date-range-label">
          End Date
        </label>
        <input
          type="date"
          id="end-date"
          value={endDate}
          min={startDate}
          onChange={handleEndDateChange}
          className="date-range-input"
        />
      </div>

      {(startDate || endDate) && (
        <div className="date-range-actions">
          <button
            type="button"
            onClick={clearDates}
            className="button button-small button-secondary"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
} 