import { useState, useEffect } from 'react';
import axios from 'axios';

/**
 * React hook to access configuration values
 * This will fetch the config from the database via an API endpoint
 */
export function useConfig(key: string, defaultValue: string = ''): [string, boolean] {
  const [value, setValue] = useState<string>(defaultValue);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`/api/config?key=${encodeURIComponent(key)}`);
        if (response.data && response.data.value) {
          setValue(response.data.value);
        }
      } catch (error) {
        console.error(`Error fetching config ${key}:`, error);
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, [key]);

  return [value, loading];
}

/**
 * Get a public configuration value (available on client side)
 * This is for configs that must be available during build time
 */
export function getPublicConfig(key: string, defaultValue: string = ''): string {
  // Check if we have a public variable
  const publicKey = `NEXT_PUBLIC_${key}`;
  const publicValue = typeof window !== 'undefined' 
    ? (window as any).__NEXT_PUBLIC_CONFIG__?.[key] 
    : process.env[publicKey];

  return publicValue || defaultValue;
}

/**
 * Set config value
 */
export async function setConfigValue(key: string, value: string): Promise<boolean> {
  try {
    await axios.post('/api/config', { key, value });
    return true;
  } catch (error) {
    console.error(`Error setting config ${key}:`, error);
    return false;
  }
} 