import { Pool } from 'pg';
import { config } from 'dotenv';

// Load environment variables for database connection
config();

// Connection pool
const pool = new Pool({
  user: process.env.DB_USERNAME,
  password: process.env.DB_PASSWORD,
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME,
  ssl: {
    rejectUnauthorized: false,
    require: process.env.DB_SSLMODE === 'require'
  }
});

// Configuration table schema
// CREATE TABLE IF NOT EXISTS app_config (
//   key VARCHAR(255) PRIMARY KEY,
//   value TEXT NOT NULL,
//   description TEXT,
//   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
//   updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
// );

/**
 * Get a configuration value from the database
 */
export async function getConfig(key: string, defaultValue: string = ''): Promise<string> {
  try {
    const result = await pool.query('SELECT value FROM app_config WHERE key = $1', [key]);
    if (result.rows.length > 0) {
      return result.rows[0].value;
    }
    return defaultValue;
  } catch (error) {
    console.error('Error fetching config from database:', error);
    return defaultValue;
  }
}

/**
 * Set a configuration value in the database
 */
export async function setConfig(key: string, value: string, description: string = ''): Promise<boolean> {
  try {
    await pool.query(
      'INSERT INTO app_config (key, value, description) VALUES ($1, $2, $3) ' +
      'ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = CURRENT_TIMESTAMP',
      [key, value, description]
    );
    return true;
  } catch (error) {
    console.error('Error setting config in database:', error);
    return false;
  }
}

/**
 * Initialize database connection and check for configuration table
 */
export async function initConfigDb(): Promise<void> {
  try {
    // Create config table if it doesn't exist
    await pool.query(`
      CREATE TABLE IF NOT EXISTS app_config (
        key VARCHAR(255) PRIMARY KEY,
        value TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    console.log('Configuration database initialized');
  } catch (error) {
    console.error('Error initializing config database:', error);
  }
} 