import { config } from 'dotenv';
import { Pool } from 'pg';
import fs from 'fs';
import path from 'path';
import { initConfigDb, setConfig } from '../app/utils/config-db';

// Load environment variables
config();

async function migrateEnvToDb() {
  console.log('Starting migration of environment variables to database...');
  
  // Initialize database and create config table if needed
  await initConfigDb();
  
  // Read .env file
  const envPath = path.resolve(process.cwd(), '../.env');
  if (!fs.existsSync(envPath)) {
    console.error('.env file not found at:', envPath);
    return;
  }
  
  const envContent = fs.readFileSync(envPath, 'utf8');
  const envLines = envContent.split('\n');
  
  // Parse env file
  const configEntries: Array<{ key: string; value: string; description: string }> = [];
  let currentDescription = '';
  
  for (const line of envLines) {
    const trimmedLine = line.trim();
    
    // Skip empty lines
    if (!trimmedLine) continue;
    
    // Comments become descriptions for the next variable
    if (trimmedLine.startsWith('#')) {
      currentDescription = trimmedLine.substring(1).trim();
      continue;
    }
    
    // Parse variable
    const match = trimmedLine.match(/^([A-Za-z0-9_]+)=(.*)$/);
    if (match) {
      const [, key, value] = match;
      configEntries.push({
        key,
        value,
        description: currentDescription
      });
      currentDescription = '';
    }
  }
  
  // Store variables in database
  console.log(`Found ${configEntries.length} variables to migrate`);
  
  for (const entry of configEntries) {
    console.log(`Migrating ${entry.key}...`);
    await setConfig(entry.key, entry.value, entry.description);
  }
  
  console.log('Migration completed successfully');
}

// Run the migration
migrateEnvToDb().catch(error => {
  console.error('Migration failed:', error);
  process.exit(1);
}); 