# Database-Based Configuration System

This project has been updated to store configuration and secrets in a database instead of environment variables or .env files. This approach provides better security and centralized management of application settings.

## How It Works

1. Configuration values are stored in a PostgreSQL database table called `app_config`
2. The application accesses these values through API endpoints or the database directly (server-side)
3. This eliminates the need for environment variables containing sensitive information

## Setup Instructions

### 1. Initial Migration

To migrate your existing .env file to the database:

```bash
# Install dependencies
npm install

# Run the migration script
npm run migrate-config
```

This will:
1. Read your .env file
2. Create the necessary database table if it doesn't exist
3. Store all values in the database with their descriptions

### 2. Accessing Configuration Values

#### In React Components (Client-Side)

```tsx
import { useConfig } from '../utils/use-config';

function MyComponent() {
  // Get a configuration value with a default fallback
  const [apiBaseUrl, loading] = useConfig('API_BASE_URL', '/api');
  
  if (loading) {
    return <div>Loading...</div>;
  }
  
  return <div>API URL: {apiBaseUrl}</div>;
}
```

#### In API Routes or Server Components

```tsx
import { getConfig } from '../utils/config-db';

export async function GET() {
  const apiKey = await getConfig('API_KEY');
  
  // Use the value...
}
```

### 3. Setting Configuration Values

You can set values programmatically:

```tsx
import { setConfigValue } from '../utils/use-config';

// In an admin component or setup script
await setConfigValue('API_KEY', 'new-api-key-value');
```

Or, build an admin interface using the API endpoint at `/api/config`.

## Security Considerations

- The database should be properly secured with strong credentials
- Access to the configuration API should be restricted to authorized users
- API keys and credentials should only be accessed server-side when possible

## Benefits Over .env Files

- Centralized management of configuration
- No environment variables to maintain across deployments
- Ability to update values at runtime
- Better audit trail of configuration changes
- Reduced risk of accidentally committing sensitive values 