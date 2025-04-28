# YouTube Analytics Dashboard

An advanced YouTube analytics dashboard built with Next.js that connects to the ML-powered backend APIs to provide insights, predictions, and data analysis for YouTube creators.

## Features

- Modern UI with smooth animations and transitions
- YouTube data analysis and visualization
- ML model training and management
- Prediction capabilities for video performance
- Earnings estimations and analytics
- Dataset management
- Secure database-based configuration storage (no environment variables)

## Backend Integration

This frontend application connects to the Backend API which provides:

- Dataset management
- Machine Learning model training
- Predictions and analytics
- Feature engineering
- Hyperparameter tuning
- TensorFlow model support

## Getting Started

### Prerequisites

- Node.js 18.x or later
- npm or yarn

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/youtube-analytics-dashboard.git
cd youtube-analytics-dashboard
```

2. Install dependencies:

```bash
npm install
# or
yarn install
```

3. Set up environment variables:
   
Create a `.env.local` file in the root directory with the following content:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
```

Replace the URL with your Backend API endpoint.

4. Start the development server:

```bash
npm run dev
# or
yarn dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Connecting to the Backend API

To connect to the backend API, ensure that the Backend server is running. The frontend application communicates with the backend through APIs defined in `app/services/api.ts`.

### Backend API Configuration

The Backend API needs to be running to use all features of this dashboard. By default, the application will try to connect to `http://localhost:8080`, but you can change this by updating the `.env.local` file.

### API Endpoints

The application integrates with the following key API endpoints:

- **Dataset Management**: Upload, process, and delete datasets
- **ML Model Training**: Train custom machine learning models
- **Predictions**: Predict video performance based on trained models
- **Analytics**: Get data-driven insights from your videos
- **Feature Engineering**: Process and transform raw data into features

## ML Lab

The ML Lab section provides a comprehensive interface for working with machine learning models:

1. **Train Models**: Select datasets and train various ML models
2. **Tune Models**: Perform hyperparameter tuning
3. **Make Predictions**: Use trained models to predict video performance
4. **Process Datasets**: Prepare data for machine learning

## Development

The codebase follows Next.js 13+ App Router conventions:

- `app/`: Contains all pages and components
- `app/components/`: Reusable UI components
- `app/services/`: API integration and data fetching
- `app/ml-lab/`: Machine learning interface
- `app/dashboard/`: Analytics dashboard

## Building for Production

To build the application for production:

```bash
npm run build
# or
yarn build
```

To start the production server:

```bash
npm start
# or
yarn start
```

## Environment Variables

The application now uses a database-based configuration system rather than environment variables for improved security. See the [Database Configuration Documentation](./docs/DB-CONFIG.md) for details on how to set up and use this system.

### Legacy Environment Variables (for development only)

These environment variables can be used during development, but in production, the database configuration system is used:

- `NEXT_PUBLIC_API_BASE_URL`: Backend API endpoint URL
- `NEXT_PUBLIC_ML_API_BASE_URL`: Optional separate ML API endpoint (if different from main API)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Deploying to Digital Ocean

This application is configured for easy deployment to Digital Ocean App Platform.

### Option 1: Deploying from GitHub

1. Fork this repository to your GitHub account
2. Create a new App on Digital Ocean App Platform
3. Connect your GitHub account and select this repository
4. Choose the main branch for deployment
5. Configure as follows:
   - Source Directory: `adv-analytics`
   - Build Command: `npm run build`
   - Run Command: `npm start`
   - HTTP Port: `3000`
6. Click "Launch App" to deploy

### Option 2: Deploying using Dockerfile (recommended)

1. Create a `Dockerfile` in the root of the `adv-analytics` directory:

```Dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app
COPY package.json yarn.lock* package-lock.json* ./
RUN npm ci

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
```

2. Create a new App on Digital Ocean App Platform, selecting "Deploy from Dockerfile"
3. Choose the repository and branch
4. Digital Ocean will automatically detect and use the Dockerfile
5. Configure environment variables if needed
6. Deploy the application

### Environment Variables

The following environment variables can be configured:

| Variable | Description | Default |
|----------|-------------|---------|
| NEXT_PUBLIC_API_BASE_URL | Base URL for API calls | `/api` |
| PORT | Port to run the server on | `3000` |

## Development

This is a standalone Next.js application that uses mock data for development. In production, you can connect it to a real API by setting the `NEXT_PUBLIC_API_BASE_URL` environment variable.

### Structure

- `app/` - Next.js application using the App Router
- `app/components/` - React components
- `app/services/` - API services and data management
- `app/analytics/` - Analytics page
- `app/ml-lab/` - ML Lab page for predictions

### Building for Production

```bash
npm run build
# or
yarn build
```

The build output will be in the `.next` directory. The application is configured with `output: 'standalone'` in `next.config.js` to optimize for production deployment.

## License

This project is licensed under the MIT License.
