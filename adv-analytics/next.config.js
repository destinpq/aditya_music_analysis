/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Environment variables are now loaded from database via app/utils/config-db.ts
  webpack: (config, { isServer }) => {
    // Fix for Plotly.js import issue
    config.resolve.alias = {
      ...config.resolve.alias,
      'plotly.js/dist/plotly': 'plotly.js',
    };
    
    return config;
  },
  output: 'standalone', // Optimizes for production deployment
}

module.exports = nextConfig 