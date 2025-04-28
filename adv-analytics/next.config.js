/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || '/api',
  },
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