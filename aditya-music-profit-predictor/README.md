# Aditya Music AI Profit Predictor

An AI-powered tool for predicting YouTube and Instagram views and revenue for music videos based on genre and target country.

## Features

- Select music genre and target country
- AI-powered prediction of views and engagement
- Calculate estimated revenue based on country-specific CPM rates
- Visual charts comparing potential revenue across different countries
- Suggested video type for maximum performance

## Tech Stack

- Next.js 14 with App Router
- TypeScript
- Tailwind CSS
- Chart.js for data visualization
- OpenAI API for AI predictions

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- OpenAI API key

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd aditya-music-profit-predictor
```

2. Install dependencies
```bash
npm install
# or
yarn
```

3. Configure the OpenAI API key
Create a `.env.local` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY="your-api-key-here"
```

4. Run the development server
```bash
npm run dev
# or
yarn dev
```

5. Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## Project Structure

- `/src/app/page.tsx` - Home page
- `/src/app/predict/page.tsx` - Genre and country selection page
- `/src/app/results/page.tsx` - Results display page
- `/src/app/api/predict/route.ts` - API route for prediction
- `/src/components/` - Reusable components

## Notes for Deployment

- This application is designed to be deployed on Vercel
- For production, ensure you set the OPENAI_API_KEY in your environment variables

## Future Enhancements

- Add more genres and countries
- Improve AI prediction accuracy with more training data
- Add more detailed analytics and insights
- Allow for custom CPM rates
- Historical performance data comparison

---

Built for Aditya Music with ❤️
