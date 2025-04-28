import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

// Configure OpenAI with API key from environment variables
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, 
});

// CPM values for each country (Revenue per thousand views)
const COUNTRY_CPM: Record<string, number> = {
  'India': 120,
  'USA': 600,
  'UK': 500,
  'Canada': 450,
  'Australia': 450,
};

// Video types based on genre
const VIDEO_TYPES: Record<string, string[]> = {
  'Romantic': ['Lyric Video', 'Music Video', 'Dance Cover'],
  'Classical': ['Performance Video', 'Visualizer'],
  'Folk': ['Performance Video', 'Dance Video'],
  'Devotional': ['Lyric Video', 'Animated Video'],
  'Pop': ['Music Video', 'Dance Cover', 'Lyric Video'],
  'Melody': ['Lyric Video', 'Animation'],
  'Item Song': ['Dance Video', 'Music Video'],
  'Dance': ['Dance Video', 'Studio Performance'],
  'Hip Hop': ['Music Video', 'Dance Video', 'Lyric Video']
};

export async function POST(request: NextRequest) {
  try {
    // Parse the request body
    const { genre, country } = await request.json();

    if (!genre || !country) {
      return NextResponse.json(
        { error: 'Missing genre or country parameter' },
        { status: 400 }
      );
    }

    // Verify that the country has a CPM value
    if (!COUNTRY_CPM[country]) {
      return NextResponse.json(
        { error: 'Invalid country parameter' },
        { status: 400 }
      );
    }

    let predictedViews = 0;
    let engagementLevel = '';

    try {
      // Call the OpenAI API to get prediction
      const aiResponse = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "You are a music industry analytics expert. Provide realistic YouTube and Instagram view projections for music videos based on genre and target audience."
          },
          {
            role: "user",
            content: `Predict the average monthly YouTube views for a ${genre} music video by Aditya Music targeting ${country} audience. Only provide a number (no text).`
          }
        ],
        temperature: 0.4,
      });

      const responseText = aiResponse.choices[0]?.message?.content || '';
      
      // Extract the number from the AI response
      const viewsMatch = responseText.match(/\d[\d,. ]*/);
      if (viewsMatch) {
        // Convert the matched string to a number (removing commas, spaces, etc.)
        predictedViews = parseInt(viewsMatch[0].replace(/[,. ]/g, ''), 10);
      } else {
        // Fallback to simulated views if AI response doesn't contain a number
        console.log("Using fallback prediction as AI didn't return a valid number");
        predictedViews = generateFallbackViews(genre, country);
      }
    } catch (error) {
      console.error("Error with OpenAI API:", error);
      // Fallback to simulated views if API call fails
      predictedViews = generateFallbackViews(genre, country);
    }
    
    // Determine engagement level based on views
    if (predictedViews > 1500000) {
      engagementLevel = 'Very High';
    } else if (predictedViews > 1000000) {
      engagementLevel = 'High';
    } else if (predictedViews > 500000) {
      engagementLevel = 'Moderate';
    } else {
      engagementLevel = 'Low';
    }
    
    // Select a random video type from the available types for the genre
    const availableVideoTypes = VIDEO_TYPES[genre] || ['Lyric Video', 'Music Video'];
    const videoType = availableVideoTypes[Math.floor(Math.random() * availableVideoTypes.length)];

    // Calculate revenue based on views and country CPM
    const cpm = COUNTRY_CPM[country];
    const revenue = Math.floor((predictedViews / 1000) * cpm);

    return NextResponse.json({
      views: predictedViews,
      revenue: revenue,
      engagement: engagementLevel,
      videoType: videoType
    });
  } catch (error) {
    console.error('Error processing prediction request:', error);
    return NextResponse.json(
      { error: 'Failed to process prediction request' },
      { status: 500 }
    );
  }
}

// Function to generate fallback views if the API call fails
function generateFallbackViews(genre: string, country: string): number {
  const baseViews = 500000; // Base views
  
  // Genre popularity multiplier
  const genreMultiplierMap: Record<string, number> = {
    'Romantic': 1.5,
    'Classical': 0.7,
    'Folk': 0.8,
    'Devotional': 1.2,
    'Pop': 1.7,
    'Melody': 1.3,
    'Item Song': 1.8,
    'Dance': 1.6,
    'Hip Hop': 1.4
  };
  
  const genreMultiplier = genreMultiplierMap[genre] || 1.0;
  
  // Country audience multiplier
  const countryMultiplierMap: Record<string, number> = {
    'India': 2.0,
    'USA': 1.0,
    'UK': 0.8,
    'Canada': 0.7,
    'Australia': 0.6
  };
  
  const countryMultiplier = countryMultiplierMap[country] || 1.0;
  
  // Calculate views with some randomness
  const randomFactor = 0.7 + Math.random() * 0.6; // Between 0.7 and 1.3
  return Math.floor(baseViews * genreMultiplier * countryMultiplier * randomFactor);
} 