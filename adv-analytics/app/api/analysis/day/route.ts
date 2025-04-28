import { NextRequest, NextResponse } from 'next/server';
import { getDayAnalysis } from '../../../controllers/analysisController';

// Tell Next.js this route is dynamic and shouldn't be statically optimized
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    // Get query parameters
    const searchParams = request.nextUrl.searchParams;
    const datasetId = parseInt(searchParams.get('dataset_id') || '0', 10);
    const startDate = searchParams.get('start_date') || undefined;
    const endDate = searchParams.get('end_date') || undefined;
    
    if (!datasetId) {
      return NextResponse.json(
        { error: 'dataset_id is required' },
        { status: 400 }
      );
    }
    
    // Get analysis data
    const dayMetrics = await getDayAnalysis(datasetId, startDate || undefined, endDate || undefined);
    
    // Return the data
    return NextResponse.json(dayMetrics);
  } catch (error) {
    console.error('Error in day analysis:', error);
    return NextResponse.json(
      { error: 'Failed to get day analysis data' },
      { status: 500 }
    );
  }
} 