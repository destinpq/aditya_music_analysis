import { NextRequest, NextResponse } from 'next/server';

// Tell Next.js this route is dynamic and shouldn't be statically optimized
export const dynamic = 'force-dynamic';

// This is a simplified mock implementation
// In a real app, you would handle file parsing and database storage
export async function POST(request: NextRequest) {
  try {
    // Unused request parameter is preserved to maintain function signature
    
    // In a real implementation:
    // 1. Parse the form data
    // 2. Save the file
    // 3. Process the CSV data
    // 4. Store in database
    // 5. Return success with dataset ID

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Return mock success response
    return NextResponse.json({
      success: true,
      datasetId: 123,
      rowCount: 42,
      insertedRows: 42
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'Failed to process upload' },
      { status: 500 }
    );
  }
} 