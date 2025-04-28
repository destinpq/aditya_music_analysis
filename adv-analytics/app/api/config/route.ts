import { NextRequest, NextResponse } from 'next/server';
import { getConfig, setConfig } from '../../utils/config-db';

// GET /api/config?key=X - Get a config value
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const key = searchParams.get('key');
  
  if (!key) {
    return NextResponse.json({ error: 'Key parameter is required' }, { status: 400 });
  }
  
  try {
    const value = await getConfig(key);
    return NextResponse.json({ key, value });
  } catch (error) {
    console.error('Error retrieving config:', error);
    return NextResponse.json({ error: 'Failed to retrieve configuration' }, { status: 500 });
  }
}

// POST /api/config - Set a config value
export async function POST(request: NextRequest) {
  try {
    const { key, value, description } = await request.json();
    
    if (!key || value === undefined) {
      return NextResponse.json({ error: 'Key and value are required' }, { status: 400 });
    }
    
    const success = await setConfig(key, value, description || '');
    
    if (success) {
      return NextResponse.json({ success: true });
    } else {
      return NextResponse.json({ error: 'Failed to save configuration' }, { status: 500 });
    }
  } catch (error) {
    console.error('Error setting config:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
} 