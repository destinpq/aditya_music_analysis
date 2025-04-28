#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting ADV Analytics Environment...${NC}"

# Stop any existing processes on these ports
echo -e "${YELLOW}Stopping any existing processes on ports 1111 and 2222...${NC}"
lsof -ti:1111 | xargs kill -9 2>/dev/null || true
lsof -ti:2222 | xargs kill -9 2>/dev/null || true

# Start Backend on port 1111
echo -e "${YELLOW}Starting Backend on port 1111...${NC}"
cd Backend/fastapi_app
python main.py &
BACKEND_PID=$!

# Wait a moment for the backend to initialize
sleep 2

# Start Frontend on port 2222
echo -e "${YELLOW}Starting Frontend on port 2222...${NC}"
cd ../../adv-analytics
npm run dev -- -p 2222 &
FRONTEND_PID=$!

echo -e "${GREEN}Services started:${NC}"
echo -e "${GREEN}Backend running on http://localhost:1111${NC}"
echo -e "${GREEN}Frontend running on http://localhost:2222${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"

# Function to handle script termination
cleanup() {
    echo -e "${YELLOW}Stopping services...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}Services stopped${NC}"
    exit 0
}

# Register the cleanup function for when SIGINT is received
trap cleanup SIGINT

# Keep the script running
wait 