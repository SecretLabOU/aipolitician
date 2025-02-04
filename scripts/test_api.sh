#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base URL
BASE_URL="http://localhost:8000"

# Print with color
print_color() {
    color=$1
    message=$2
    printf "${color}%s${NC}\n" "$message"
}

# Function to make API calls
call_api() {
    endpoint=$1
    method=${2:-GET}
    data=$3
    
    print_color $YELLOW "\nTesting $method $endpoint..."
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -X GET "$BASE_URL$endpoint")
    else
        response=$(curl -s -X $method "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "Response:"
        echo $response | python3 -m json.tool
    else
        print_color $RED "Error making request"
        exit 1
    fi
}

# Test health endpoint
call_api "/health"

# Test topics endpoint
call_api "/topics"

# Test chat endpoint with sample queries
queries=(
    "What is John Smith's position on healthcare?"
    "How did he vote on the Healthcare Reform Bill?"
    "What are his views on climate change?"
)

for query in "${queries[@]}"; do
    call_api "/chat" "POST" "{\"text\": \"$query\"}"
done

# Test conversation history
call_api "/conversation_history"

# Test cache stats
call_api "/cache/stats"

# Test metrics endpoint
print_color $YELLOW "\nTesting metrics endpoint..."
curl -s "$BASE_URL/metrics" | head -n 10

print_color $GREEN "\nAPI tests completed!"
