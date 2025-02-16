from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat
from app.models.chat import ChatResponse

app = FastAPI(
    title="AI Politician",
    description="An AI system for interacting with virtual political agents",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "AI Politician API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
