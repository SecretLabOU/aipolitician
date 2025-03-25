#!/usr/bin/env python3
"""
High-Performance API Server

Production-ready FastAPI implementation with streaming responses, automatic rate limiting,
comprehensive monitoring, and GPU resource management.
"""

import os
import sys
import time
import asyncio
import logging
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator, Union, Annotated
from datetime import datetime, timedelta
from pathlib import Path
import functools
import json
import inspect
import traceback

# Add the parent directory to the path
parent_dir = Path(__file__).parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.getenv("API_LOG_FILE", "logs/api.log"))
    ]
)
logger = logging.getLogger("api_server")

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Import local modules
try:
    from models import model_manager
    from rag import get_context, rag_engine
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, Response, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.security import APIKeyHeader
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.routing import APIRoute
    from starlette.middleware.base import BaseHTTPMiddleware
    from pydantic import BaseModel, Field, validator, EmailStr, conint, confloat
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.error("FastAPI not available. Install with 'pip install fastapi uvicorn'")
    FASTAPI_AVAILABLE = False
    sys.exit(1)

# Optional telemetry
try:
    import psutil
    import platform
    import torch
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("Monitoring packages not available. Some telemetry will be limited.")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration from environment variables
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
ENABLE_API_KEY = os.getenv("ENABLE_API_KEY", "false").lower() == "true"
API_KEY_NAME = os.getenv("API_KEY_NAME", "X-API-Key")
API_KEY = os.getenv("API_KEY", "")
MAX_SESSION_IDLE_TIME = int(os.getenv("MAX_SESSION_IDLE_TIME", "3600"))  # 1 hour
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # window in seconds

# Data structures for rate limiting and sessions
rate_limits = {}  # Store client request counts
active_sessions = {}  # Store active chat sessions

class SessionManager:
    """Manager for chat sessions with cleanup and persistence"""
    
    def __init__(self, max_idle_time=MAX_SESSION_IDLE_TIME):
        self.sessions = {}
        self.max_idle_time = max_idle_time
        self.sessions_dir = Path(os.getenv("SESSIONS_DIR", "chat_sessions"))
        self.sessions_dir.mkdir(exist_ok=True, parents=True)
        self._load_persisted_sessions()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_idle_sessions())
    
    def _load_persisted_sessions(self):
        """Load persisted sessions from disk"""
        try:
            for file_path in self.sessions_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        session_data = json.load(f)
                        session_id = file_path.stem
                        self.sessions[session_id] = session_data
                        logger.debug(f"Loaded session {session_id} from disk")
                except Exception as e:
                    logger.error(f"Error loading session from {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading persisted sessions: {str(e)}")
    
    def _persist_session(self, session_id):
        """Persist session to disk"""
        try:
            if session_id in self.sessions:
                file_path = self.sessions_dir / f"{session_id}.json"
                with open(file_path, "w") as f:
                    json.dump(self.sessions[session_id], f)
        except Exception as e:
            logger.error(f"Error persisting session {session_id}: {str(e)}")
    
    def create_session(self, persona, metadata=None):
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "persona": persona,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "history": [],
            "metadata": metadata or {}
        }
        self.sessions[session_id] = session
        self._persist_session(session_id)
        return session_id
    
    def get_session(self, session_id):
        """Get a session by ID"""
        return self.sessions.get(session_id)
    
    def add_exchange(self, session_id, user_message, bot_response, context=None):
        """Add a message exchange to a session"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Add exchange to history
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "bot": bot_response,
            "context": context
        }
        session["history"].append(exchange)
        
        # Update last activity
        session["last_activity"] = datetime.now().isoformat()
        
        # Persist changes
        self._persist_session(session_id)
        return True
    
    def list_sessions(self, limit=100, skip=0):
        """List active sessions with pagination"""
        sessions = list(self.sessions.values())
        
        # Sort by last activity (most recent first)
        sessions.sort(key=lambda s: s["last_activity"], reverse=True)
        
        # Apply pagination
        return sessions[skip:skip+limit]
    
    def delete_session(self, session_id):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            # Remove persisted file
            file_path = self.sessions_dir / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
            
            return True
        return False
    
    async def _cleanup_idle_sessions(self):
        """Periodically clean up idle sessions"""
        while True:
            try:
                now = datetime.now()
                to_delete = []
                
                for session_id, session in self.sessions.items():
                    last_activity = datetime.fromisoformat(session["last_activity"])
                    idle_time = (now - last_activity).total_seconds()
                    
                    if idle_time > self.max_idle_time:
                        to_delete.append(session_id)
                
                for session_id in to_delete:
                    logger.info(f"Cleaning up idle session {session_id}")
                    self.delete_session(session_id)
                
                # Sleep for a minute before next cleanup
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {str(e)}")
                await asyncio.sleep(60)  # Still sleep on error


# Initialize session manager
session_manager = SessionManager()

# API Key security dependency
if ENABLE_API_KEY:
    api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
    
    async def get_api_key(api_key: str = Depends(api_key_header)):
        if api_key != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
                headers={"WWW-Authenticate": API_KEY_NAME},
            )
        return api_key
    
    # Apply to all endpoints
    api_key_dependency = [Depends(get_api_key)]
else:
    # No API key required
    api_key_dependency = []

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # Get client identifier (IP or API key if available)
        client_id = request.headers.get(API_KEY_NAME, request.client.host)
        
        # Skip rate limiting for excluded paths
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Check rate limit
        now = time.time()
        
        if client_id not in rate_limits:
            # First request from this client
            rate_limits[client_id] = {"count": 1, "reset_at": now + RATE_LIMIT_WINDOW}
        else:
            # Existing client
            client_rate = rate_limits[client_id]
            
            # Check if window has reset
            if now > client_rate["reset_at"]:
                # Reset window
                client_rate["count"] = 1
                client_rate["reset_at"] = now + RATE_LIMIT_WINDOW
            else:
                # Increment count
                client_rate["count"] += 1
                
                # Check if over limit
                if client_rate["count"] > RATE_LIMIT_REQUESTS:
                    # Calculate retry-after seconds
                    retry_after = int(client_rate["reset_at"] - now)
                    
                    # Return rate limit response
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "detail": "Rate limit exceeded",
                            "retry_after": retry_after
                        },
                        headers={"Retry-After": str(retry_after)}
                    )
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Only add headers if client has rate data
        if client_id in rate_limits:
            remaining = max(0, RATE_LIMIT_REQUESTS - rate_limits[client_id]["count"])
            reset_at = int(rate_limits[client_id]["reset_at"])
            
            response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_at)
        
        return response

# Custom route class for error handling and logging
class ErrorLoggingRoute(APIRoute):
    """Custom route class that logs exceptions"""
    
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()
        
        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except Exception as ex:
                # Log detailed exception info
                logger.error(f"Exception during request: {request.url.path}")
                logger.error(f"Method: {request.method}")
                logger.error(f"Headers: {request.headers}")
                logger.error(f"Client: {request.client}")
                logger.error(f"Exception: {str(ex)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Determine status code
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                if isinstance(ex, HTTPException):
                    status_code = ex.status_code
                
                # Return appropriate error response
                return JSONResponse(
                    status_code=status_code,
                    content={"detail": str(ex)}
                )
        
        return custom_route_handler

# API Models
class ChatMode(str, Enum):
    """Chat mode options"""
    STANDARD = "standard"
    RAG = "rag"
    CREATIVE = "creative"
    CONCISE = "concise"

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    cuda_available: bool
    gpu_info: Optional[List[Dict[str, Any]]] = None
    system_info: Optional[Dict[str, Any]] = None

class PersonaInfo(BaseModel):
    """Information about an available persona"""
    id: str
    display_name: str
    description: str

class PersonasResponse(BaseModel):
    """Response containing available personas"""
    personas: List[PersonaInfo]

class ChatRequest(BaseModel):
    """Chat request payload"""
    persona: str
    message: str
    session_id: Optional[str] = None
    mode: ChatMode = ChatMode.STANDARD
    max_length: conint(ge=1, le=2048) = 512  # Constrained integer
    temperature: confloat(ge=0.0, le=1.0) = 0.7  # Constrained float
    stream: bool = False
    
    @validator('persona')
    def validate_persona(cls, v):
        if v not in model_manager.get_available_personas():
            raise ValueError(f"Invalid persona: {v}")
        return v

class ChatResponse(BaseModel):
    """Chat response payload"""
    id: str
    persona: str
    message: str
    response: str
    session_id: Optional[str]
    created_at: datetime
    context: Optional[str] = None
    processing_time: float

class SessionCreateRequest(BaseModel):
    """Request to create a new session"""
    persona: str
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('persona')
    def validate_persona(cls, v):
        if v not in model_manager.get_available_personas():
            raise ValueError(f"Invalid persona: {v}")
        return v

class SessionResponse(BaseModel):
    """Response containing session information"""
    id: str
    persona: str
    created_at: datetime
    last_activity: datetime
    history_length: int
    metadata: Optional[Dict[str, Any]] = None

class SessionsResponse(BaseModel):
    """Response containing active sessions"""
    sessions: List[SessionResponse]
    total: int
    limit: int
    skip: int

class SystemInfo(BaseModel):
    """System information"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: Optional[List[Dict[str, Any]]] = None
    uptime: float

# Monitor system resources
async def get_system_info():
    """Get current system information"""
    if not MONITORING_AVAILABLE:
        return None
    
    try:
        info = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "uptime": time.time() - psutil.boot_time(),
            "gpu_usage": None
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                # Get memory info
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                
                gpu_info.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                    "memory_total_gb": round(memory_total, 2),
                    "utilization_percent": round((memory_allocated / memory_total) * 100, 2)
                })
            
            info["gpu_usage"] = gpu_info
        
        return info
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return None

# Create FastAPI app
app = FastAPI(
    title="AI Political Chat API",
    description="Advanced API for communicating with AI political personas",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    default_response_class=JSONResponse
)

# Use custom route class for error handling
app.router.route_class = ErrorLoggingRoute

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be more specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting middleware if enabled
if RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)

@app.on_event("startup")
async def startup_event():
    """Runs when the API server starts"""
    logger.info("Starting API server")
    
    # Log configuration
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"API Key enabled: {ENABLE_API_KEY}")
    logger.info(f"Rate limiting: {RATE_LIMIT_ENABLED}")
    
    # Check CUDA availability
    if hasattr(model_manager, 'init_gpu'):
        model_manager.init_gpu()
    
    # Log available personas
    personas = model_manager.get_available_personas()
    logger.info(f"Available personas: {', '.join(personas)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the API server shuts down"""
    logger.info("Shutting down API server")
    
    # Clean up resources
    if hasattr(model_manager, 'shutdown'):
        model_manager.shutdown()
    else:
        model_manager.clear_cache()
    
    if hasattr(rag_engine, 'close'):
        rag_engine.close()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    cuda_available = torch.cuda.is_available() if 'torch' in sys.modules else False
    
    # Get GPU info if available
    gpu_info = None
    if cuda_available:
        import torch
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        
        for i in range(gpu_count):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2)
            })
    
    # Get system info if monitoring available
    system_info = await get_system_info()
    
    return HealthResponse(
        cuda_available=cuda_available,
        gpu_info=gpu_info,
        system_info=system_info
    )

@app.get("/system", dependencies=api_key_dependency)
async def system_info():
    """System monitoring endpoint"""
    info = await get_system_info()
    if not info:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="System monitoring not available"
        )
    return info

@app.get("/personas", response_model=PersonasResponse, dependencies=api_key_dependency)
async def list_personas():
    """List available personas"""
    available_personas = model_manager.get_available_personas()
    
    # Create persona info objects
    personas = []
    for persona_id in available_personas:
        display_name = model_manager.get_display_name(persona_id)
        
        # You could add more detailed descriptions in a real implementation
        descriptions = {
            "trump": "45th President of the United States with distinctive speaking style",
            "biden": "46th President of the United States with focus on unity and policy details"
        }
        
        description = descriptions.get(persona_id, f"AI version of {display_name}")
        
        personas.append(PersonaInfo(
            id=persona_id,
            display_name=display_name,
            description=description
        ))
    
    return PersonasResponse(personas=personas)

@app.post("/chat", response_model=ChatResponse, dependencies=api_key_dependency)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Chat with a political persona (non-streaming)"""
    if request.stream:
        return await stream_chat(request)
    
    start_time = time.time()
    
    # Generate a unique ID for this exchange
    exchange_id = str(uuid.uuid4())
    
    try:
        # Check and get existing session if provided
        session_id = request.session_id
        
        # Get context if RAG mode is enabled
        use_rag = request.mode == ChatMode.RAG
        context = await get_context(request.message, request.persona) if use_rag else None
        
        # Adjust parameters based on mode
        max_length = request.max_length
        temperature = request.temperature
        
        if request.mode == ChatMode.CREATIVE:
            temperature = min(0.9, temperature * 1.3)  # Increase temperature
        elif request.mode == ChatMode.CONCISE:
            max_length = max(10, max_length // 2)  # Shorter responses
            temperature = max(0.3, temperature * 0.8)  # Lower temperature
        
        # Generate response
        response = await model_manager.generate_response(
            request.persona, 
            request.message, 
            context,
            max_length,
            temperature
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update session if provided
        if session_id:
            # Check if session exists
            session = session_manager.get_session(session_id)
            if not session or session["persona"] != request.persona:
                # Create new session if not exists or persona mismatch
                session_id = session_manager.create_session(request.persona)
            
            # Add exchange to session
            session_manager.add_exchange(
                session_id, 
                request.message, 
                response, 
                context
            )
        
        # Create response
        chat_response = ChatResponse(
            id=exchange_id,
            persona=request.persona,
            message=request.message,
            response=response,
            session_id=session_id,
            created_at=datetime.now(),
            context=context,
            processing_time=processing_time
        )
        
        # Log exchange
        logger.info(f"Generated response for persona {request.persona} in {processing_time:.2f}s")
        
        return chat_response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_streaming_response(request: ChatRequest):
    """Generate a streaming response"""
    try:
        # Get context if RAG mode is enabled
        use_rag = request.mode == ChatMode.RAG
        context = await get_context(request.message, request.persona) if use_rag else None
        
        # Adjust parameters based on mode
        max_length = request.max_length
        temperature = request.temperature
        
        if request.mode == ChatMode.CREATIVE:
            temperature = min(0.9, temperature * 1.3)
        elif request.mode == ChatMode.CONCISE:
            max_length = max(10, max_length // 2)
            temperature = max(0.3, temperature * 0.8)
        
        # Start stream
        full_response = ""
        async for chunk in model_manager.generate_response(
            request.persona, 
            request.message, 
            context,
            max_length,
            temperature,
            streaming=True
        ):
            # Format as server-sent event
            full_response += chunk
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        
        # Send the final event with the full response
        yield f"data: {json.dumps({'text': '[DONE]', 'full_response': full_response})}\n\n"
        
        # Update session if provided
        if request.session_id:
            session_manager.add_exchange(
                request.session_id,
                request.message,
                full_response,
                context
            )
        
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"

@app.post("/chat/stream", dependencies=api_key_dependency)
async def stream_chat(request: ChatRequest):
    """Chat with streaming response"""
    # Ensure streaming is requested
    if not request.stream:
        return await chat(request)
    
    # Set up streaming response
    return StreamingResponse(
        generate_streaming_response(request),
        media_type="text/event-stream"
    )

@app.post("/sessions", response_model=SessionResponse, dependencies=api_key_dependency)
async def create_session(request: SessionCreateRequest):
    """Create a new chat session"""
    session_id = session_manager.create_session(request.persona, request.metadata)
    session = session_manager.get_session(session_id)
    
    return SessionResponse(
        id=session_id,
        persona=session["persona"],
        created_at=datetime.fromisoformat(session["created_at"]),
        last_activity=datetime.fromisoformat(session["last_activity"]),
        history_length=len(session["history"]),
        metadata=session["metadata"]
    )

@app.get("/sessions/{session_id}", response_model=SessionResponse, dependencies=api_key_dependency)
async def get_session(session_id: str):
    """Get session information"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        id=session_id,
        persona=session["persona"],
        created_at=datetime.fromisoformat(session["created_at"]),
        last_activity=datetime.fromisoformat(session["last_activity"]),
        history_length=len(session["history"]),
        metadata=session["metadata"]
    )

@app.get("/sessions", response_model=SessionsResponse, dependencies=api_key_dependency)
async def list_sessions(limit: int = 100, skip: int = 0):
    """List all active sessions"""
    sessions_list = session_manager.list_sessions(limit, skip)
    total_sessions = len(session_manager.sessions)
    
    # Convert to response objects
    session_responses = []
    for session in sessions_list:
        session_responses.append(SessionResponse(
            id=session["id"],
            persona=session["persona"],
            created_at=datetime.fromisoformat(session["created_at"]),
            last_activity=datetime.fromisoformat(session["last_activity"]),
            history_length=len(session["history"]),
            metadata=session["metadata"]
        ))
    
    return SessionsResponse(
        sessions=session_responses,
        total=total_sessions,
        limit=limit,
        skip=skip
    )

@app.delete("/sessions/{session_id}", status_code=204, dependencies=api_key_dependency)
async def delete_session(session_id: str):
    """Delete a session"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)

@app.get("/sessions/{session_id}/history", dependencies=api_key_dependency)
async def get_session_history(session_id: str):
    """Get the full history of a session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"history": session["history"]}

def run_server():
    """Run the API server"""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available. Install with 'pip install fastapi uvicorn'")
        return
    
    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        log_config={
            "version": 1,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                }
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "default",
                    "class": "logging.FileHandler",
                    "filename": "logs/uvicorn.log",
                }
            },
            "loggers": {
                "uvicorn": {"handlers": ["default", "file"], "level": "INFO"},
                "uvicorn.error": {"handlers": ["default", "file"], "level": "INFO"},
                "uvicorn.access": {"handlers": ["default", "file"], "level": "INFO"},
            }
        },
        reload=DEBUG
    )

if __name__ == "__main__":
    run_server() 