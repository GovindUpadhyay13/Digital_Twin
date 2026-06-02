import uuid
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from core.orchestrator import KarpathyTwinOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Karpathy Digital Twin API")

# CORS middleware for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Cache: session_id -> KarpathyTwinOrchestrator
sessions: dict[str, KarpathyTwinOrchestrator] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class Source(BaseModel):
    title: str
    year: int
    type: str
    relevance: float

class ChatResponse(BaseModel):
    response: str
    sources: List[Source]
    timestamp: str

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/session/new")
async def create_session():
    new_id = str(uuid.uuid4())
    logger.info(f"Created new session: {new_id}")
    return {"session_id": new_id}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    message = request.message
    
    if session_id not in sessions:
        logger.info(f"Instantiating new orchestrator for session: {session_id}")
        try:
            # We initialize it lazily per session_id
            sessions[session_id] = KarpathyTwinOrchestrator(session_id=session_id)
        except Exception as e:
            logger.error(f"Failed to create orchestrator: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail={"error": "Failed to initialize orchestrator."})

    orchestrator = sessions[session_id]
    
    try:
        # Orchestrator is synchronous, so we offload it to a thread
        logger.info(f"Session {session_id} sending message: {message[:50]}...")
        result = await asyncio.to_thread(orchestrator.chat, user_message=message)
        
        parsed_sources = []
        for s in result.get("sources", []):
            parsed_sources.append(Source(
                title=s.get("title", "Unknown"),
                year=s.get("year", 2023),
                type=s.get("source_type", s.get("type", "unknown")),
                relevance=s.get("relevance", 0.0)
            ))
            
        return ChatResponse(
            response=result.get("response", ""),
            sources=parsed_sources,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Error during chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Internal server error during chat processing."})

@app.get("/api/session/{session_id}/history")
async def get_history(session_id: str):
    # Returns conversation history if available
    # For now, orchestrator abstracts it, but we can return an empty list
    # or implement extracting memory window later.
    return {"history": []}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        logger.info(f"Closing and removing session: {session_id}")
        try:
            sessions[session_id].close()
        except Exception as e:
            logger.error(f"Error closing orchestrator: {e}")
        del sessions[session_id]
        return {"status": "deleted"}
    return {"status": "not_found"}

# Mount static files to serve the frontend
import os
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

