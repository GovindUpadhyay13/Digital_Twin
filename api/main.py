import os
import json
import uuid
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from core.orchestrator import KarpathyTwinOrchestrator
from core.session_store import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Karpathy Digital Twin API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Store (LRU Cache of active sessions)
sessions = SessionStore()

@app.on_event("startup")
async def startup_event():
    # Eviction task for stale sessions
    async def eviction_task():
        while True:
            await asyncio.sleep(600)  # 10 minutes
            logger.info("Evicting stale sessions")
            sessions.evict_stale(max_age_seconds=3600)
    asyncio.create_task(eviction_task())
    
    # Preload retriever and embedder at startup
    try:
        KarpathyTwinOrchestrator.initialize_global_retriever()
        logger.info("Preloaded embedding model and ChromaDB client.")
    except Exception as e:
        logger.error(f"Failed to preload models/databases on startup: {e}", exc_info=True)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/session/new")
async def create_session():
    new_id = str(uuid.uuid4())
    logger.info(f"Created new session: {new_id}")
    return {"session_id": new_id}

@app.post("/chat")
async def chat_sse(request: ChatRequest):
    session_id = request.session_id
    message = request.message
    
    try:
        orchestrator = sessions.get_or_create(session_id)
    except Exception as e:
        logger.error(f"Failed to get or create orchestrator for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize orchestrator: {e}")
        
    async def event_generator():
        try:
            # Yield tokens as they stream from Gemini
            async for token in orchestrator.chat_stream(message):
                yield f"data: {json.dumps({'text': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in chat_sse stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/session/{session_id}/history")
async def get_history(session_id: str):
    # Retrieve in-memory history if active
    if session_id in sessions:
        orchestrator = sessions.get_or_create(session_id)
        return {"history": orchestrator.history}
    return {"history": []}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        logger.info(f"Removing session: {session_id}")
        sessions.remove(session_id)
        return {"status": "deleted"}
    return {"status": "not_found"}

# Mount static files to serve the frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")
