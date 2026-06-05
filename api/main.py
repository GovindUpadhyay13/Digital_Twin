
import uuid
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
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
    year: Optional[int] = None
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
    
    try:
        if session_id not in sessions:
            logger.info(f"Instantiating new orchestrator for session: {session_id}")
            # We initialize it lazily per session_id
            sessions[session_id] = KarpathyTwinOrchestrator(session_id=session_id)
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}", exc_info=True)
        # Even if orchestrator fails, we return a safe response
        return ChatResponse(
            response="Hey! I'm having a little trouble getting started, but let's chat. What do you want to know about neural networks?",
            sources=[],
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    orchestrator = sessions[session_id]
    
    try:
        # Orchestrator is synchronous, so we offload it to a thread
        logger.info(f"Session {session_id} sending message: {message[:50]}...")
        result = await asyncio.to_thread(orchestrator.chat, user_message=message)
        
        parsed_sources = []
        for s in result.get("sources", []):
            year_val = None
            raw_year = s.get("year")
            if raw_year is not None:
                if isinstance(raw_year, int):
                    year_val = raw_year
                elif isinstance(raw_year, str) and raw_year.strip() != "":
                    try:
                        year_val = int(raw_year.strip())
                    except (ValueError, TypeError):
                        year_val = None
            parsed_sources.append(Source(
                title=s.get("title", "Unknown"),
                year=year_val,
                type=s.get("source_type", s.get("type", "unknown")),
                relevance=float(s.get("relevance", 0.0))
            ))
            
        return ChatResponse(
            response=result.get("response", ""),
            sources=parsed_sources,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Error during chat: {e}", exc_info=True)
        # Return fallback response on any error
        return ChatResponse(
            response="Hey! Let's talk about neural networks. What are you curious about?",
            sources=[],
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

# Streaming endpoint (simulated since our LLM call is synchronous for now)
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id
    message = request.message
    
    if session_id not in sessions:
        logger.info(f"Instantiating new orchestrator for session: {session_id}")
        try:
            sessions[session_id] = KarpathyTwinOrchestrator(session_id=session_id)
        except Exception as e:
            logger.error(f"Failed to create orchestrator: {e}", exc_info=True)
            async def error_stream():
                yield "data: Hey! I'm having a little trouble getting started, but let's chat.\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")

    orchestrator = sessions[session_id]
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Get full response first
            result = await asyncio.to_thread(orchestrator.chat, user_message=message)
            full_response = result.get("response", "")
            sources = result.get("sources", [])
            
            # Simulate streaming by splitting into words and sending chunks
            words = full_response.split()
            chunk = ""
            for i, word in enumerate(words):
                chunk += word + " "
                if i % 3 == 0:  # Send every 3 words
                    yield f"data: {chunk}\n\n"
                    chunk = ""
                    await asyncio.sleep(0.05)  # Simulate delay
            if chunk:
                yield f"data: {chunk}\n\n"
                
            # Send sources as final event
            yield f"data: __SOURCES__{json.dumps(sources)}\n\n"
            yield "data: __END__\n\n"
        except Exception as e:
            logger.error(f"Error during streaming chat: {e}", exc_info=True)
            yield f"data: Hey! Let's talk about neural networks.\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/session/{session_id}/history")
async def get_history(session_id: str):
    # Returns conversation history if available
    # For now, orchestrator abstracts it, but we can return an empty list
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
import json
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")
