
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
from typing import List, Optional, AsyncGenerator
from core.orchestrator import KarpathyTwinOrchestrator
from core.config_loader import load_env
load_env()  # ensure .env is loaded before voice endpoints read os.environ
from audio.voicebox_client import VoiceboxClient
from audio.cadence_formatter import format_for_tts

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
from core.session_store import SessionStore
sessions = SessionStore()

@app.on_event("startup")
async def startup_event():
    async def eviction_task():
        while True:
            await asyncio.sleep(600)  # 10 minutes
            logger.info("Evicting stale sessions")
            sessions.evict_stale(max_age_seconds=3600)
    asyncio.create_task(eviction_task())

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

class FeedbackRequest(BaseModel):
    session_id: str
    message_id_or_index: str
    feedback_type: str
    note: Optional[str] = None

from feedback.store import FeedbackStore
feedback_store = FeedbackStore()

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
        orchestrator = sessions.get_or_create(session_id)
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}", exc_info=True)
        # Even if orchestrator fails, we return a safe response
        return ChatResponse(
            response="Hey! I'm having a little trouble getting started, but let's chat. What do you want to know about neural networks?",
            sources=[],
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    try:
        # Orchestrator is synchronous, so we offload it to a thread
        logger.info(f"Session {session_id} sending message: {message[:50]}...")
        result = await asyncio.to_thread(orchestrator.chat, user_message=message, thread_id=session_id)
        
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
    
    try:
        orchestrator = sessions.get_or_create(session_id)
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}", exc_info=True)
        async def error_stream():
            yield "data: Hey! I'm having a little trouble getting started, but let's chat.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            from agent.nodes import input_node, rag_node, memory_node, timeline_node, persona_prompt_node, memory_save_node, validation_node, output_node, get_gemini_client, get_system_prompt
            from google.genai import types
            
            # Initial state
            state = {
                "user_message": message,
                "thread_id": session_id,
                "messages": [],
                "system_prompt_cache": "",
                "short_term_summary": "",
                "is_valid": True,
                "validation_issues": []
            }
            
            config = {"configurable": {"thread_id": session_id}}
            checkpoint = orchestrator.graph.get_state(config)
            if checkpoint and checkpoint.values:
                state["messages"] = checkpoint.values.get("messages", [])
                
            # Run prep nodes
            state.update(input_node(state))
            state.update(rag_node(state, orchestrator.retriever))
            state.update(memory_node(state, orchestrator.memory))
            state.update(timeline_node(state, orchestrator.timeline))
            state.update(persona_prompt_node(state, orchestrator.prompt_builder))
            
            # Prepare for Gemini
            client = get_gemini_client()
            contents = []
            for msg in state["messages"]:
                role = "user" if getattr(msg, "type", "human") == "human" else "model"
                txt = getattr(msg, "content", str(msg))
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=txt)]))
                
            system_instruction = state.get("system_prompt_cache") or get_system_prompt()
            
            # Execute streaming request in thread to avoid blocking event loop
            def fetch_stream():
                return client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                        top_p=0.95,
                    )
                )
                
            response_stream = await asyncio.to_thread(fetch_stream)
            
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    # Send raw text chunks, let frontend handle assembling
                    yield f"data: {chunk.text.replace(chr(10), ' ')}\n\n"
                    
            state["response"] = full_response
            
            # Post-processing
            state.update(memory_save_node(state, orchestrator.memory))
            state.update(validation_node(state, orchestrator.validator))
            state.update(output_node(state))
            
            # Update checkpoint
            orchestrator.graph.update_state(config, state)
            
            # Send sources
            yield f"data: __SOURCES__{json.dumps(state['sources'])}\n\n"
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

@app.post("/api/feedback")
async def receive_feedback(request: FeedbackRequest):
    feedback_store.log_feedback(
        session_id=request.session_id,
        message_id_or_index=request.message_id_or_index,
        feedback_type=request.feedback_type,
        note=request.note
    )
    return {"status": "ok"}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        logger.info(f"Closing and removing session: {session_id}")
        try:
            sessions.remove(session_id)
        except Exception as e:
            logger.error(f"Error closing orchestrator: {e}")
        return {"status": "deleted"}
    return {"status": "not_found"}

# Mount static files to serve the frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

_voicebox_client: Optional[VoiceboxClient] = None

def get_voicebox() -> VoiceboxClient:
    global _voicebox_client
    if _voicebox_client is None:
        _voicebox_client = VoiceboxClient()
    return _voicebox_client

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.get("/api/voice/status")
async def voice_status():
    enabled = bool(os.environ.get("VOICEBOX_PROFILE_ID"))
    available = get_voicebox().is_available()
    profile_id = os.environ.get("VOICEBOX_PROFILE_ID")
    return {"enabled": enabled, "available": available, "profile_id": profile_id}

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    profile_id = os.environ.get("VOICEBOX_PROFILE_ID")
    if not profile_id:
        raise HTTPException(status_code=500, detail="VOICEBOX_PROFILE_ID not set in .env")
    
    if not get_voicebox().is_available():
        raise HTTPException(status_code=503, detail="Voicebox is not running. Start the Voicebox desktop app.")
        
    formatted_text = format_for_tts(request.text)
    
    os.makedirs("storage/audio", exist_ok=True)
    audio_id = str(uuid.uuid4())
    output_path = f"storage/audio/{audio_id}.wav"
    
    result_path = await asyncio.to_thread(get_voicebox().synthesize, formatted_text, output_path)
    
    if not result_path:
        raise HTTPException(status_code=500, detail="Failed to generate audio from Voicebox")
        
    return FileResponse(result_path, media_type="audio/wav")
