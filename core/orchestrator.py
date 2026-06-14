import os
import logging
from typing import Dict, Any, AsyncGenerator

from core.config_loader import load_config, load_env
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class KarpathyTwinOrchestrator:
    retriever = None
    
    @classmethod
    def initialize_global_retriever(cls, config_path: str = "config.yaml"):
        """Initialize the single global retriever once to preload models/databases on startup"""
        load_env()
        config = load_config(config_path)
        rag_conf = config.get("rag", {})
        
        from rag.retriever import Retriever
        cls.retriever = Retriever(
            model_name=rag_conf.get("embedding_model", "all-MiniLM-L6-v2"),
            persist_dir=rag_conf.get("persist_dir", "storage/chroma_db"),
            collection_name=rag_conf.get("collection_name", "karpathy_knowledge"),
            top_k=5
        )
        logger.info("Global RAG retriever initialized successfully.")

    def __init__(self, session_id: str = None, config_path: str = "config.yaml"):
        # Load environment variables
        load_env()
        self.session_id = session_id
        
        # Ensure retriever is initialized
        if self.retriever is None:
            self.initialize_global_retriever(config_path)
            
        # simple in-memory list of last 8 messages as the conversation history
        # Format: [{"role": "user"|"model", "text": str}]
        self.history = []
        
        logger.info(f"Digital Twin Orchestrator initialized. Session ID: {self.session_id}")

    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Retrieves context, calls Gemini using async streaming, yields text chunks, and saves to history.
        """
        # 1. Retrieve RAG context (top-5 cosine similarity)
        results = self.retriever.retrieve(user_message, top_k=5)
        rag_context = self.retriever.format_context(results)
        
        # 2. Convert history to types.Content structure
        contents = []
        for h in self.history:
            contents.append(types.Content(
                role=h["role"],
                parts=[types.Part.from_text(text=h["text"])]
            ))
            
        # Layer current query with RAG context
        current_user_content = f"""[RETRIEVED KNOWLEDGE FROM MY WORKS]
{rag_context}

[MY CURRENT QUESTION]
{user_message}"""

        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=current_user_content)]
        ))
        
        # 3. Call Gemini async stream API
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            # Fallback if no API key is present
            fallback_text = "Hey! It looks like the GEMINI_API_KEY is not set. Under the hood, I'm just a model waiting for its key. Please set it in your .env file."
            yield fallback_text
            self.history.append({"role": "user", "text": user_message})
            self.history.append({"role": "model", "text": fallback_text})
            self.history = self.history[-8:]
            return

        client = genai.Client(api_key=api_key)
        from core.persona import get_system_prompt
        system_instruction = get_system_prompt()
        
        full_response_text = ""
        try:
            # We use client.aio for async calls
            response_stream = await client.aio.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    top_p=0.95,
                )
            )
            
            async for chunk in response_stream:
                if chunk.text:
                    full_response_text += chunk.text
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}", exc_info=True)
            yield f"⚠️ [Error generating response: {e}]"
            return
            
        # 4. Save to history (last 8 messages only)
        self.history.append({"role": "user", "text": user_message})
        self.history.append({"role": "model", "text": full_response_text})
        self.history = self.history[-8:]

    def close(self):
        """No persistent resources need saving with simple memory"""
        pass
