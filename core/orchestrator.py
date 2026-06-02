import os
import yaml
from pathlib import Path
from typing import Dict, Any

from core.config_loader import load_config, load_env
from rag.retriever import Retriever
from memory.manager import MemoryManager
from timeline.engine import TimelineEngine
from core.prompt_builder import PromptBuilder
from agent.persona_validator import PersonaValidator
from agent.graph import build_graph

class KarpathyTwinOrchestrator:
    def __init__(self, session_id: str = None, config_path: str = "config.yaml"):
        # Load environment variables
        load_env()
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Instantiate pipeline components
        rag_conf = self.config.get("rag", {})
        self.retriever = Retriever(
            model_name=rag_conf.get("embedding_model", "all-MiniLM-L6-v2"),
            persist_dir=rag_conf.get("persist_dir", "storage/chroma_db"),
            collection_name=rag_conf.get("collection_name", "karpathy_knowledge"),
            top_k=rag_conf.get("top_k", 5),
            boost_primary=rag_conf.get("boost_primary", 1.3)
        )
        
        self.memory = MemoryManager(session_id=session_id, config_path=config_path)
        self.timeline = TimelineEngine()
        self.prompt_builder = PromptBuilder()
        self.validator = PersonaValidator()
        
        # Build compiled LangGraph
        self.graph = build_graph(
            retriever=self.retriever,
            memory=self.memory,
            timeline=self.timeline,
            prompt_builder=self.prompt_builder,
            validator=self.validator
        )
        
        print(f"Digital Twin Orchestrator compiled successfully. Session ID: {self.memory.session_id}")

    def chat(self, user_message: str, thread_id: str = "default_thread") -> Dict[str, Any]:
        """
        Runs the user query through the LangGraph agent state graph.
        
        Args:
            user_message: Raw user query string.
            thread_id: Unique thread identifier for conversation memory tracking.
            
        Returns:
            Dict containing the assistant response, matched sources, and validation states.
        """
        inputs = {
            "user_message": user_message,
            "thread_id": thread_id,
            "messages": [], # Will be restored by MemorySaver checkpoint
            "conversation_summary": "",
            "is_valid": True,
            "validation_issues": []
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke state graph
        state_output = self.graph.invoke(inputs, config)
        
        return {
            "response": state_output.get("response", ""),
            "sources": state_output.get("sources", []),
            "is_valid": state_output.get("is_valid", True),
            "validation_issues": state_output.get("validation_issues", [])
        }

    def close(self):
        """Close session and save episodic memory"""
        self.memory.save_session()
