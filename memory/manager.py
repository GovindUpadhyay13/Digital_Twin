import os
import yaml
import uuid
from typing import List, Dict
from pathlib import Path
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.consolidator import MemoryConsolidator

class MemoryManager:
    def __init__(self, session_id: str = None, config_path: str = "config.yaml"):
        # Load configuration
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f).get("memory", {})
        else:
            self.config = {}

        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        # Initialize sub-modules
        window_size = self.config.get("short_term_window", 10)
        self.short_term = ShortTermMemory(window_size=window_size)
        
        persist_dir = self.config.get("db_path", "storage/chroma_db")
        if persist_dir.endswith(".db"):
            # If path points to sqlite sessions.db, let's extract folder name for chroma
            persist_dir = str(Path(persist_dir).parent / "chroma_db")
            
        self.long_term = LongTermMemory(persist_dir=persist_dir)
        self.consolidator = MemoryConsolidator()

    def add_user_message(self, message: str):
        """Record user statement in short term window"""
        self.short_term.add_turn("user", message)

    def add_assistant_message(self, response: str):
        """Record twin response in short term window and trigger consolidation if required"""
        self.short_term.add_turn("assistant", response)
        
        # Check if window overflow demands background consolidation
        if self.short_term.needs_compression():
            print("[MemoryManager] Conversation window overflow. Consolidating memory...")
            self.consolidator.consolidate_session(self.short_term, self.long_term, self.session_id)

    def get_all_user_facts(self) -> List[Dict]:
        """Fetch all facts recorded about the user"""
        return self.long_term.get_all_semantic()

    def get_formatted_memory_context(self, query: str) -> str:
        """Retrieves and formats long-term memory context relevant to the query"""
        # Retrieve context from long term storage
        semantic_memories = self.long_term.retrieve_semantic(query, top_k=3)
        episodic_memories = self.long_term.retrieve_episodic(query, top_k=2)
        important_moments = self.long_term.retrieve_important(query, top_k=2)
        
        # Assemble formatted context block
        context_parts = []
        
        if semantic_memories:
            context_parts.append("[Facts About the User]")
            for item in semantic_memories:
                context_parts.append(f"- {item['text']}")
                
        if episodic_memories:
            context_parts.append("\n[Context from Previous Sessions]")
            for item in episodic_memories:
                topics = item['metadata'].get('topics', 'general')
                context_parts.append(f"- Session {item['metadata'].get('session_id', 'unknown')} (Topics: {topics}): {item['text']}")
                
        if important_moments:
            context_parts.append("\n[Memorable Interactions/Milestones]")
            for item in important_moments:
                context_parts.append(f"- Exchange: {item['text']}")
                if item.get("reason"):
                    context_parts.append(f"  (Context: {item['reason']})")
                    
        if not context_parts:
            return "No prior conversational memory found matching this context."
            
        return "\n".join(context_parts)

    def save_session(self):
        """Explicitly triggers episodic saving at session end"""
        self.consolidator.save_end_of_session(self.short_term, self.long_term, self.session_id)

    def clear_memory(self):
        """Clears all session and persistent memories"""
        self.short_term.clear()
        self.long_term.clear_all()
