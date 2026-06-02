from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """State that flows through the LangGraph"""
    
    # Input
    user_message: str
    thread_id: str
    
    # RAG context
    rag_results: List[Dict[str, Any]]
    rag_context: str
    
    # Memory context
    memory_context: str
    user_facts: List[str]
    
    # Timeline
    timeline_context: str
    year_range: Optional[tuple]
    
    # Conversation history
    messages: List[BaseMessage]
    conversation_summary: str
    
    # Final output
    response: str
    sources: List[Dict]
    
    # Metadata
    is_valid: bool
    validation_issues: List[str]
