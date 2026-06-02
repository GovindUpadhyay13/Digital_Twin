import os
import pytest
from core.orchestrator import KarpathyTwinOrchestrator

def test_full_pipeline_orchestrator(mock_config):
    # Instantiate orchestrator using the test mock config
    orchestrator = KarpathyTwinOrchestrator(session_id="test_chat_session", config_path=mock_config)
    
    # We query about micrograd (which triggers mock answers if GEMINI_API_KEY is not set, or actual API call if it is)
    query = "Tell me about micrograd."
    result = orchestrator.chat(query, thread_id="test_thread")
    
    assert "response" in result
    assert len(result["response"]) > 0
    assert "sources" in result
    assert "is_valid" in result
    
    # Clean up
    orchestrator.close()
