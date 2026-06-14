
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.config_loader import load_config


def test_config_loader():
    """Test config loading (even if config doesn't exist)"""
    config = load_config()
    assert isinstance(config, dict)


def test_project_structure():
    """Test project structure exists"""
    assert project_root.exists()
    assert (project_root / "api").exists()
    assert (project_root / "agent").exists()
    assert (project_root / "core").exists()
    assert (project_root / "static").exists()
    assert (project_root / "requirements.txt").exists()



def test_thread_isolation():
    """Test that two different thread_ids have independent memory histories"""
    from core.orchestrator import KarpathyTwinOrchestrator
    orchestrator = KarpathyTwinOrchestrator(session_id="test_session")
    
    # Thread A
    orchestrator.chat(user_message="Hello, my name is Alice", thread_id="thread_A")
    state_A = orchestrator.graph.get_state({"configurable": {"thread_id": "thread_A"}})
    messages_A = state_A.values.get("messages", [])
    
    # Thread B
    orchestrator.chat(user_message="Hi, I am Bob", thread_id="thread_B")
    state_B = orchestrator.graph.get_state({"configurable": {"thread_id": "thread_B"}})
    messages_B = state_B.values.get("messages", [])
    
    assert any("Alice" in msg.content for msg in messages_A), "Thread A should remember Alice"
    assert not any("Bob" in msg.content for msg in messages_A), "Thread A should not know Bob"
    
    assert any("Bob" in msg.content for msg in messages_B), "Thread B should remember Bob"
    assert not any("Alice" in msg.content for msg in messages_B), "Thread B should not know Alice"


def test_retriever_logic():
    """Test source boosting and category weights"""
    from rag.retriever import Retriever
    import unittest.mock as mock
    
    retriever = Retriever(boost_primary=2.0)
    
    # Mock embedder and chroma collections
    retriever.embedder.embed_text = mock.MagicMock(return_value=mock.MagicMock(tolist=lambda: [0.1]*384))
    mock_collection = mock.MagicMock()
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "distances": [[0.1, 0.1]], # Equal distance -> equal base score
        "metadatas": [[
            {"type": "unknown"}, 
            {"type": "paper"} # Should get boosted
        ]]
    }
    retriever.collections = {k: mock_collection for k in retriever.collections}
    
    # Test retrieve_from_collection (source boosting)
    results = retriever.retrieve_from_collection("knowledge", "test query")
    if results:
        assert len(results) == 2
        assert results[1]["score"] > results[0]["score"], "Primary source should be boosted"
        assert round(results[1]["score"], 3) == round(results[0]["score"] * 2.0, 3)
    
    # Test retrieve (category weights)
    retriever.classify_query = mock.MagicMock(return_value=["technical"])
    all_results = retriever.retrieve("test query", top_k=2)
    assert len(all_results) > 0


def test_validation_routing():
    from agent.graph import route_after_validation
    assert route_after_validation({"is_valid": False, "severity": "major", "retry_count": 1}) == "retry"
    assert route_after_validation({"is_valid": False, "severity": "major", "retry_count": 2}) == "proceed"
    assert route_after_validation({"is_valid": True, "severity": "none", "retry_count": 0}) == "proceed"


if __name__ == "__main__":
    test_config_loader()
    test_project_structure()
    test_thread_isolation()
    test_retriever_logic()
    test_validation_routing()
    print("✅ All basic tests passed!")


