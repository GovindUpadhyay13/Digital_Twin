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
    assert (project_root / "core").exists()
    assert (project_root / "static").exists()
    assert (project_root / "requirements.txt").exists()

def test_retriever_logic():
    """Test simplified retriever query logic"""
    from rag.retriever import Retriever
    import unittest.mock as mock
    
    retriever = Retriever()
    
    # Mock embedder and chroma collection
    retriever.embedder.embed_text = mock.MagicMock(return_value=mock.MagicMock(tolist=lambda: [0.1]*384))
    
    mock_collection = mock.MagicMock()
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[
            {"type": "unknown"}, 
            {"type": "paper"}
        ]]
    }
    retriever.collection = mock_collection
    
    results = retriever.retrieve("test query", top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "id1"
    assert results[0]["score"] == 0.9  # 1.0 - 0.1

if __name__ == "__main__":
    test_config_loader()
    test_project_structure()
    test_retriever_logic()
    print("✅ All basic tests passed!")
