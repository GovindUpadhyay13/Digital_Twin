import os
import shutil
import pytest
from pathlib import Path
from rag.retriever import Retriever
from memory.manager import MemoryManager
from timeline.engine import TimelineEngine
from agent.persona_validator import PersonaValidator

@pytest.fixture
def test_storage_dir(tmp_path):
    """Fixture that creates a temporary storage directory for tests and cleans up afterwards"""
    db_dir = tmp_path / "storage"
    db_dir.mkdir()
    yield str(db_dir)
    if db_dir.exists():
        try:
            shutil.rmtree(db_dir)
        except OSError:
            # On Windows, open db handles (ChromaDB/sqlite) might temporarily block deletion.
            # Pytest's tmp_path fixture automatically cleans up old run dirs, so we can ignore it safely.
            pass

@pytest.fixture
def mock_config(test_storage_dir):
    """Fixture that generates a temporary config path"""
    import yaml
    config_data = {
        "data_collection": {
            "output_dir": str(Path(test_storage_dir) / "raw"),
            "github_repos": ["karpathy/micrograd"]
        },
        "rag": {
            "persist_dir": test_storage_dir,
            "collection_name": "test_knowledge",
            "chunk_size": 256,
            "overlap": 20,
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 3,
            "boost_primary": 1.3
        },
        "memory": {
            "short_term_window": 4,
            "db_path": str(Path(test_storage_dir) / "test_sessions.db")
        }
    }
    
    config_file = Path(test_storage_dir) / "test_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
        
    return str(config_file)

@pytest.fixture
def retriever(test_storage_dir):
    return Retriever(
        model_name="all-MiniLM-L6-v2",
        persist_dir=test_storage_dir,
        collection_name="test_knowledge"
    )

@pytest.fixture
def memory_manager(mock_config):
    return MemoryManager(session_id="test_session_id", config_path=mock_config)

@pytest.fixture
def timeline_engine():
    return TimelineEngine()

@pytest.fixture
def validator():
    return PersonaValidator()
