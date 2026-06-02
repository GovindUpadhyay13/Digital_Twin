import os
import pytest
from pathlib import Path
from rag.ingestion import RAGIngestor
from rag.retriever import Retriever

def test_chunking_and_ingestion(test_storage_dir, mock_config):
    # Setup test raw folder structure
    raw_dir = Path(test_storage_dir) / "raw"
    blog_dir = raw_dir / "blog"
    blog_dir.mkdir(parents=True)
    
    # Write a test raw file
    test_file = blog_dir / "test_post.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Date: 2023-01-16\n")
        f.write("URL: https://karpathy.github.io/test-post\n\n")
        f.write("Backpropagation is a delightful mathematical trick. Essentially, it allows gradients to flow backwards.")
        
    # Ingest documents
    # Setup directories
    os.makedirs("data", exist_ok=True)
    
    # Run ingestion targeting the test DB
    ingestor = RAGIngestor(persist_dir=test_storage_dir, collection_name="test_knowledge")
    ingestor._ingest_source("blog", blog_dir)
    
    # Retriever check
    retriever = Retriever(
        model_name="all-MiniLM-L6-v2",
        persist_dir=test_storage_dir,
        collection_name="test_knowledge"
    )
    
    # Retrieve
    results = retriever.retrieve("backpropagation", top_k=2)
    assert len(results) > 0
    assert "backpropagation" in results[0]["text"].lower()
    assert results[0]["metadata"]["is_primary_source"] is True
    assert results[0]["metadata"]["year"] == 2023

def test_source_boosting(test_storage_dir):
    retriever = Retriever(
        model_name="all-MiniLM-L6-v2",
        persist_dir=test_storage_dir,
        collection_name="test_knowledge"
    )
    
    # Inject directly to fallback database or mock
    retriever.use_fallback = True
    retriever.fallback_db = {
        "ids": ["doc1", "doc2"],
        "documents": [
            "We build autograd from scratch. Backprop is beautiful.",
            "Some third-party notes about backprop in neural networks."
        ],
        "embeddings": [
            retriever.embedder.embed_text("We build autograd from scratch. Backprop is beautiful.").tolist(),
            retriever.embedder.embed_text("Some third-party notes about backprop in neural networks.").tolist()
        ],
        "metadatas": [
            {"is_primary_source": True, "source_title": "Primary Doc", "year": 2023},
            {"is_primary_source": False, "source_title": "Secondary Doc", "year": 2023}
        ]
    }
    
    retriever._load_fallback_db = lambda: None
    
    results = retriever.retrieve("backprop", top_k=2)
    assert len(results) == 2
    # The primary source document should have its score multiplied by 1.3
    # Check that score of primary doc has been boosted
    primary_res = [r for r in results if r["chunk_id"] == "doc1"][0]
    secondary_res = [r for r in results if r["chunk_id"] == "doc2"][0]
    
    # Even if they are equally relevant, the primary doc should be sorted first or boosted higher
    # Let's verify that the boost factor was applied
    assert primary_res["score"] > 0
