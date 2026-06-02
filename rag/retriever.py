import os
import json
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
from rag.embedder import Embedder

class Retriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "storage/chroma_db",
        collection_name: str = "karpathy_knowledge",
        top_k: int = 5,
        boost_primary: float = 1.3,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.boost_primary = boost_primary
        self.embedder = Embedder(model_name)
        self.use_fallback = False
        self.collection = None
        self.fallback_db_path = Path(persist_dir) / "fallback_db.json"
        
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            self.use_fallback = True
            self._load_fallback_db()

    def _load_fallback_db(self):
        """Load fallback database"""
        if self.fallback_db_path.exists():
            with open(self.fallback_db_path, "r", encoding="utf-8") as f:
                self.fallback_db = json.load(f)
        else:
            self.fallback_db = {"documents": [], "embeddings": [], "metadatas": [], "ids": []}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        year_range: Optional[tuple] = None,
    ) -> List[dict]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query: User's question
            top_k: Override default
            year_range: (start_year, end_year) to filter
        
        Returns:
            List of result dicts with text, metadata, score
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_text(query)
        
        if self.use_fallback:
            return self._retrieve_fallback(query_embedding, k, year_range)
            
        # ChromaDB flow
        if self.collection.count() == 0:
            print("[Retriever] Warning: ChromaDB collection is empty!")
            return []
            
        # Build metadata filter
        where_filter = None
        if year_range:
            start_year, end_year = year_range
            where_filter = {
                "$and": [
                    {"year": {"$gte": int(start_year)}},
                    {"year": {"$lte": int(end_year)}}
                ]
            }
            
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(k * 2, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[Retriever] Query error: {e}. Attempting query without filters.")
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(k * 2, self.collection.count()),
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as ex:
                print(f"[Retriever] Final Query error: {ex}")
                return []
            
        processed = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for i in range(len(ids)):
                # L2 distance to similarity score
                similarity = 1.0 / (1.0 + distances[i])
                processed.append({
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "score": similarity,
                })
                
        # Apply boosting & sorting
        processed = self._apply_source_boosting(processed)
        processed.sort(key=lambda x: x["score"], reverse=True)
        return processed[:k]

    def _retrieve_fallback(self, query_emb: np.ndarray, k: int, year_range: Optional[tuple]) -> List[dict]:
        """Perform vector retrieval using the JSON fallback DB"""
        self._load_fallback_db()
        if not self.fallback_db["documents"]:
            return []
            
        processed = []
        for i in range(len(self.fallback_db["ids"])):
            metadata = self.fallback_db["metadatas"][i]
            
            # Apply year range filters
            if year_range:
                start_year, end_year = year_range
                doc_year = metadata.get("year", 2023)
                if not (start_year <= doc_year <= end_year):
                    continue
                    
            db_emb = np.array(self.fallback_db["embeddings"][i])
            # Cosine similarity = dot product of normalized vectors
            dot_product = float(np.dot(query_emb, db_emb))
            # Shift cosine similarity from [-1, 1] to [0, 1]
            similarity = 0.5 + 0.5 * dot_product
            
            processed.append({
                "chunk_id": self.fallback_db["ids"][i],
                "text": self.fallback_db["documents"][i],
                "metadata": metadata,
                "score": similarity,
            })
            
        processed = self._apply_source_boosting(processed)
        processed.sort(key=lambda x: x["score"], reverse=True)
        return processed[:k]

    def _apply_source_boosting(self, results: List[dict]) -> List[dict]:
        """Boost scores for primary source chunks"""
        for result in results:
            if result["metadata"].get("is_primary_source"):
                result["score"] *= self.boost_primary
        return results

    def format_context(self, results: List[dict]) -> str:
        """Format results into context string for LLM prompt"""
        if not results:
            return "No relevant sources found in my works."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source_title", "Unknown")
            year = result["metadata"].get("year", "")
            source_type = result["metadata"].get("source_type", "")
            
            header = f"[Source {i}: {source}"
            if year:
                header += f" ({year})"
            if source_type:
                header += f" — {source_type.upper()}"
            header += "]"
            
            context_parts.append(f"{header}\n{result['text']}")
        
        return "\n\n---\n\n".join(context_parts)
