import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from rag.embedder import Embedder

class LongTermMemory:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "storage/chroma_db",
    ):
        self.persist_dir = persist_dir
        from rag.embedder import get_embedder
        self.embedder = get_embedder(model_name)
        self.use_fallback = False
        self.client = None
        
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.episodic = self.client.get_or_create_collection("episodic_memory")
            self.semantic = self.client.get_or_create_collection("semantic_memory")
            self.important = self.client.get_or_create_collection("important_moments")
            print("ChromaDB long-term memory collections initialized.")
        except Exception as e:
            print(f"[WARN] ChromaDB failed for long term memory: {e}. Standardizing on JSON fallback.")
            self.use_fallback = True
            self.fallback_path = Path(persist_dir) / "long_term_memory.json"
            self._load_fallback()

    def _load_fallback(self):
        """Loads memory collections from JSON file"""
        if self.fallback_path.exists():
            with open(self.fallback_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "episodic": [], # list of {id, text, embedding, metadata}
                "semantic": [], # list of {id, text, embedding, metadata}
                "important": [] # list of {id, text, embedding, metadata}
            }

    def _save_fallback(self):
        """Saves memory collections to JSON file"""
        # Ensure parent directories exist
        os.makedirs(self.fallback_path.parent, exist_ok=True)
        with open(self.fallback_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    # ─── EPISODIC MEMORY ────────────────────────────────────────────
    
    def store_episodic(
        self,
        session_id: str,
        summary: str,
        topics: List[str],
        turn_count: int,
        user_sentiment: str = ""
    ):
        """Store a session summary"""
        embedding = self.embedder.embed_text(summary).tolist()
        metadata = {
            "session_id": session_id,
            "topics": ",".join(topics) if isinstance(topics, list) else topics,
            "turn_count": turn_count,
            "user_sentiment": user_sentiment,
        }
        
        if self.use_fallback:
            # Overwrite if session_id already exists
            self.data["episodic"] = [item for item in self.data["episodic"] if item["id"] != session_id]
            self.data["episodic"].append({
                "id": session_id,
                "text": summary,
                "embedding": embedding,
                "metadata": metadata
            })
            self._save_fallback()
        else:
            # Delete if exists
            try:
                self.episodic.delete(ids=[session_id])
            except:
                pass
            self.episodic.add(
                ids=[session_id],
                embeddings=[embedding],
                documents=[summary],
                metadatas=[metadata]
            )

    def retrieve_episodic(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant past sessions"""
        query_emb = self.embedder.embed_text(query)
        
        if self.use_fallback:
            return self._retrieve_fallback_collection("episodic", query_emb, top_k)
            
        if self.episodic.count() == 0:
            return []
            
        results = self.episodic.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k, self.episodic.count()),
            include=["documents", "metadatas"]
        )
        
        processed = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                processed.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                })
        return processed

    # ─── SEMANTIC MEMORY ────────────────────────────────────────────
    
    def store_semantic(
        self,
        fact_id: str,
        fact: str,
        category: str,
        confidence: float = 0.8
    ):
        """Store a fact about the user"""
        embedding = self.embedder.embed_text(fact).tolist()
        metadata = {
            "category": category,
            "confidence": confidence,
        }
        
        if self.use_fallback:
            self.data["semantic"] = [item for item in self.data["semantic"] if item["id"] != fact_id]
            self.data["semantic"].append({
                "id": fact_id,
                "text": fact,
                "embedding": embedding,
                "metadata": metadata
            })
            self._save_fallback()
        else:
            try:
                self.semantic.delete(ids=[fact_id])
            except:
                pass
            self.semantic.add(
                ids=[fact_id],
                embeddings=[embedding],
                documents=[fact],
                metadatas=[metadata]
            )

    def retrieve_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant user facts"""
        query_emb = self.embedder.embed_text(query)
        
        if self.use_fallback:
            return self._retrieve_fallback_collection("semantic", query_emb, top_k)
            
        if self.semantic.count() == 0:
            return []
            
        results = self.semantic.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k, self.semantic.count()),
            include=["documents", "metadatas"]
        )
        
        processed = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                processed.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                })
        return processed

    def get_all_semantic(self) -> List[Dict]:
        """Get ALL user facts (always injected into prompt)"""
        if self.use_fallback:
            return [{"fact": item["text"], "meta": item["metadata"]} for item in self.data["semantic"]]
            
        if self.semantic.count() == 0:
            return []
            
        results = self.semantic.get(include=["documents", "metadatas"])
        processed = []
        if results and results["ids"]:
            for i in range(len(results["ids"])):
                processed.append({
                    "fact": results["documents"][i],
                    "meta": results["metadatas"][i]
                })
        return processed

    # ─── IMPORTANT MOMENTS ──────────────────────────────────────────
    
    def store_important(
        self,
        moment_id: str,
        exchange: str,
        importance_reason: str,
        session_id: str
    ):
        """Store an important moment"""
        embedding = self.embedder.embed_text(exchange).tolist()
        metadata = {
            "importance_reason": importance_reason,
            "session_id": session_id,
        }
        
        if self.use_fallback:
            self.data["important"] = [item for item in self.data["important"] if item["id"] != moment_id]
            self.data["important"].append({
                "id": moment_id,
                "text": exchange,
                "embedding": embedding,
                "metadata": metadata
            })
            self._save_fallback()
        else:
            try:
                self.important.delete(ids=[moment_id])
            except:
                pass
            self.important.add(
                ids=[moment_id],
                embeddings=[embedding],
                documents=[exchange],
                metadatas=[metadata]
            )

    def retrieve_important(self, query: str, top_k: int = 2) -> List[Dict]:
        """Retrieve relevant important moments"""
        query_emb = self.embedder.embed_text(query)
        
        if self.use_fallback:
            results = self._retrieve_fallback_collection("important", query_emb, top_k)
            return [
                {
                    "text": r["text"],
                    "reason": r["metadata"].get("importance_reason")
                }
                for r in results
            ]
            
        if self.important.count() == 0:
            return []
            
        results = self.important.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k, self.important.count()),
            include=["documents", "metadatas"]
        )
        
        processed = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                processed.append({
                    "text": results["documents"][0][i],
                    "reason": results["metadatas"][0][i].get("importance_reason"),
                })
        return processed

    # ─── UTILITIES ──────────────────────────────────────────────────
    
    def _retrieve_fallback_collection(self, key: str, query_emb: np.ndarray, top_k: int) -> List[Dict]:
        """Generic vector search inside local collections"""
        items = self.data[key]
        if not items:
            return []
            
        scored_items = []
        for item in items:
            db_emb = np.array(item["embedding"])
            dot_product = float(np.dot(query_emb, db_emb))
            similarity = 0.5 + 0.5 * dot_product
            scored_items.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "score": similarity
            })
            
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]

    def get_memory_stats(self) -> Dict:
        """Return memory statistics"""
        if self.use_fallback:
            return {
                "episodic_count": len(self.data["episodic"]),
                "semantic_count": len(self.data["semantic"]),
                "important_count": len(self.data["important"]),
            }
        return {
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
            "important_count": self.important.count(),
        }
    
    def clear_all(self):
        """Clear all memories"""
        if self.use_fallback:
            self.data = {"episodic": [], "semantic": [], "important": []}
            self._save_fallback()
        else:
            try:
                self.episodic.delete(self.episodic.get(include=[])["ids"])
            except:
                pass
            try:
                self.semantic.delete(self.semantic.get(include=[])["ids"])
            except:
                pass
            try:
                self.important.delete(self.important.get(include=[])["ids"])
            except:
                pass
