
import os
import re
from typing import Optional, List, Dict
from pathlib import Path
from rag.embedder import Embedder


class Retriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "storage/chroma_db",
        top_k: int = 10,
        collection_name: str = "karpathy_knowledge",
        boost_primary: float = 1.3,
        *args,
        **kwargs,
    ):
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.embedder = Embedder(model_name)
        
        # Load ChromaDB
        try:
            import chromadb
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collections = {
                "knowledge": self.client.get_or_create_collection("karpathy_knowledge"),
                "analogies": self.client.get_or_create_collection("karpathy_analogies"),
                "reasoning_traces": self.client.get_or_create_collection("karpathy_reasoning"),
                "teaching_style": self.client.get_or_create_collection("karpathy_teaching"),
                "quotes": self.client.get_or_create_collection("karpathy_quotes")
            }
        except Exception as e:
            print(f"ChromaDB init failed, using fallback: {e}")

    def classify_query(self, query: str) -> List[str]:
        """Step 1: Classify the user's query into categories"""
        categories = {
            "conceptual": ["what", "why", "how", "explain", "concept", "principle", "theory"],
            "technical": ["code", "implement", "algorithm", "model", "build", "debug"],
            "research": ["paper", "scaling", "experiment", "study", "research"],
            "debugging": ["error", "bug", "problem", "issue", "fix"],
            "opinion": ["think", "believe", "opinion", "view", "prefer"],
            "career": ["career", "job", "path", "advice", "learn"],
            "philosophy": ["philosophy", "ethics", "future", "agi", "intelligence"]
        }
        
        matches = []
        query_lower = query.lower()
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matches.append(category)
                    break
        
        return matches if matches else ["conceptual"]

    def retrieve_from_collection(
        self,
        collection_name: str,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """Retrieve from a specific collection"""
        query_embedding = self.embedder.embed_text(query)
        
        try:
            results = self.collections[collection_name].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            processed = []
            if results and results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0]
                
                for i in range(len(ids)):
                    score = 1.0 / (1.0 + dists[i])
                    processed.append({
                        "chunk_id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i],
                        "score": score,
                        "collection": collection_name
                    })
                    
            return processed
        except Exception as e:
            print(f"Retrieval failed for {collection_name}: {e}")
            return []

    def rerank(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """Step 3: Simple reranking based on keyword match + score"""
        query_lower = query.lower()
        for res in results:
            text_lower = res["text"].lower()
            keyword_score = sum(
                1 for token in re.findall(r'\b\w+\b', query_lower)
                if token in text_lower
            )
            res["score"] *= 1.0 + (keyword_score / len(query_lower.split()))
        
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        return results_sorted[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        year_range: Optional[tuple] = None,
    ) -> List[Dict]:
        """Run full multi-collection retrieval pipeline"""
        k = top_k or self.top_k
        
        # Step 1: Classify query
        categories = self.classify_query(query)
        
        # Step 2: Retrieve from all relevant collections
        all_results = []
        for col in ["knowledge", "analogies", "reasoning_traces", "teaching_style", "quotes"]:
            col_results = self.retrieve_from_collection(col, query, top_k=3)
            all_results.extend(col_results)
        
        # Step 3: Rerank
        final_results = self.rerank(query, all_results, top_k=k)
        
        return final_results

    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved context nicely for the LLM"""
        if not results:
            return "No relevant sources found."
            
        grouped = {}
        for res in results:
            col = res.get("collection", "unknown")
            if col not in grouped:
                grouped[col] = []
            grouped[col].append(res)
            
        parts = []
        for col_name, items in grouped.items():
            parts.append(f"--- {col_name.replace('_', ' ').title()} ---\n")
            for i, item in enumerate(items):
                parts.append(f"[{i+1}] {item['text']}\n")
                
        return "\n".join(parts)
