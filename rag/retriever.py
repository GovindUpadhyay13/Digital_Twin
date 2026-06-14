
import os
import re
from typing import Optional, List, Dict
from pathlib import Path
from rag.embedder import Embedder

CATEGORY_COLLECTION_WEIGHTS = {
    "technical": {"knowledge": 1.5, "reasoning_traces": 1.2, "analogies": 0.8, "teaching_style": 0.8, "quotes": 0.5},
    "opinion":   {"quotes": 1.5, "reasoning_traces": 1.3, "knowledge": 0.8, "analogies": 0.8, "teaching_style": 0.8},
    "conceptual":{"teaching_style": 1.3, "analogies": 1.3, "knowledge": 1.0, "reasoning_traces": 1.0, "quotes": 0.7},
    "research":  {"knowledge": 1.5, "reasoning_traces": 1.2, "quotes": 0.6, "analogies": 0.7, "teaching_style": 0.7},
    "career":    {"quotes": 1.3, "reasoning_traces": 1.2, "teaching_style": 1.1, "knowledge": 0.8, "analogies": 0.8},
    "debugging": {"knowledge": 1.4, "reasoning_traces": 1.3, "teaching_style": 1.0, "analogies": 0.7, "quotes": 0.5},
    "philosophy":{"quotes": 1.4, "reasoning_traces": 1.3, "knowledge": 0.9, "teaching_style": 0.9, "analogies": 0.8},
}


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
        self.boost_primary = boost_primary
        from rag.embedder import get_embedder
        self.embedder = get_embedder(model_name)
        
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
                    
                    source_type = metas[i].get("source_type", metas[i].get("type", "unknown"))
                    if source_type in {"paper", "github_repo", "official_blog", "blog"}:
                        score *= self.boost_primary
                        
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
        
        collection_weights = {}
        for col in ["knowledge", "analogies", "reasoning_traces", "teaching_style", "quotes"]:
            weights = [CATEGORY_COLLECTION_WEIGHTS.get(cat, {}).get(col, 1.0) for cat in categories]
            collection_weights[col] = sum(weights) / len(weights) if weights else 1.0
        
        # Step 2: Retrieve from all relevant collections
        all_results = []
        for col in ["knowledge", "analogies", "reasoning_traces", "teaching_style", "quotes"]:
            col_results = self.retrieve_from_collection(col, query, top_k=3)
            for res in col_results:
                res["score"] *= collection_weights[col]
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
        global_index = 1
        for col_name, items in grouped.items():
            parts.append(f"--- {col_name.replace('_', ' ').title()} ---\n")
            for item in items:
                parts.append(f"[{global_index}] {item['text']}\n")
                global_index += 1
                
        return "\n".join(parts)
