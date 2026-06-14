import os
from typing import Optional, List, Dict
from rag.embedder import get_embedder

class Retriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./data/chroma",
        collection_name: str = "karpathy_knowledge",
        top_k: int = 5,
        *args,
        **kwargs,
    ):
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.collection_name = collection_name
        self.embedder = get_embedder(model_name)
        
        # Load ChromaDB
        import chromadb
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Simplified RAG retrieval: embed query -> top-5 cosine similarity -> done"""
        k = top_k or self.top_k
        
        try:
            query_embedding = self.embedder.embed_text(query)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            processed = []
            if results and results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0]
                
                for i in range(len(ids)):
                    # Cosine distance to similarity: 1 - distance, or 1.0 / (1.0 + distance)
                    score = 1.0 - dists[i]
                    processed.append({
                        "chunk_id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i],
                        "score": score
                    })
            return processed
        except Exception as e:
            print(f"Retrieval failed: {e}")
            return []

    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved context nicely for the LLM"""
        if not results:
            return "No relevant sources found."
            
        parts = []
        for i, item in enumerate(results, 1):
            parts.append(f"[{i}] {item['text']}\n")
                
        return "\n".join(parts)
