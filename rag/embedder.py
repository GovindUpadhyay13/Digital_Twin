import numpy as np
from typing import Dict, List

_embedder_cache: Dict[str, "Embedder"] = {}

def get_embedder(model_name: str):
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = Embedder(model_name)
    return _embedder_cache[model_name]

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384 # Default dimension for all-MiniLM-L6-v2
        
        if "nomic" in model_name:
            self.embedding_dim = 768
            
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model '{model_name}'...")
            self.model = SentenceTransformer(model_name)
            # Find the actual dimensions
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Embedding model loaded successfully. Dimensions: {self.embedding_dim}")
        except Exception as e:
            print(f"[WARN] Could not load SentenceTransformer model '{model_name}': {e}. Using mock embedding fallback.")
            self.model = None

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        if self.model:
            try:
                # sentence-transformers encode returns numpy array by default
                emb = self.model.encode(text, convert_to_numpy=True)
                if isinstance(emb, list):
                    return np.array(emb, dtype=np.float32)
                return emb.astype(np.float32)
            except Exception as e:
                print(f"[WARN] Encoding failed: {e}. Generating mock embedding.")
        
        # Generate deterministic mock embedding based on string hash
        return self._generate_mock_embedding(text)

    def embed_batch(self, texts: List[str] if 'List' in globals() else list) -> np.ndarray:
        """Embed a batch of texts efficiently."""
        if self.model:
            try:
                emb = self.model.encode(texts, batch_size=32, convert_to_numpy=True)
                if isinstance(emb, list):
                    return np.array(emb, dtype=np.float32)
                return emb.astype(np.float32)
            except Exception as e:
                print(f"[WARN] Batch encoding failed: {e}. Generating mock embeddings.")
        
        return np.stack([self._generate_mock_embedding(t) for t in texts])

    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic mock normalized unit vector of size embedding_dim"""
        state = sum(ord(c) for c in text)
        rng = np.random.default_rng(state)
        vec = rng.standard_normal(self.embedding_dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
