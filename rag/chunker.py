import re
from typing import List, Dict

class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = None
        
        try:
            from transformers import AutoTokenizer
            # Set local files only to false, but keep a try/except around the download
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", local_files_only=False)
        except Exception as e:
            print(f"[WARN] Could not load Transformers tokenizer: {e}. Falling back to character/word-based splitting.")

    def chunk(self, text: str, source_metadata: Dict) -> List[Dict]:
        """
        Chunk text and return list of chunks with metadata.
        Each chunk dict contains:
        {
            "text": "chunk text...",
            "metadata": {...},
            "chunk_index": 0,
            "token_count": 512,
        }
        """
        if not text or not text.strip():
            return []

        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                chunks = []
                start_idx = 0
                chunk_idx = 0
                
                while start_idx < len(tokens):
                    end_idx = min(start_idx + self.chunk_size, len(tokens))
                    chunk_tokens = tokens[start_idx:end_idx]
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    # Merge metadata
                    chunk_metadata = {
                        **source_metadata,
                        "chunk_index": chunk_idx,
                        "token_count": len(chunk_tokens),
                    }
                    
                    # Ensure year is present as integer (helpful for filtering in ChromaDB)
                    if "year" in chunk_metadata:
                        try:
                            chunk_metadata["year"] = int(chunk_metadata["year"])
                        except:
                            pass
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata,
                        "chunk_index": chunk_idx,
                    })
                    
                    # Move start forward (with overlap)
                    start_idx = end_idx - self.overlap
                    if start_idx >= len(tokens) or end_idx == len(tokens):
                        break
                    chunk_idx += 1
                
                return chunks
            except Exception as e:
                print(f"[WARN] Tokenizer chunking failed: {e}. Falling back to word-split.")
                
        # Word-split fallback (approx 1.3 words per token)
        words = text.split()
        word_chunk_size = int(self.chunk_size * 0.75)
        word_overlap = int(self.overlap * 0.75)
        
        chunks = []
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + word_chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunk_metadata = {
                **source_metadata,
                "chunk_index": chunk_idx,
                "token_count": len(chunk_words), # rough estimate
            }
            
            if "year" in chunk_metadata:
                try:
                    chunk_metadata["year"] = int(chunk_metadata["year"])
                except:
                    pass

            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
                "chunk_index": chunk_idx,
            })
            
            start_idx = end_idx - word_overlap
            if start_idx >= len(words) or end_idx == len(words):
                break
            chunk_idx += 1
            
        return chunks
