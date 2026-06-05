import os
import json
from pathlib import Path
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.preprocessor import DocumentPreprocessor

class RAGIngestor:
    def __init__(self, persist_dir: str = "storage/chroma_db", collection_name: str = "karpathy_knowledge"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunker = TextChunker(chunk_size=512, overlap=50)
        self.embedder = Embedder("all-MiniLM-L6-v2")
        self.use_fallback = False
        self.collection = None
        
        try:
            import chromadb
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            print(f"ChromaDB initialized at {persist_dir}.")
        except Exception as e:
            print(f"[WARN] ChromaDB initialization failed: {e}. Standardizing on JSON vector store fallback.")
            self.use_fallback = True
            self.fallback_db_path = Path(persist_dir) / "fallback_db.json"
            os.makedirs(self.fallback_dir, exist_ok=True) if hasattr(self, "fallback_dir") else os.makedirs(Path(persist_dir), exist_ok=True)
            self._load_fallback_db()

    def _load_fallback_db(self):
        """Load the local JSON database fallback"""
        if self.fallback_db_path.exists():
            with open(self.fallback_db_path, "r", encoding="utf-8") as f:
                self.fallback_db = json.load(f)
        else:
            self.fallback_db = {"documents": [], "embeddings": [], "metadatas": [], "ids": []}

    def _save_fallback_db(self):
        """Save the local JSON database fallback"""
        with open(self.fallback_db_path, "w", encoding="utf-8") as f:
            json.dump(self.fallback_db, f, indent=2)

    def ingest_all(self):
        """Ingest all documents from data/raw/"""
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            print(f"[WARN] Raw data directory '{raw_dir}' does not exist. Please run data collection first.")
            return
            
        for source_type in ["twitter", "blog", "papers", "github", "biography", "transcripts"]:
            source_dir = raw_dir / source_type
            if source_dir.exists():
                self._ingest_source(source_type, source_dir)
            elif source_type == "biography":
                # Also check root data/raw for any .md or .txt files
                for file_path in list(raw_dir.glob("*.md")) + list(raw_dir.glob("*.txt")):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        self._process_document("biography", file_path.name, content)
                    except Exception as e:
                        print(f"[ERROR] Failed to ingest {file_path.name}: {e}")
        
        # Also check separate transcripts directory in project root
        transcripts_root_dir = Path("transcripts")
        if transcripts_root_dir.exists():
            self._ingest_transcripts_from_root(transcripts_root_dir)

    def _process_document(self, source_type: str, doc_name: str, content: str):
        """Helper to preprocess, chunk, embed, and index a document text"""
        # Enrich metadata using Preprocessor
        metadata = DocumentPreprocessor.enrich_metadata(source_type, doc_name, content)
        
        # Chunk
        chunks = self.chunker.chunk(content, metadata)
        if not chunks:
            return
        
        texts = [chunk["text"] for chunk in chunks]
        embeddings = [self.embedder.embed_text(txt).tolist() for txt in texts]
        stem = doc_name.replace(".txt", "")
        ids = [f"{stem}_chunk_{chunk['chunk_index']}" for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        if self.use_fallback:
            for i in range(len(ids)):
                # If ID already exists, overwrite
                if ids[i] in self.fallback_db["ids"]:
                    idx = self.fallback_db["ids"].index(ids[i])
                    self.fallback_db["documents"][idx] = texts[i]
                    self.fallback_db["embeddings"][idx] = embeddings[i]
                    self.fallback_db["metadatas"][idx] = metadatas[i]
                else:
                    self.fallback_db["ids"].append(ids[i])
                    self.fallback_db["documents"].append(texts[i])
                    self.fallback_db["embeddings"].append(embeddings[i])
                    self.fallback_db["metadatas"].append(metadatas[i])
            self._save_fallback_db()
        else:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        
        print(f"[OK] Ingested {doc_name}: {len(chunks)} chunks")

    def _ingest_source(self, source_type: str, source_dir: Path):
        """Ingest all files from a single source directory, supporting both .txt and .jsonl"""
        file_paths = list(source_dir.glob("*.txt")) + list(source_dir.glob("*.jsonl"))
        for file_path in file_paths:
            if file_path.suffix == '.jsonl':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for idx, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception as je:
                            print(f"[WARN] Invalid JSON line in {file_path.name}: {je}")
                            continue
                            
                        # Handle Twitter JSONL item
                        if source_type == "twitter":
                            text = item.get("text") or item.get("full_text") or item.get("content")
                            if not text:
                                continue
                            item_id = item.get("id") or item.get("id_str") or f"{idx}"
                            created_at = item.get("created_at") or item.get("date") or "2023-01-01"
                            likes = item.get("likes") or item.get("favorite_count") or 0
                            retweets = item.get("retweets") or item.get("retweet_count") or 0
                            
                            content = (
                                f"# Tweet by @karpathy\n"
                                f"Date: {created_at}\n"
                                f"URL: https://twitter.com/karpathy/status/{item_id}\n"
                                f"Likes: {likes}\n"
                                f"Retweets: {retweets}\n\n"
                                f"{text}"
                            )
                            doc_name = f"tweet_{item_id}.txt"
                            
                        else:
                            # Generic JSONL item support
                            text = item.get("text") or item.get("content") or str(item)
                            if not text:
                                continue
                            item_id = item.get("id") or f"{idx}"
                            content = text
                            doc_name = f"{source_type}_{item_id}.txt"
                            
                        self._process_document(source_type, doc_name, content)
                except Exception as e:
                    print(f"[ERROR] Failed to ingest JSONL file {file_path.name}: {e}")
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self._process_document(source_type, file_path.name, content)
                except Exception as e:
                    print(f"[ERROR] Failed to ingest text file {file_path.name}: {e}")

    def _ingest_transcripts_from_root(self, transcripts_dir: Path):
        """Ingest JSON transcript files from project root transcripts directory in user's exact format"""
        for json_file in transcripts_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                
                # Extract data from user's exact transcript format
                video_title = transcript_data.get("video_title", "Unknown Transcript")
                video_id = transcript_data.get("video_id", "")
                raw_text = transcript_data.get("raw_text", "")
                timestamp_start = transcript_data.get("timestamp_start", 0)
                timestamp_end = transcript_data.get("timestamp_end", 0)
                
                # Format into document
                full_text = f"# {video_title}\n\n"
                full_text += f"Video ID: {video_id}\n"
                full_text += f"Duration: {timestamp_start} to {timestamp_end} seconds\n\n"
                full_text += raw_text
                
                doc_name = f"transcript_{video_id or json_file.stem}.txt"
                self._process_document("transcript", doc_name, full_text)
                
            except Exception as e:
                print(f"[ERROR] Failed to ingest transcript {json_file.name}: {e}")

if __name__ == "__main__":
    RAGIngestor().ingest_all()
