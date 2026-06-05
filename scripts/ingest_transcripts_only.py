
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.ingestion import RAGIngestor

def ingest_transcripts_only():
    """Only ingest transcripts from the transcripts directory"""
    ingestor = RAGIngestor()
    transcripts_dir = project_root / "transcripts"
    if transcripts_dir.exists():
        ingestor._ingest_transcripts_from_root(transcripts_dir)
    else:
        print("No transcripts directory found!")

if __name__ == "__main__":
    ingest_transcripts_only()
