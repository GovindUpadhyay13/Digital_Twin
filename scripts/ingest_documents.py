import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag.ingestion import RAGIngestor

def main():
    print("=" * 60)
    print("Karpathy Digital Twin — Document Ingestion & Vector Indexing")
    print("=" * 60)
    
    try:
        ingestor = RAGIngestor()
        ingestor.ingest_all()
        print("\nIngestion completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
