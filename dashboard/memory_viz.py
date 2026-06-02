import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from memory.manager import MemoryManager

def run_dashboard():
    print("=" * 70)
    print("        📊  KARPATHY TWIN — PERSISTENT MEMORY DASHBOARD  📊        ")
    print("=" * 70)
    
    # Initialize Memory Manager
    try:
        # Pass a dummy session just to read the database
        memory = MemoryManager(session_id="dashboard_viewer")
    except Exception as e:
        print(f"[ERROR] Failed to load memory database: {e}")
        return
        
    stats = memory.long_term.get_memory_stats()
    print("Memory Statistics:")
    print(f"  • User Facts (Semantic Collection):     {stats['semantic_count']}")
    print(f"  • Past Session Summaries (Episodic):    {stats['episodic_count']}")
    print(f"  • Memorable Milestones (Important):    {stats['important_count']}")
    print("-" * 70)
    
    # 1. Print Semantic Memory (User Facts)
    facts = memory.get_all_user_facts()
    print("RECORDED USER PROFILE & FACTS:")
    if not facts:
        print("  (No profile facts recorded yet)")
    else:
        for idx, f in enumerate(facts, 1):
            category = f["meta"].get("category", "general").upper()
            confidence = f["meta"].get("confidence", 0.8)
            print(f"  {idx}. [{category}] {f['fact']} (Confidence: {confidence})")
    print("-" * 70)
    
    # 2. Print Episodic Memory (Session Summaries)
    if stats["episodic_count"] > 0:
        # Retrieve all episodic records
        if memory.long_term.use_fallback:
            sessions = memory.long_term.data["episodic"]
            print("CONVERSATION SESSION ARCHIVES:")
            for idx, s in enumerate(sessions, 1):
                meta = s["metadata"]
                print(f"  Session {meta.get('session_id')} (Turns: {meta.get('turn_count')})")
                print(f"  Topics: {meta.get('topics')}")
                print(f"  Summary: {s['text']}")
                print()
        else:
            print("CONVERSATION SESSION ARCHIVES:")
            try:
                results = memory.long_term.episodic.get(include=["documents", "metadatas"])
                for i in range(len(results["ids"])):
                    meta = results["metadatas"][i]
                    print(f"  Session {results['ids'][i]} (Turns: {meta.get('turn_count')})")
                    print(f"  Topics: {meta.get('topics')}")
                    print(f"  Summary: {results['documents'][i]}")
                    print()
            except Exception as e:
                print(f"  Error reading episodic collection: {e}")
    else:
        print("CONVERSATION SESSION ARCHIVES:")
        print("  (No session summaries archived yet)")
    print("-" * 70)

    # 3. Print Important Moments
    if stats["important_count"] > 0:
        if memory.long_term.use_fallback:
            moments = memory.long_term.data["important"]
            print("MEMORABLE DIALOGUE INTERACTIONS:")
            for idx, m in enumerate(moments, 1):
                meta = m["metadata"]
                print(f"  Moment {idx}: {meta.get('importance_reason')}")
                print(f"  Exchange:\n  {m['text']}")
                print()
        else:
            print("MEMORABLE DIALOGUE INTERACTIONS:")
            try:
                results = memory.long_term.important.get(include=["documents", "metadatas"])
                for i in range(len(results["ids"])):
                    meta = results["metadatas"][i]
                    print(f"  Moment {i+1}: {meta.get('importance_reason')}")
                    print(f"  Exchange:\n  {results['documents'][i]}")
                    print()
            except Exception as e:
                print(f"  Error reading important moments: {e}")
    else:
        print("MEMORABLE DIALOGUE INTERACTIONS:")
        print("  (No important moments flagged yet)")
        
    print("=" * 70)

if __name__ == "__main__":
    run_dashboard()
