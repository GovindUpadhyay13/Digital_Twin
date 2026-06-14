import sys
import os
import uuid
from core.orchestrator import KarpathyTwinOrchestrator

def main():
    print("=" * 70)
    print("====== DIGITAL TWIN OF ANDREJ KARPATHY — INTERACTIVE CLI ======")
    print("=" * 70)
    print("Welcome! Speak to Andrej's digital twin about neural networks, Tesla Autopilot,")
    print("OpenAI, nanoGPT, or first principles learning.")
    print("Commands:")
    print("  - Type 'exit' or 'quit' to save session and end.")
    print("  - Type 'sources' to toggle source visibility for responses.")
    print("=" * 70)
    
    # Generate unique thread ID for the session
    thread_id = f"cli_{uuid.uuid4().hex[:6]}"
    
    # Initialize Orchestrator
    try:
        orchestrator = KarpathyTwinOrchestrator(session_id=thread_id)
    except Exception as e:
        print(f"[ERROR] Failed to start orchestrator: {e}")
        return
        
    show_sources = True
    print("\nHey! Andrej here. What are we building today?\n")
    
    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("\nConsolidating session memory and saving logs...")
                orchestrator.close()
                print("Goodbye! Back to first principles.\n")
                break
                
            if user_input.lower() == "sources":
                show_sources = not show_sources
                print(f"Source citations: {'ON' if show_sources else 'OFF'}")
                continue
                
            # Run query through LangGraph pipeline
            result = orchestrator.chat(user_message=user_input, thread_id=thread_id)
            
            print(f"\nAndrej > {result['response']}")
            
            # Print sources if enabled and present
            if show_sources and result.get("sources"):
                print("\n[Retrieved Sources:]")
                for s in result["sources"]:
                    print(f"  • {s['title']} ({s['year']}) — {s['type'].upper()} [Relevance: {s['relevance']}]")
            print()
            
        except KeyboardInterrupt:
            print("\nExiting. Saving memory session...")
            orchestrator.close()
            break
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")

if __name__ == "__main__":
    main()
