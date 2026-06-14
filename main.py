import sys
import os
import uuid
import asyncio
from core.orchestrator import KarpathyTwinOrchestrator

async def async_main():
    print("=" * 70)
    print("====== DIGITAL TWIN OF ANDREJ KARPATHY — INTERACTIVE CLI ======")
    print("=" * 70)
    print("Welcome! Speak to Andrej's digital twin about neural networks, Tesla Autopilot,")
    print("OpenAI, nanoGPT, or first principles learning.")
    print("Commands:")
    print("  - Type 'exit' or 'quit' to save session and end.")
    print("=" * 70)
    
    # Generate unique thread ID for the session
    thread_id = f"cli_{uuid.uuid4().hex[:6]}"
    
    # Preload the retriever on startup
    KarpathyTwinOrchestrator.initialize_global_retriever()
    
    # Initialize Orchestrator
    try:
        orchestrator = KarpathyTwinOrchestrator(session_id=thread_id)
    except Exception as e:
        print(f"[ERROR] Failed to start orchestrator: {e}")
        return
        
    print("\nHey! Andrej here. What are we building today?\n")
    
    while True:
        try:
            # Running synchronous input() in a thread to keep loop interactive if needed
            user_input = await asyncio.to_thread(input, "You > ")
            user_input = user_input.strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("\nConsolidating session memory and saving logs...")
                orchestrator.close()
                print("Goodbye! Back to first principles.\n")
                break
                
            print("\nAndrej > ", end="", flush=True)
            async for token in orchestrator.chat_stream(user_input):
                print(token, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\nExiting. Saving memory session...")
            orchestrator.close()
            break
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
