import os
from typing import Any
from agent.state import AgentState
from rag.retriever import Retriever
from memory.manager import MemoryManager
from agent.persona import get_system_prompt
from agent.persona_validator import PersonaValidator
from core.prompt_builder import PromptBuilder
from timeline.engine import TimelineEngine
from google import genai
from google.genai import types

# ─── NODE FUNCTIONS ─────────────────────────────────────────────

def input_node(state: AgentState) -> dict:
    """Receives and validates user input"""
    messages = state.get("messages") or []
    return {
        "messages": messages,
        "is_valid": True,
        "validation_issues": []
    }


def rag_node(state: AgentState, retriever: Retriever) -> dict:
    """Retrieve relevant context from RAG"""
    results = retriever.retrieve(
        state["user_message"],
        year_range=state.get("year_range")
    )
    
    return {
        "rag_results": results,
        "rag_context": retriever.format_context(results)
    }


def memory_node(
    state: AgentState,
    memory: MemoryManager
) -> dict:
    """Fetch short-term and long-term memory"""
    # Add current message to short-term sliding window
    memory.add_user_message(state["user_message"])
    
    # Get memory context
    memory_context = memory.get_formatted_memory_context(state["user_message"])
    
    # Get all user facts
    user_facts = [f["fact"] for f in memory.get_all_user_facts()]
    
    return {
        "memory_context": memory_context,
        "user_facts": user_facts
    }


def timeline_node(
    state: AgentState,
    timeline: TimelineEngine
) -> dict:
    """Detect timeline context and constraints"""
    timeline_context = timeline.get_timeline_context(state["user_message"])
    year_range = timeline.get_year_range_for_retrieval(state["user_message"])
    
    return {
        "timeline_context": timeline_context,
        "year_range": year_range
    }


def persona_prompt_node(
    state: AgentState,
    prompt_builder: PromptBuilder
) -> dict:
    """Assemble the master prompt with all context"""
    system_prompt = get_system_prompt()
    
    # Get recent conversation turns (we can use the messages list in the state)
    recent_turns = state["messages"][-4:] if state.get("messages") else []
    
    conversation_history = prompt_builder.build_prompt(
        system_prompt=system_prompt,
        user_query=state["user_message"],
        rag_context=state["rag_context"],
        memory_context=state["memory_context"],
        timeline_context=state["timeline_context"],
        conversation_summary=state.get("conversation_summary", ""),
        recent_turns=recent_turns,
    )
    
    return {
        "messages": conversation_history,
        # Keep track of system prompt in state for the LLM node
        "conversation_summary": system_prompt
    }


def llm_node(state: AgentState) -> dict:
    """Call Gemini 2.5 Flash to generate response"""
    api_key = os.environ.get("GEMINI_API_KEY")
    system_instruction = state.get("conversation_summary") or get_system_prompt()
    
    if not api_key:
        print("[LLM Node] Warning: GEMINI_API_KEY is not set in environment.")
        # Provide deterministic mock answers based on keywords in query for testing
        q = state["user_message"].lower()
        if "micrograd" in q:
            response_text = "Hey! Yeah, micrograd is essentially a tiny autograd engine. Under the hood, it's just about 100 lines of clean Python. It builds a dynamic DAG of scalar values where backprop runs in a single topological sort. Literally, you define a Value class, override double underscores like __add__ and __mul__, and keep track of children and local gradients. Beautifully simple."
        elif "tesla" in q or "autopilot" in q:
            response_text = "Hey there. When I was leading Autopilot at Tesla, we had to process pixels from 8 cameras directly into control vectors. It's essentially a massive HydroNet architecture where a shared backbone feeds multiple task heads. Over time, we transitioned from local heuristics to a complete vision-only system, using occupancy networks to represent 3D volumes. It was intense, but really exciting scaling it up."
        else:
            response_text = "Hey! Good to chat. Yeah, I love building neural networks from scratch. It's the only way to build real intuition. Whether it's training nanoGPT or writing C CUDA kernels in llm.c, you want to remove the frameworks and understand how the math works under the hood. What are we diving into?"
        return {"response": response_text}

    try:
        client = genai.Client(api_key=api_key)
        
        # Convert state messages to types.Content structure for the Gemini client
        contents = []
        for msg in state["messages"]:
            # LangChain messages standard is .content and .type
            # We map "human" -> "user" and "ai" -> "model"
            role = "user" if getattr(msg, "type", "human") == "human" else "model"
            txt = getattr(msg, "content", str(msg))
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=txt)]
            ))
            
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.95,
            )
        )
        
        response_text = response.text
    except Exception as e:
        print(f"[LLM Node] Error: {e}")
        response_text = "Hey. I ran into a technical hiccup talking to the Gemini backend. Essentially, check your GEMINI_API_KEY setup, and let's try that again. What's on your mind?"
    
    return {"response": response_text}


def memory_save_node(
    state: AgentState,
    memory: MemoryManager
) -> dict:
    """Save assistant response and update memory"""
    # Add response to short-term memory
    memory.add_assistant_message(state["response"])
    return {}


def validation_node(
    state: AgentState,
    validator: PersonaValidator
) -> dict:
    """Validate response for persona consistency"""
    response_text = state["response"]
    validation = validator.validate(response_text)
    
    # If minor issues, fix it automatically
    if validation["severity"] == "minor":
        response_text = validator.auto_fix_minor(response_text)
        # Re-evaluate
        validation = validator.validate(response_text)
        
    return {
        "response": response_text,
        "is_valid": validation["is_valid"],
        "validation_issues": validation["issues"]
    }


def output_node(state: AgentState) -> dict:
    """Format and return final response"""
    sources = [
        {
            "title": r["metadata"].get("source_title", "Unknown"),
            "year": r["metadata"].get("year", ""),
            "type": r["metadata"].get("source_type", ""),
            "relevance": round(r["score"], 3),
        }
        for r in state.get("rag_results", [])
    ]
    
    return {"sources": sources}
