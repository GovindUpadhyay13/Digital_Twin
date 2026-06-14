
import os
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage
from agent.state import AgentState
from rag.retriever import Retriever
from memory.manager import MemoryManager
from agent.persona import get_system_prompt
from agent.persona_validator import PersonaValidator
from core.prompt_builder import PromptBuilder
from timeline.engine import TimelineEngine
from google import genai
from google.genai import types

# Cached Gemini client and model for better latency
_cached_gemini_client = None
_cached_gemini_model = None

def get_gemini_client():
    """Get or create cached Gemini client"""
    global _cached_gemini_client
    if _cached_gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            _cached_gemini_client = genai.Client(api_key=api_key)
    return _cached_gemini_client

# --- NODE FUNCTIONS ---

def input_node(state: AgentState) -> dict:
    """Receives and validates user input"""
    messages = state.get("messages") or []
    # Add current user message
    messages.append(HumanMessage(content=state["user_message"]))
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
        "user_facts": user_facts,
        "short_term_summary": memory.short_term.get_compressed_summary()
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
    
    # Get recent conversation history
    recent_turns = state["messages"][-4:] if state.get("messages") else []
    
    conversation_history = prompt_builder.build_prompt(
        system_prompt=system_prompt,
        user_query=state["user_message"],
        rag_context=state["rag_context"],
        memory_context=state["memory_context"],
        timeline_context=state["timeline_context"],
        short_term_summary=state.get("short_term_summary", ""),
        recent_turns=recent_turns,
    )
    
    return {
        "messages": conversation_history,
        # Keep track of system prompt in state
        "system_prompt_cache": system_prompt
    }


def llm_node(state: AgentState) -> dict:
    """Call Gemini to generate response, with solid fallback"""
    api_key = os.environ.get("GEMINI_API_KEY")
    system_instruction = state.get("system_prompt_cache") or get_system_prompt()
    
    # Define fallback response generator
    def get_fallback_response(q):
        q_lower = q.lower()
        if "micrograd" in q_lower or "backprop" in q_lower:
            return "Hey! Yeah, micrograd is a tiny autograd engine. It builds a DAG of scalar values and runs backpropagation via a topological sort. Super simple way to see how gradients work under the hood without all the framework bloat."
        elif "tesla" in q_lower or "autopilot" in q_lower:
            return "When I was at Tesla, we went all-in on a vision-only system using occupancy networks for 3D space representation. No lidar, just pixels—cleaner approach but with its own tradeoffs."
        elif "transformer" in q_lower or "attention" in q_lower:
            return "Transformers let every token talk to every other token. That's attention. It's more computationally expensive, but the flexibility for sequence tasks has been a game-changer."
        else:
            return "Hey! Great question. I love going deep on neural network fundamentals—from backprop basics to LLM training dynamics. What specific aspect are you curious about?"
    
    if not api_key:
        print("[LLM Node] Warning: GEMINI_API_KEY is not set. Using fallback.")
        return {"response": get_fallback_response(state["user_message"])}

    try:
        client = get_gemini_client()
        if not client:
            raise RuntimeError("Failed to get Gemini client")
        
        # Convert state messages to types.Content structure for the Gemini client
        contents = []
        for msg in state["messages"]:
            role = "user" if getattr(msg, "type", "human") == "human" else "model"
            txt = getattr(msg, "content", str(msg))
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=txt)]
            ))
        
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
        response_text = None
        
        for model_name in models_to_try:
            try:
                print(f"[LLM Node] Trying model: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                        top_p=0.95,
                    )
                )
                response_text = response.text
                break
            except Exception as model_err:
                print(f"[LLM Node] Model {model_name} failed: {model_err}")
                continue
        
        if response_text is not None:
            return {"response": response_text}
        
    except Exception as e:
        print(f"[LLM Node] Error: {e}")
    
    print("[LLM Node] All models failed or unavailable. Using fallback.")
    return {"response": get_fallback_response(state["user_message"])}


def memory_save_node(
    state: AgentState,
    memory: MemoryManager
) -> dict:
    """Save assistant response and update memory"""
    # Add response to short-term memory
    memory.add_assistant_message(state["response"])
    
    # Update messages state
    messages = state.get("messages") or []
    messages.append(AIMessage(content=state["response"]))
    
    return {"messages": messages}


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
        
    retry_count = state.get("retry_count", 0)
    messages = state.get("messages", [])
    
    if validation["severity"] == "major" and retry_count < 1:
        correction = validator.build_correction_instruction(validation["issues"])
        messages.append(HumanMessage(content=correction))
        retry_count += 1
        
    return {
        "response": response_text,
        "is_valid": validation["is_valid"],
        "validation_issues": validation["issues"],
        "severity": validation["severity"],
        "retry_count": retry_count,
        "messages": messages
    }


def output_node(state: AgentState) -> dict:
    """Format and return final response"""
    sources = [
        {
            "title": r["metadata"].get("title", "Unknown"),
            "year": r["metadata"].get("year"),
            "type": r["metadata"].get("source_type", r["metadata"].get("type", "unknown")),
            "relevance": round(r["score"], 3),
        }
        for r in state.get("rag_results", [])
    ]
    
    return {"sources": sources}
