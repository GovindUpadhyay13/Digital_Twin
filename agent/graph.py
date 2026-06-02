from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.nodes import (
    input_node, rag_node, memory_node, timeline_node,
    persona_prompt_node, llm_node, memory_save_node,
    validation_node, output_node
)

def build_graph(retriever, memory, timeline, prompt_builder, validator):
    """
    Builds the LangGraph StateGraph.
    
    Flow:
    input → [rag, memory, timeline] → persona_prompt → llm →
    memory_save → validation → output → END
    """
    
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("input", input_node)
    graph.add_node("rag", lambda state: rag_node(state, retriever))
    graph.add_node("memory", lambda state: memory_node(state, memory))
    graph.add_node("timeline", lambda state: timeline_node(state, timeline))
    graph.add_node("persona_prompt", lambda state: persona_prompt_node(state, prompt_builder))
    graph.add_node("llm", llm_node)
    graph.add_node("memory_save", lambda state: memory_save_node(state, memory))
    graph.add_node("validation", lambda state: validation_node(state, validator))
    graph.add_node("output", output_node)
    
    # Set entry point
    graph.set_entry_point("input")
    
    # Add edges: Input forks into rag, memory, and timeline in parallel
    graph.add_edge("input", "rag")
    graph.add_edge("input", "memory")
    graph.add_edge("input", "timeline")
    
    # RAG, memory, and timeline merge into persona_prompt
    graph.add_edge("rag", "persona_prompt")
    graph.add_edge("memory", "persona_prompt")
    graph.add_edge("timeline", "persona_prompt")
    
    # Sequential flow to end
    graph.add_edge("persona_prompt", "llm")
    graph.add_edge("llm", "memory_save")
    graph.add_edge("memory_save", "validation")
    graph.add_edge("validation", "output")
    graph.add_edge("output", END)
    
    # Compile with memory
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
