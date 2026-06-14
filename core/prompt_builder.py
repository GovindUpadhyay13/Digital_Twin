from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class PromptBuilder:
    def build_prompt(
        self,
        system_prompt: str,
        user_query: str,
        rag_context: str,
        memory_context: str,
        timeline_context: str,
        short_term_summary: str,
        recent_turns: List[BaseMessage]
    ) -> List[BaseMessage]:
        """
        Assembles context (RAG, Memory, Timeline) and conversation history into a list of messages.
        
        Args:
            system_prompt: The system persona instruction.
            user_query: The current raw query from the user.
            rag_context: Formatted context blocks from ChromaDB.
            memory_context: User facts and episodic context from memory.
            timeline_context: Career milestones relating to the query.
            short_term_summary: High level dialogue compression context (if any).
            recent_turns: Last few messages in the conversation thread.
            
        Returns:
            List of BaseMessage objects.
        """
        messages = []
        
        # 1. Inject the system prompt
        messages.append(SystemMessage(content=system_prompt))
        
        # 2. Inject context as a synthetic HumanMessage -> AIMessage pair
        context_blocks = []
        if rag_context:
            context_blocks.append(f"[RETRIEVED KNOWLEDGE FROM MY WORKS]\n{rag_context}")
        if timeline_context:
            context_blocks.append(f"[CAREER TIMELINE INFORMATION]\n{timeline_context}")
        if memory_context:
            context_blocks.append(f"[CONTEXT FROM PREVIOUS CONVERSATIONS]\n{memory_context}")
        if short_term_summary:
            context_blocks.append(f"[RECENT SESSION SUMMARY]\n{short_term_summary}")
            
        if context_blocks:
            messages.append(HumanMessage(content="[CONTEXT RETRIEVAL]"))
            messages.append(AIMessage(content="\n\n".join(context_blocks)))
            
        # 3. Add recent turns (conversation history)
        if recent_turns:
            # Filter out any duplicate current messages to prevent loop
            messages.extend(recent_turns)
            
        # 4. Add the final user query
        messages.append(HumanMessage(content=user_query))
        
        return messages
