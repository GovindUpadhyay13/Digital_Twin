from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class PromptBuilder:
    def build_prompt(
        self,
        system_prompt: str,
        user_query: str,
        rag_context: str,
        memory_context: str,
        timeline_context: str,
        conversation_summary: str,
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
            conversation_summary: High level dialogue compression context (if any).
            recent_turns: Last few messages in the conversation thread.
            
        Returns:
            List of BaseMessage objects.
        """
        messages = []
        
        # Add recent turns (conversation history)
        if recent_turns:
            # Filter out any duplicate current messages to prevent loop
            messages.extend(recent_turns)
            
        # Build the current user turn with layered context injection
        # Gemini excels at handling this stuffed context within the prompt structure.
        enriched_user_content = f"""[RETRIEVED KNOWLEDGE FROM MY WORKS]
{rag_context}

[CAREER TIMELINE INFORMATION]
{timeline_context}

[CONTEXT FROM PREVIOUS CONVERSATIONS]
{memory_context}

[CONVERSATION COMPRESSION SUMMARY]
{conversation_summary if conversation_summary else "No previous compression."}

[MY CURRENT QUESTION]
{user_query}"""

        messages.append(HumanMessage(content=enriched_user_content))
        return messages
