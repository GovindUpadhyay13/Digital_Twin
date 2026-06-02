import os
import json
import uuid
import re
from typing import Dict, List
from google import genai
from google.genai import types
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

class MemoryConsolidator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    @property
    def client(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            print(f"[WARN] Failed to initialize GenAI client in consolidator: {e}")
            return None

    def consolidate_session(self, short_term: ShortTermMemory, long_term: LongTermMemory, session_id: str):
        """
        Compresses short term memory when it overflows.
        Extracts user facts and important moments, storing them in long-term memory.
        Updates short term memory's compressed summary.
        """
        if not short_term.needs_compression():
            return
            
        overflow_text = short_term.get_overflow_text()
        previous_summary = short_term.get_compressed_summary()
        
        # 1. Update session summary
        new_summary = self._summarize_conversation(previous_summary, overflow_text)
        short_term.set_compressed_summary(new_summary)
        
        # 2. Extract facts and store in long term semantic memory
        facts = self._extract_user_facts(overflow_text)
        for fact in facts:
            fact_id = str(uuid.uuid4())[:8]
            long_term.store_semantic(
                fact_id=f"fact_{fact_id}",
                fact=fact.get("fact", ""),
                category=fact.get("category", "general"),
                confidence=fact.get("confidence", 0.8)
            )
            print(f"[MEM] Saved semantic memory: {fact.get('fact')}")
            
        # 3. Extract important moments
        moments = self._extract_important_moments(overflow_text)
        for moment in moments:
            moment_id = str(uuid.uuid4())[:8]
            long_term.store_important(
                moment_id=f"moment_{moment_id}",
                exchange=moment.get("exchange", ""),
                importance_reason=moment.get("reason", ""),
                session_id=session_id
            )
            print(f"[MOMENT] Saved important moment: {moment.get('reason')}")

    def save_end_of_session(self, short_term: ShortTermMemory, long_term: LongTermMemory, session_id: str):
        """Saves final session summary as episodic memory"""
        recent_turns = short_term.get_recent_turns()
        if not recent_turns:
            return
            
        turns_text = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in recent_turns])
        previous_summary = short_term.get_compressed_summary()
        
        final_summary = self._summarize_conversation(previous_summary, turns_text)
        
        # Determine topics discussed in the session
        topics = self._extract_topics(final_summary)
        
        long_term.store_episodic(
            session_id=session_id,
            summary=final_summary,
            topics=topics,
            turn_count=short_term.turn_count
        )
        print(f"[OK] Saved episodic memory for session {session_id}")

    def _summarize_conversation(self, previous_summary: str, new_turns: str) -> str:
        """Call Gemini to summarize conversation history"""
        if not self.client:
            # Fallback simple concatenation if no API key
            return f"{previous_summary}\nUpdated with new turns: {new_turns[:100]}..."
            
        prompt = f"""You are a memory manager. Summarize the following dialogue.
        
Previous Summary:
{previous_summary}

New dialogue turns:
{new_turns}

Write a clean, concise updated summary in bullet points of what happened, focus on key technical topics and user details. Keep it short (under 150 words)."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"[WARN] Summarization failed: {e}")
            return f"{previous_summary}\n[Failed to summarize new turns]"

    def _extract_user_facts(self, text: str) -> List[Dict]:
        """Extract user facts as structured JSON list"""
        if not self.client:
            # Fallback regex parser for basic facts (e.g. "my name is X", "I use X")
            facts = []
            name_match = re.search(r"my name is (\w+)", text, re.IGNORECASE)
            if name_match:
                facts.append({"fact": f"User's name is {name_match.group(1)}", "category": "profile", "confidence": 0.9})
            return facts
            
        prompt = f"""Analyze the dialogue below. Extract facts about the user (preferences, experience level, projects they are working on, operating system, tools they use, or general background). 
        Format the output strictly as a JSON array of objects. Do not write markdown blocks other than JSON, just return a JSON list.
        Each object must have "fact" (string describing the fact), "category" (e.g., "projects", "profile", "preferences"), and "confidence" (float between 0.0 and 1.0).
        If no facts are present, return an empty JSON array [].
        
Dialogue:
{text}"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            # Strip any markdown blocks
            clean_json = response.text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
            return json.loads(clean_json)
        except Exception as e:
            print(f"[WARN] Fact extraction failed: {e}")
            return []

    def _extract_important_moments(self, text: str) -> List[Dict]:
        """Extract important moments as structured JSON list"""
        if not self.client:
            return []
            
        prompt = f"""Analyze the dialogue below. Identify if there are any highly important, memorable milestones or explicit requests to remember something (e.g., "remember this project phase", "I successfully built micrograd").
        Format the output strictly as a JSON array of objects. Do not write markdown blocks other than JSON, just return a JSON list.
        Each object must have "exchange" (string snippet showing the dialogue exchange) and "reason" (why this is an important moment to remember).
        If no important moments are present, return an empty JSON array [].
        
Dialogue:
{text}"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            clean_json = response.text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
            return json.loads(clean_json)
        except Exception as e:
            print(f"[WARN] Important moment extraction failed: {e}")
            return []

    def _extract_topics(self, text: str) -> List[str]:
        """Simple rule-based topic extraction from text summary"""
        keywords = ["gpt", "nanogpt", "micrograd", "autopilot", "backpropagation", "tokenization", "attention", "transformer", "c", "cuda", "teaching"]
        topics = []
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                topics.append(kw)
        if not topics:
            topics.append("general")
        return topics
