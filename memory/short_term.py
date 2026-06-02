from collections import deque
from typing import List, Dict
from datetime import datetime

class ShortTermMemory:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.turns = deque(maxlen=window_size)
        self.overflow_buffer = []  # Older turns awaiting compression
        self.turn_count = 0
        self.compressed_summary = ""
    
    def add_turn(self, role: str, content: str):
        """Add a turn (user or assistant)"""
        # If the deque is full, we capture the item falling out and add to overflow buffer
        if len(self.turns) == self.window_size:
            discarded = self.turns.popleft()
            self.overflow_buffer.append(discarded)

        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.turn_count += 1
    
    def get_recent_turns(self) -> List[Dict]:
        """Return recent turns for prompt context"""
        return list(self.turns)
    
    def needs_compression(self) -> bool:
        """Check if overflow buffer needs compression (e.g. at least 4 turns in buffer)"""
        return len(self.overflow_buffer) >= 4
    
    def get_overflow_text(self) -> str:
        """Get text that needs compression"""
        return "\n".join([
            f"{t['role'].upper()}: {t['content']}"
            for t in self.overflow_buffer
        ])
    
    def set_compressed_summary(self, summary: str):
        """Store compressed summary and clear overflow buffer"""
        self.compressed_summary = summary
        self.overflow_buffer = []
    
    def get_compressed_summary(self) -> str:
        """Get compressed summary"""
        return self.compressed_summary
    
    def get_context_for_prompt(self) -> Dict:
        """Return formatted context for prompt"""
        return {
            "recent_turns": self.get_recent_turns(),
            "summary": self.compressed_summary,
            "total_turns": self.turn_count,
        }
    
    def clear(self):
        """Clear all turns"""
        self.turns.clear()
        self.overflow_buffer.clear()
        self.turn_count = 0
        self.compressed_summary = ""
