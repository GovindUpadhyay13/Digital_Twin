import os
import json
import time
from typing import Optional, Dict

class FeedbackStore:
    def __init__(self, db_path: str = "storage/feedback.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_feedback(self, session_id: str, message_id_or_index: str, feedback_type: str, note: Optional[str] = None):
        entry = {
            "session_id": session_id,
            "message_id": message_id_or_index,
            "feedback_type": feedback_type,
            "note": note,
            "timestamp": time.time()
        }
        
        with open(self.db_path, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            
            data.append(entry)
            
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
