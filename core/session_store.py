import sqlite3
import time
import os
from typing import Dict
from collections import OrderedDict
from core.orchestrator import KarpathyTwinOrchestrator

class SessionStore:
    def __init__(self, db_path: str = "storage/sessions.db", max_lru_size: int = 100):
        self.db_path = db_path
        self.max_lru_size = max_lru_size
        self._lru_cache: Dict[str, KarpathyTwinOrchestrator] = OrderedDict()
        
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL,
                    last_active REAL
                )
            """)

    def get_or_create(self, session_id: str) -> KarpathyTwinOrchestrator:
        self.touch(session_id)
        
        if session_id in self._lru_cache:
            self._lru_cache.move_to_end(session_id)
            return self._lru_cache[session_id]
            
        orchestrator = KarpathyTwinOrchestrator(session_id=session_id)
        self._lru_cache[session_id] = orchestrator
        self._lru_cache.move_to_end(session_id)
        
        if len(self._lru_cache) > self.max_lru_size:
            oldest_id, oldest_orch = self._lru_cache.popitem(last=False)
            try:
                oldest_orch.close()
            except Exception as e:
                pass
            
        return orchestrator

    def touch(self, session_id: str):
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, created_at, last_active)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_active = excluded.last_active
            """, (session_id, now, now))

    def evict_stale(self, max_age_seconds: int = 3600):
        cutoff = time.time() - max_age_seconds
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT session_id FROM sessions WHERE last_active < ?", (cutoff,))
            stale_ids = [row[0] for row in cursor.fetchall()]
            
        for sid in stale_ids:
            if sid in self._lru_cache:
                try:
                    self._lru_cache[sid].close()
                except Exception:
                    pass
                del self._lru_cache[sid]
                
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE last_active < ?", (cutoff,))

    def remove(self, session_id: str):
        if session_id in self._lru_cache:
            try:
                self._lru_cache[session_id].close()
            except Exception:
                pass
            del self._lru_cache[session_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
    def __contains__(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            return cursor.fetchone() is not None
