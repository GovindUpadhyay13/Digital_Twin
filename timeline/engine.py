import re
import yaml
from pathlib import Path
from typing import Optional, List, Dict

class TimelineEngine:
    def __init__(self, phases_path: str = None):
        if not phases_path:
            phases_path = Path(__file__).parent / "phases.yaml"
            
        self.phases = []
        if Path(phases_path).exists():
            with open(phases_path, "r", encoding="utf-8") as f:
                self.phases = yaml.safe_load(f).get("phases", [])
        else:
            print("[WARN] phases.yaml not found. Using default timeline phases in memory.")
            self._set_default_phases()

    def _set_default_phases(self):
        self.phases = [
            {"id": "stanford_phd", "title": "Stanford PhD & CS231n Creator", "start_year": 2011, "end_year": 2016, "keywords": ["stanford", "phd", "cs231n"], "description": "Stanford PhD under Fei-Fei Li."},
            {"id": "tesla_autopilot", "title": "Tesla Director of AI", "start_year": 2017, "end_year": 2022, "keywords": ["tesla", "autopilot", "fsd"], "description": "Director of AI at Tesla leading Autopilot vision."}
        ]

    def get_matching_phases(self, query: str) -> List[Dict]:
        """Finds all phases matching keywords in the query"""
        query_lower = query.lower()
        matched = []
        for phase in self.phases:
            # Check if any keyword matches
            for kw in phase.get("keywords", []):
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', query_lower):
                    matched.append(phase)
                    break
        return matched

    def get_timeline_context(self, query: str) -> str:
        """Constructs timeline context based on the matching phases in the query"""
        matched = self.get_matching_phases(query)
        
        # If specific phases match, describe them. Otherwise list the whole timeline briefly.
        if matched:
            parts = ["[Timeline Context detected in your question]"]
            for phase in matched:
                parts.append(f"• Phase: {phase['title']} ({phase['start_year']}-{phase['end_year']})")
                parts.append(f"  Info: {phase['description']}")
            return "\n".join(parts)
            
        # Default brief timeline awareness dump
        parts = ["[General Timeline Awareness (Your Career Phases)]"]
        for phase in self.phases:
            parts.append(f"- {phase['start_year']}-{phase['end_year']}: {phase['title']}")
        return "\n".join(parts)

    def get_year_range_for_retrieval(self, query: str) -> Optional[tuple]:
        """
        Detects if user is asking about a specific year or phase and returns (start_year, end_year)
        For example:
        - "in 2018" -> (2018, 2018)
        - "when you were at Tesla" -> (2017, 2022)
        """
        # 1. Look for specific year in query
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            return (year, year)
            
        # 2. Check for matching phase keywords
        matched = self.get_matching_phases(query)
        if matched:
            # Return range covering all matched phases
            start = min(p["start_year"] for p in matched)
            end = max(p["end_year"] for p in matched)
            return (start, end)
            
        return None
