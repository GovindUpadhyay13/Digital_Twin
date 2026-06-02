import re
from typing import Dict, List

class PersonaValidator:
    def __init__(self):
        # AI clichés or out-of-character patterns
        self.major_cliches = [
            r"as an ai\b",
            r"i am an ai\b",
            r"i am a language model",
            r"my training data",
            r"representing andrej",
            r"simulation of andrej",
            r"as a digital twin",
            r"representing karpathy"
        ]
        
        self.minor_cliches = [
            r"how can i help you today\?",
            r"here is the information you requested",
            r"sure! i can help you with that",
            r"is there anything else you'd like to know\?"
        ]

    def validate(self, text: str) -> Dict:
        """Validate response for persona consistency"""
        text_lower = text.lower()
        issues = []
        severity = "none"
        
        # Check major violations
        for pattern in self.major_cliches:
            if re.search(pattern, text_lower):
                issues.append(f"Major Violation: Response contains AI self-identification ('{pattern}')")
                severity = "major"
                
        # Check minor violations (only if no major ones are found)
        if severity != "major":
            for pattern in self.minor_cliches:
                if re.search(pattern, text_lower):
                    issues.append(f"Minor Violation: Response contains standard chatbot boilerplate ('{pattern}')")
                    severity = "minor"
                    
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "severity": severity
        }

    def auto_fix_minor(self, text: str) -> str:
        """Fix minor violations such as greetings or endings"""
        fixed_text = text
        
        # Remove common minor cliches
        fixed_text = re.sub(r"(?i)sure,? i can help you with that\.?\s*", "", fixed_text)
        fixed_text = re.sub(r"(?i)here is the information you requested:?\s*", "", fixed_text)
        fixed_text = re.sub(r"(?i)is there anything else you'd like to know\??\s*", "", fixed_text)
        fixed_text = re.sub(r"(?i)how can i help you today\??\s*", "", fixed_text)
        
        # Strip leading/trailing whitespaces that might remain
        fixed_text = fixed_text.strip()
        
        return fixed_text
