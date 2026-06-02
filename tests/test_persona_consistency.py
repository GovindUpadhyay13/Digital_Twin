import pytest
from agent.persona_validator import PersonaValidator

def test_persona_validator_boilerplate():
    validator = PersonaValidator()
    
    # 1. Test clean responses
    clean_resp = "Hey! micrograd is literally a tiny autograd engine. Under the hood, it's very simple."
    res = validator.validate(clean_resp)
    assert res["is_valid"] is True
    assert res["severity"] == "none"
    
    # 2. Test major violation (AI self-identification)
    ai_resp = "As an AI, I represent Andrej Karpathy. I am a digital twin trained by Google."
    res2 = validator.validate(ai_resp)
    assert res2["is_valid"] is False
    assert res2["severity"] == "major"
    assert len(res2["issues"]) > 0
    
    # 3. Test minor violation (chatbot endings)
    minor_resp = "Here is the information you requested: backpropagation is the chain rule. How can I help you today?"
    res3 = validator.validate(minor_resp)
    assert res3["is_valid"] is False
    assert res3["severity"] == "minor"

def test_persona_auto_fix():
    validator = PersonaValidator()
    
    minor_resp = "Here is the information you requested: we build neural nets from scratch. Is there anything else you'd like to know?"
    fixed = validator.auto_fix_minor(minor_resp)
    
    # The cliches should be removed, leaving just the content
    assert "Here is the information" not in fixed
    assert "anything else you'd like to know" not in fixed
    assert "we build neural nets from scratch." in fixed
    
    # Validate the fixed text
    res = validator.validate(fixed)
    assert res["is_valid"] is True
    assert res["severity"] == "none"
