import pytest
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.manager import MemoryManager

def test_short_term_memory_sliding_window():
    stm = ShortTermMemory(window_size=3)
    
    # Add 3 turns
    stm.add_turn("user", "Hello")
    stm.add_turn("assistant", "Hi there")
    stm.add_turn("user", "I want to build a GPT")
    
    assert len(stm.get_recent_turns()) == 3
    assert len(stm.overflow_buffer) == 0
    
    # Add 4th turn -> oldest turn ("user", "Hello") should fall out and enter overflow
    stm.add_turn("assistant", "Let's do that!")
    
    assert len(stm.get_recent_turns()) == 3
    assert len(stm.overflow_buffer) == 1
    assert stm.overflow_buffer[0]["content"] == "Hello"
    assert stm.get_recent_turns()[0]["content"] == "Hi there"

def test_long_term_memory_io(test_storage_dir):
    ltm = LongTermMemory(persist_dir=test_storage_dir)
    ltm.clear_all()
    
    # Store user fact
    ltm.store_semantic("fact1", "User is building nanoGPT in PyTorch", "projects", 0.95)
    
    # Retrieve
    facts = ltm.retrieve_semantic("nanoGPT")
    assert len(facts) > 0
    assert "fact1" in ltm.fallback_path.read_text() if ltm.use_fallback else True
    assert "nanoGPT" in facts[0]["text"]
    
    # Store important moment
    ltm.store_important("mom1", "USER: I completed my PhD! ASSISTANT: Amazing!", "Completed academic milestone", "session123")
    moments = ltm.retrieve_important("PhD")
    assert len(moments) > 0
    assert "Academic" in moments[0]["reason"].lower() or "completed" in moments[0]["reason"].lower()

def test_memory_manager_coordination(mock_config):
    # Short term window size is 4 in mock_config
    mm = MemoryManager(session_id="test_sess", config_path=mock_config)
    
    mm.add_user_message("My name is John.")
    mm.add_assistant_message("Nice to meet you John!")
    mm.add_user_message("I am building micrograd.")
    mm.add_assistant_message("micrograd is delightful.")
    
    # 4 turns added, short-term turns = 4. Adding 5th will trigger overflow
    assert len(mm.short_term.get_recent_turns()) == 4
    assert len(mm.short_term.overflow_buffer) == 0
    
    mm.add_user_message("Can we talk about backpropagation?")
    assert len(mm.short_term.overflow_buffer) == 1
    assert mm.short_term.overflow_buffer[0]["content"] == "My name is John."
