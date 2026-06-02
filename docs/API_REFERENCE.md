# API Reference Guide

This document lists the public API interfaces for the core components of the Karpathy Digital Twin.

---

## 🚀 Core Orchestrator (`core/orchestrator.py`)

### `KarpathyTwinOrchestrator`
The main pipeline orchestrator compiling and running the LangGraph state machine.

* **`__init__(session_id: str = None, config_path: str = "config.yaml")`**
  Loads environment settings, instantiates retriever, dual-memory manager, timeline engine, and builds the state graph.
* **`chat(user_message: str, thread_id: str = "default_thread") -> Dict[str, Any]`**
  Submits the query to the state graph. Returns:
  ```python
  {
      "response": str,            # The final generated response
      "sources": List[Dict],      # Citations of matched chunks
      "is_valid": bool,           # Persona compliance state
      "validation_issues": List   # Validation warnings (if any)
  }
  ```
* **`close()`**
  Saves episodic session memory and triggers consolidations.

---

## 🔍 Context Retrieval (`rag/retriever.py`)

### `Retriever`
Handles semantic retrieval and source boosting from ChromaDB or JSON fallback databases.

* **`__init__(model_name: str, persist_dir: str, collection_name: str, top_k: int, boost_primary: float)`**
* **`retrieve(query: str, top_k: int = None, year_range: tuple = None) -> List[dict]`**
  Finds matching chunks. Applies a `1.3x` score multiplier if `is_primary_source` is True.
* **`format_context(results: List[dict]) -> str`**
  Formats retrieved document text and metadata headers into a stuffed prompt context block.

---

## 🧠 Memory Systems (`memory/`)

### `MemoryManager` (`memory/manager.py`)
Coordinates short-term active buffers and persistent sqlite/chromadb long-term storage.

* **`add_user_message(message: str)`**
* **`add_assistant_message(response: str)`**
  Saves turns in the sliding window. Triggers background memory consolidation if the window overflows.
* **`get_formatted_memory_context(query: str) -> str`**
  Queries long-term episodic, semantic, and important momentos matching the query context.
* **`get_all_user_facts() -> List[Dict]`**
  Returns all facts extracted about the user.

---

## ⏱️ Timeline Engine (`timeline/engine.py`)

### `TimelineEngine`
Determines career-phase contexts and extracts query-time temporal search filters.

* **`get_matching_phases(query: str) -> List[Dict]`**
  Matches keywords in the question against the phase database in `phases.yaml`.
* **`get_timeline_context(query: str) -> str`**
  Assembles descriptions of matched phases to inject into the system prompt.
* **`get_year_range_for_retrieval(query: str) -> Optional[tuple]`**
  Detects specific years (e.g. `2018`) or phase spans (e.g. `(2017, 2022)`) to filter the retriever.
