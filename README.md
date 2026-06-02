# 🧠 Digital Twin of Andrej Karpathy

An interactive AI application built with **LangGraph**, **Gemini 2.5 Flash**, and a **dual persistent memory model** that emulates Andrej Karpathy — his scholarly expertise, reasoning style, deep learning pedagogy, and approachable, builder-centric persona.

![FastAPI Framework](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-blue?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-AI-blueviolet?style=for-the-badge)

---

## 🌟 Core Features

- **High-Fidelity Persona Emulation**: Implements a dedicated prompting engine that captures Andrej's unique voice — pedagogical, humble, obsessed with first principles, and encouraging. The persona validator ensures robotic, boilerplate AI disclaimers are automatically stripped.
- **Multi-Source RAG Knowledge Base**: Turing-complete technical retrieval grounded directly in Andrej's real-world output. The database spans chunked transcripts from his *Neural Networks: Zero to Hero* YouTube lectures, active open-source repositories (such as `nanoGPT`, `micrograd`, `llm.c`), technical blog posts, arXiv papers, and Twitter insights.
- **Dual-Track Persistent Memory**:
  - *Short-term (Episodic)*: Manages immediate context within a sliding-window message queue. Older turns are automatically summarized into a dense episodic summary to preserve context while optimizing token usage.
  - *Long-term (Semantic)*: Runs an asynchronous background extraction thread to deduce facts about the user (e.g., "The user is studying CUDA," "The user is building micrograd from scratch") and persists them in ChromaDB.
- **Dynamic Memory Dashboard**: Exposes the internal state of the long-term memory system, logging semantic facts, past conversation histories, and key milestones to make the digital construct transparent.
- **Biographical Timeline Engine**: Detects time-anchored inputs (e.g., "What are you working on in 2018?") and constrains the RAG retriever to only pull historical data up to that year, adapting the agent's world knowledge to match historical milestones (e.g., Stanford, Tesla, OpenAI, Eureka Labs).

---

## 🏗️ System Architecture & Engineering Detail

The system is constructed as a structured pipeline of modular subsystems. Rather than sending raw text directly to an LLM, the orchestrator stages and contextualizes the query through several independent processors.

![System Architecture & Process Flow](SYSTEM_ARCHITECHTURE.png)

### 1. The Persona & Outbound Validator (`agent/`)
- **System Prompt**: Structures the identity, vocabulary (e.g., using builder terms like *"literally," "under the hood," "beautifully simple"*), and deep-seated core beliefs (e.g., *“you must build it from scratch to truly understand it”*).
- **Outbound Validator**: A post-processing layer (`PersonaValidator`) runs regex and keyword scans to detect common AI disclaimers (e.g., *"As an AI assistant," "I do not have feelings"*). It automatically strips or rewrites these lines to protect the illusion of the digital twin.

### 2. Dual-Track Memory Manager (`memory/`)
- **Episodic Sliding-Window**: Stores raw dialogue rounds in an active session state. If memory grows too large, a consolidation job compiles these rounds into an episodic summary.
- **Semantic Facts Vector Space**: Extracts user profile facts, preferences, and experience levels as JSON, stores them via the `all-MiniLM-L6-v2` embedding model, and persists them inside ChromaDB (`user_facts`). On subsequent queries, facts are matched and injected as part of the context layer.

### 3. RAG Pipeline & Multi-Source Scraper (`rag/`)
- **Data Collection System**: Includes automated scrapers in `data_collection/` built to pull raw materials across Youtube, Twitter, Github, and Blogs, storing the results in `data/raw`.
- **Primary Source Boosting**: Chunks are processed via `TextChunker` (512 tokens, 50-token overlap). Primary source documents (Andrej's actual code repositories and technical write-ups) receive a `1.3x` relevance multiplier boost to prioritize his authentic explanations over Wikipedia or third-party reports.

### 4. Biographical Timeline Engine (`timeline/`)
- Analyzes inputs for temporal references (e.g., *"When you were leading Autopilot at Tesla"* or *"In 2015"*).
- Adjusts the vector database query search filter dynamically to partition and restrict context boundaries to that particular chronological window.
- Handles edge cases—such as questions about modern tools (e.g., ChatGPT) during a "Tesla-era 2018 conversation"—by instructing the model to hypothesize from its current chronological vantage point.

### 5. API Key Rotation Module (`core/api_rotator.py`)
- Provides a secure, production-grade key management class that parses rotating API keys dynamically from environment variables.
- Supports multi-key strings (`GEMINI_API_KEYS="key1,key2,..."`) or individual indexes (`GEMINI_API_KEY_1` to `GEMINI_API_KEY_5`), rotating through them under lock-based synchronization to prevent backend rate-limiting.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API Key

### Installation

1. Clone the repository.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Copy `.env.example` to `.env` and insert your Gemini API Key(s):
   ```bash
   copy .env.example .env
   ```

### Ingestion & Data Preparation

Populate the local database before running the twin:

1. Run the data collectors:
   ```bash
   python data_collection/run_all.py --source all
   ```
2. Parse the dataset, generate embeddings, and build the local ChromaDB vector store:
   ```bash
   python scripts/ingest_documents.py
   ```

### Launching the Dashboard

Run the FastAPI backend which serves the interactive dark-mode dashboard:

```bash
uvicorn api.main:app --port 8000 --reload
```
Navigate to `http://localhost:8000` in your web browser.

---

## 🧪 Testing

Verify core components, RAG retrieval quality, and memory consolidation pipelines:

```bash
pytest
```

## 📜 License
MIT License.
