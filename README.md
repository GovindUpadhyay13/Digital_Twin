# Digital Twin of Andrej Karpathy

A sophisticated AI agent built using **LangGraph**, **Gemini 2.5 Flash**, and **dual persistent memory** that emulates Andrej Karpathy — his knowledge, reasoning style, teaching philosophy, and personality.

## 🚀 Key Features

- **Multi-Source Collector:** Gathers data across YouTube (Zero to Hero), Twitter, Blog posts, arXiv papers, and GitHub open-source repositories.
- **RAG Pipeline:** Utilizes `sentence-transformers` for technical semantic retrieval, with metadata filters and primary-source boosting (1.3x).
- **Dual Memory System:** Combines short-term sliding-window conversation memory with long-term episodic, semantic, and important moments memory.
- **Timeline Engine:** Detects the target period of Karpathy's life (e.g., Stanford PhD, Tesla, OpenAI, Eureka Labs) to align memory, context, and answers.
- **Persona Validator:** Monitors outputs to filter robotic AI boilerplate and maintain Andrej's unique, approachable tone.

## 📁 Project Structure

Refer to the implementation plan for the detailed directory layout.

## 🛠️ Setup & Installation

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment:
   Create a `.env` file from the template:
   ```bash
   copy .env.example .env
   ```
   Set your `GEMINI_API_KEY`.

3. Run data collection:
   ```bash
   python data_collection/run_all.py --source all
   ```

4. Build search indexes:
   ```bash
   python scripts/ingest_documents.py
   ```

5. Run CLI:
   ```bash
   python main.py
   ```
