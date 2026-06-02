# Troubleshooting Guide

This document lists common issues, their root causes, and solutions for the Andrej Karpathy Digital Twin.

---

## 🛠️ Common Issues & Fixes

### 1. `ModuleNotFoundError: No module named 'chromadb'` or ChromaDB compiler issues on Windows
* **Reason:** ChromaDB requires building C++ binary extensions on Windows, which can fail if Microsoft Visual C++ Build Tools are missing.
* **Solution:** The codebase is built with an automatic **zero-dependency JSON fallback**. If ChromaDB cannot be imported, the system automatically redirects vector indexes and long-term memory to local JSON storage files in `storage/`. The agent remains fully operational without installing ChromaDB.

### 2. Large download sizes or slow initial startup of Embedding Model
* **Reason:** The `sentence-transformers` package downloads the embedding model on first import. `nomic-embed-text` is around 500MB, which can take time or timeout.
* **Solution:** We have configured `all-MiniLM-L6-v2` as the default model in `config.yaml`. It is very light (~90MB) and loads quickly. If the system is still unable to load the model due to lack of PyTorch or sentence-transformers, it automatically falls back to generating deterministic mock unit vectors based on string hashes, allowing offline tests to complete.

### 3. API key validation error on startup
* **Reason:** No `GEMINI_API_KEY` set in environment variables or `.env` file.
* **Solution:** 
  1. Copy `.env.example` to `.env` in the root folder.
  2. Insert your Google Gemini API Key: `GEMINI_API_KEY=AIzaSy...`
  3. Note: The orchestrator handles missing keys gracefully. If no key is present, it uses keyword matching and predefined mocks to reply directly in CLI chat mode, facilitating immediate debugging and evaluation.

### 4. `UnicodeEncodeError: 'charmap' codec can't encode character...`
* **Reason:** Windows cmd/PowerShell console using `CP1252` encoding by default. Emojis and checkmarks (`✓`) in print statements trigger encoding crashes.
* **Solution:** We have stripped all unicode symbols (checkmarks, emojis, crossmarks, stars) from print statements across all script files, replacing them with standard ASCII brackets (`[OK]`, `[ERROR]`, `[MEM]`). Make sure your python command execution is configured with `PYTHONIOENCODING=utf-8` if you add custom unicode prints.
