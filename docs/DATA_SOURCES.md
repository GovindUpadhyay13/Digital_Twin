# Data Sources & Collection Pipelines

This document details how raw data is collected, formatted, and ingested for the Digital Twin.

---

## 📥 Raw Data Collectors (`data_collection/`)

The system integrates five dedicated collectors, orchestrated by `run_all.py`. Each collector includes robust error handling and fallback mechanisms to ensure the project remains functional even under API rate limits or lack of credentials.

### 1. YouTube Collector (`collect_youtube.py`)
* **Target:** Andrej's "Zero to Hero" lecture series and public talks.
* **API:** `youtube-transcript-api`.
* **Output:** Saves full transcript text files to `data/raw/youtube/`.
* **Fallback:** Generates mock autograd/backpropagation lecture transcripts if the YouTube scraping API gets rate-limited.

### 2. Twitter Collector (`collect_twitter.py`)
* **Target:** Andrej's tweets.
* **Source:** HuggingFace community dataset `dleemiller/karpathy_tweets` (no auth needed).
* **Output:** Saves a structured `tweets.jsonl` file and text representations to `data/raw/twitter/`.
* **Fallback:** Loads pre-configured mock tweets regarding nanoGPT, Tesla Autopilot, and CS231n from `config_sources.yaml`.

### 3. Blog Collector (`collect_blog.py`)
* **Target:** GitHub Pages blog posts (`karpathy.github.io`) and Medium articles.
* **Mechanism:** `beautifulsoup4` HTML scraper.
* **Output:** Scrapes individual articles and outputs clean text files to `data/raw/blog/`.
* **Fallback:** Loads full-text mock articles of famous posts like "The Unreasonable Effectiveness of RNNs" and "A Recipe for Training Neural Networks".

### 4. Paper Collector (`collect_papers.py`)
* **Target:** arXiv research papers authored or co-authored by Andrej.
* **API:** `arxiv` search client.
* **Extractor:** `pdfminer.six` for clean PDF-to-text extraction.
* **Output:** Saves structured text drafts to `data/raw/papers/`.
* **Fallback:** Loads mock abstracts and method summaries from `config_sources.yaml`.

### 5. GitHub Collector (`collect_github.py`)
* **Target:** README files and key source docs from nanoGPT, micrograd, llm.c, convnetjs, etc.
* **API:** GitHub Repositories Content API.
* **Output:** Saves repo readmes to `data/raw/github/`.
* **Fallback:** Generates mock README files with explanation of self-attention, scalar autograd, and C/CUDA loops.

---

## 🏷️ Metadata Enrichment & Auto-Tagging

During ingestion, `rag/preprocessor.py` enriches each document chunk with metadata:
* **`source_type`:** `youtube`, `twitter`, `blog`, `paper`, or `github`.
* **`is_primary_source`:** Automatically set to `True` for Andrej's direct voice, boosting its similarity score by 1.3x during retrieval.
* **`year`:** Parsed from headers or filenames (e.g. `2015` for PhD era, `2019` for Tesla, `2024` for Eureka Labs) to enable career-phase filtering.
* **`topics`:** Extracted keywords (e.g. `autograd`, `backpropagation`, `transformers`, `tesla`, `c_programming`) for targeted searching.
