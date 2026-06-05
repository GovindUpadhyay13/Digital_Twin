
import sys
import os
import json
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_env, validate_env
from google import genai
from google.genai import types
from rag.embedder import Embedder
import chromadb

# Load environment
load_env()

# Initialize Gemini client
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Transcripts and processed data paths
TRANSCRIPTS_DIR = Path("transcripts")
PROCESSED_DIR = Path("processed_data")
PROCESSED_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB
CHROMA_PATH = Path("storage") / "chroma_db"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(str(CHROMA_PATH))

# Create collections
collections = {
    "knowledge": chroma_client.get_or_create_collection("karpathy_knowledge"),
    "analogies": chroma_client.get_or_create_collection("karpathy_analogies"),
    "reasoning_traces": chroma_client.get_or_create_collection("karpathy_reasoning"),
    "teaching_style": chroma_client.get_or_create_collection("karpathy_teaching"),
    "quotes": chroma_client.get_or_create_collection("karpathy_quotes"),
}

# Initialize embedder
embedder = Embedder()

def merge_transcript_segments(segments):
    """Merge transcript segments into continuous text"""
    text_parts = []
    for seg in segments:
        text_parts.append(seg.get("text", ""))
    return " ".join(text_parts)

def semantic_chunk(text, target_size=300, max_size=1000):
    """Perform simple semantic chunking (split by topics/question markers)"""
    # Split on common transition markers
    split_patterns = [
        r"\n\n(?:Okay|Now|So|Let's|Alright|But|Wait|Well|Let me|Let's see)\s",
        r"\n\n(?:Q:|Question:|What|Why|How|When|Where)\s",
        r"\n\n[0-9]+\.\s",
        r"\n\n\*\s",
    ]
    
    # Start with entire text
    chunks = []
    current_chunk = ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        current_chunk += sentence + " "
        
        # Check if chunk is within target size
        word_count = len(current_chunk.split())
        if word_count >= target_size and word_count <= max_size:
            # Check if we can split after this sentence
            chunks.append(current_chunk.strip())
            current_chunk = ""
        elif word_count > max_size:
            # Split at last period or question mark before max size
            split_idx = current_chunk.rfind('. ', 0, max_size*5)
            if split_idx == -1:
                split_idx = current_chunk.rfind('? ', 0, max_size*5)
            if split_idx != -1:
                chunks.append(current_chunk[:split_idx+1].strip())
                current_chunk = current_chunk[split_idx+1:].strip()
            else:
                # Force split
                chunks.append(current_chunk.strip())
                current_chunk = ""
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def extract_knowledge_from_chunk(chunk_text, video_title):
    """Use Gemini to extract structured knowledge from a text chunk"""
    if not client:
        print("Skipping extraction (no Gemini API key)")
        return None
        
    system_prompt = """
You are an expert knowledge extractor. Analyze the given transcript chunk from Andrej Karpathy's content and extract structured information.

Return ONLY valid JSON with the following structure:
{
  "topic": "the main topic of this chunk",
  "question": "explicit or implicit question this chunk answers (if any)",
  "answer": "summary of Karpathy's answer/explanation",
  "analogy": {
    "topic": "what the analogy is about",
    "analogy": "the actual analogy Karpathy uses",
    "explanation": "brief explanation of the analogy"
  },
  "key_insight": "the most important insight from this chunk",
  "reasoning_trace": {
    "question": "the driving question behind this reasoning",
    "reasoning_steps": ["list", "of", "reasoning", "steps"],
    "assumptions": ["list", "of", "assumptions"],
    "tradeoffs": ["list", "of", "tradeoffs"],
    "counterarguments": ["list", "of", "counterarguments"],
    "conclusion": "the final conclusion"
  },
  "reasoning_types": ["list", "of", "tags", "from", "these: First Principles, Analogy, Debugging, Tradeoff Analysis, Research Critique, System Design, Scaling Laws, Education, Engineering Judgment, Historical Context"],
  "teaching_style_notes": "notes on how Karpathy is teaching this topic",
  "quote": "a memorable quote from this chunk (if any)",
  "confidence": 0.0-1.0
}

If no analogy, reasoning trace, or quote exists, set those fields to null.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=chunk_text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=2048,
            )
        )
        
        # Extract JSON from response
        response_text = response.text.strip()
        # Try to find JSON object in response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end+1]
            extracted = json.loads(json_str)
            extracted["video_title"] = video_title
            return extracted
        else:
            print("No valid JSON in Gemini response")
            return None
            
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

def ingest_structured_knowledge(extracted_data, chunk_text, chunk_id):
    """Ingest extracted knowledge into separate Chroma collections"""
    if not extracted_data:
        return
        
    # 1. Knowledge collection
    if extracted_data.get("topic") and extracted_data.get("answer"):
        doc_text = f"Topic: {extracted_data['topic']}\nAnswer: {extracted_data['answer']}"
        embedding = embedder.embed_text(doc_text)
        collections["knowledge"].add(
            ids=[f"knowledge_{chunk_id}"],
            documents=[doc_text],
            metadatas=[{"video_title": extracted_data["video_title"], "topic": extracted_data["topic"]}],
            embeddings=[embedding.tolist()]
        )
        
    # 2. Analogies collection
    if extracted_data.get("analogy") and extracted_data["analogy"].get("analogy"):
        analogy = extracted_data["analogy"]
        doc_text = f"Topic: {analogy['topic']}\nAnalogy: {analogy['analogy']}\nExplanation: {analogy['explanation']}"
        embedding = embedder.embed_text(doc_text)
        collections["analogies"].add(
            ids=[f"analogy_{chunk_id}"],
            documents=[doc_text],
            metadatas=[{"video_title": extracted_data["video_title"], "topic": analogy["topic"]}],
            embeddings=[embedding.tolist()]
        )
        
    # 3. Reasoning traces collection
    if extracted_data.get("reasoning_trace") and extracted_data["reasoning_trace"].get("reasoning_steps"):
        rt = extracted_data["reasoning_trace"]
        doc_text = f"Question: {rt['question']}\nSteps: {'\n'.join(rt['reasoning_steps'])}"
        embedding = embedder.embed_text(doc_text)
        collections["reasoning_traces"].add(
            ids=[f"reasoning_{chunk_id}"],
            documents=[doc_text],
            metadatas=[{"video_title": extracted_data["video_title"], "question": rt["question"]}],
            embeddings=[embedding.tolist()]
        )
        
    # 4. Teaching style collection
    if extracted_data.get("teaching_style_notes"):
        topic = extracted_data.get("topic", "general")
        doc_text = f"Concept: {topic}\nTeaching Notes: {extracted_data['teaching_style_notes']}"
        embedding = embedder.embed_text(doc_text)
        collections["teaching_style"].add(
            ids=[f"teaching_{chunk_id}"],
            documents=[doc_text],
            metadatas=[{"video_title": extracted_data["video_title"], "concept": topic}],
            embeddings=[embedding.tolist()]
        )
        
    # 5. Quotes collection
    if extracted_data.get("quote"):
        quote = extracted_data["quote"]
        topic = extracted_data.get("topic", "general")
        embedding = embedder.embed_text(quote)
        collections["quotes"].add(
            ids=[f"quote_{chunk_id}"],
            documents=[quote],
            metadatas=[{"video_title": extracted_data["video_title"], "context": topic}],
            embeddings=[embedding.tolist()]
        )

def process_all_transcripts():
    """Process all transcript files in transcripts directory"""
    if not TRANSCRIPTS_DIR.exists():
        print("No transcripts directory found")
        return
        
    for json_file in TRANSCRIPTS_DIR.glob("*.json"):
        print(f"\nProcessing {json_file.name}...")
        
        with open(json_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
            
        video_title = transcript_data["title"]
        segments = transcript_data["segments"]
        
        # Merge segments and chunk
        full_text = merge_transcript_segments(segments)
        chunks = semantic_chunk(full_text)
        print(f"Created {len(chunks)} chunks for {video_title}")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{video_title.replace(' ', '_')}_{i}"
            
            # Extract knowledge using Gemini
            extracted = extract_knowledge_from_chunk(chunk, video_title)
            
            # Ingest into collections
            if extracted:
                ingest_structured_knowledge(extracted, chunk, chunk_id)
                print(f"Processed chunk {i+1}/{len(chunks)}")
            else:
                print(f"Skipped chunk {i+1}/{len(chunks)} (no extraction)")
                
        print(f"Completed {video_title}")
        
    print("\nAll processing complete!")

if __name__ == "__main__":
    process_all_transcripts()
