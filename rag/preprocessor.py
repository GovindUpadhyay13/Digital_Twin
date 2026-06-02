import re
from typing import Dict, List

class DocumentPreprocessor:
    @staticmethod
    def detect_primary_source(source_type: str) -> bool:
        """
        Determines if the document source represents Karpathy's own voice.
        All our 4 primary data sources (twitter, blog, papers, github) are primary.
        """
        primary_sources = {"twitter", "blog", "papers", "github"}
        return source_type.lower() in primary_sources

    @staticmethod
    def tag_topics(content: str) -> List[str]:
        """Automatically tags content with relevant machine learning topics"""
        topic_keywords = {
            "autograd": [r"autograd", r"gradient", r"derivative", r"micrograd"],
            "backpropagation": [r"backprop", r"chain rule", r"backward pass"],
            "transformers": [r"transformer", r"self-attention", r"attention", r"nanogpt", r"gpt"],
            "tesla": [r"tesla", r"autopilot", r"fsd", r"dojo", r"computer vision"],
            "openai": [r"openai", r"chatgpt", r"gpt-4", r"rlhf"],
            "tokenization": [r"tokenizer", r"sentencepiece", r"byte pair", r"bpe"],
            "optimization": [r"optimizer", r"adamw", r"sgd", r"learning rate"],
            "c_programming": [r"llm\.c", r"cuda", r"pure c", r"clang"]
        }
        
        content_lower = content.lower()
        topics = []
        for topic, patterns in topic_keywords.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    topics.append(topic)
                    break
        return topics

    @classmethod
    def enrich_metadata(cls, source_type: str, filename: str, content: str) -> Dict:
        """Extract metadata from file content, filename and enrich it"""
        metadata = {
            "source_type": source_type,
            "source_title": filename.replace("_", " ").replace(".txt", ""),
            "is_primary_source": cls.detect_primary_source(source_type),
            "speaker": "Andrej Karpathy" if cls.detect_primary_source(source_type) else "Unknown"
        }
        
        # Parse common header formats
        lines = content.split('\n')[:15]
        for line in lines:
            line = line.strip()
            if line.startswith("Date:"):
                # Extract year
                date_str = line.split(":", 1)[1].strip()
                metadata["date"] = date_str
                try:
                    # extract year as integer
                    year_match = re.search(r'\d{4}', date_str)
                    if year_match:
                        metadata["year"] = int(year_match.group())
                except:
                    pass
            elif line.startswith("URL:"):
                metadata["url"] = line.split(":", 1)[1].strip()
            elif line.startswith("Video ID:"):
                metadata["video_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("arXiv ID:"):
                metadata["arxiv_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("Repository:"):
                metadata["repository"] = line.split(":", 1)[1].strip()

        # Fallback year estimation if parsing fails
        if "year" not in metadata:
            # Look for a year in the filename or header
            year_match = re.search(r'(20\d{2})', filename + "\n" + "\n".join(lines))
            if year_match:
                metadata["year"] = int(year_match.group(1))
            else:
                metadata["year"] = 2023 # default fallback
                
        # Auto-tag topics
        topics = cls.tag_topics(content)
        metadata["topics"] = ",".join(topics) if topics else "general"
        
        return metadata
