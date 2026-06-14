import re

def format_for_tts(text: str) -> str:
    if not text:
        return text

    # Replace code blocks with spoken phrase
    text = re.sub(r'```.*?```', 'code block omitted for audio', text, flags=re.DOTALL)

    # Remove inline code marks
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Remove bold/italic (markdown bold is **, italic is *)
    # Match bold first so it doesn't leave partial asterisks
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove headers (e.g. # Header, ## Header)
    text = re.sub(r'(?m)^#+\s+(.*)$', r'\1', text)

    # Remove citation brackets like [1], [2]
    text = re.sub(r'\[\d+\]', '', text)

    # Expand common abbreviations
    replacements = {
        r'\bLLM\b': 'large language model',
        r'\bLLMs\b': 'large language models',
        r'\bRAG\b': 'retrieval augmented generation',
        r'\bGPU\b': 'G P U',
        r'\bGPUs\b': 'G P U s',
        r'\bCPU\b': 'C P U',
        r'\bCPUs\b': 'C P U s',
        r'\be\.g\.\b': 'for example',
        r'\bi\.e\.\b': 'that is',
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Collapse multiple spaces into one and strip leading/trailing
    text = re.sub(r'\s+', ' ', text).strip()
    return text
