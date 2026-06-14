import requests

def judge_response(question: str, response: str, expected_topics: list[str]) -> int:
    prompt = f"""You are evaluating an AI persona's response for topical relevance.
Question: {question}
Response: {response}
Expected topics covered: {', '.join(expected_topics)}

On a scale of 1-5, how well does the response address the expected topics?
Respond with ONLY a single digit 1-5."""
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        for char in text:
            if char.isdigit():
                score = int(char)
                if 1 <= score <= 5:
                    return score
        return 3
    except Exception as e:
        print(f"[WARN] Judge failed: {e}. Ensure Ollama is running.")
        return None
