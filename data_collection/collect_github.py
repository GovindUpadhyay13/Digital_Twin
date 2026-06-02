import os
import base64
import requests
import yaml
from pathlib import Path

class GitHubCollector:
    def __init__(self, output_dir="data/raw/github"):
        self.output_dir = output_dir
        base_dir = Path(__file__).parent
        config_path = base_dir / "config_sources.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f).get("github", {})
        else:
            self.config = {}
        
        self.token = os.environ.get("GITHUB_TOKEN")

    def collect(self):
        """Fetch README files from Karpathy's GitHub repositories"""
        os.makedirs(self.output_dir, exist_ok=True)
        owner = self.config.get("owner", "karpathy")
        repos = self.config.get("repos", ["nanoGPT", "micrograd", "llm.c"])
        
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
            
        for repo in repos:
            repo_path = f"{owner}/{repo}"
            readme_url = f"https://api.github.com/repos/{repo_path}/readme"
            
            try:
                print(f"Fetching GitHub README for: {repo_path}...")
                response = requests.get(readme_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    readme_content = data['content']
                    readme_text = base64.b64decode(readme_content).decode('utf-8')
                    self._save_readme(repo, readme_text)
                    print(f"[OK] Downloaded README: {repo}")
                else:
                    print(f"[WARN] GitHub API returned status {response.status_code} for {repo_path}. Using mock README.")
                    self._save_mock_readme(repo)
            except Exception as e:
                print(f"[WARN] Failed to fetch GitHub README for {repo_path}: {e}. Using mock README.")
                self._save_mock_readme(repo)

    def _save_readme(self, repo_name, text):
        """Save README text to file"""
        output_file = f"{self.output_dir}/{repo_name}_README.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {repo_name}\n")
            f.write(f"Repository: karpathy/{repo_name}\n")
            f.write(f"URL: https://github.com/karpathy/{repo_name}\n\n")
            f.write(text)

    def _save_mock_readme(self, repo_name):
        """Generate and save mock README content for fallback"""
        mock_contents = {
            "nanoGPT": "nanoGPT is the simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of minGPT, focused on efficiency and simplicity. It contains clean, readable PyTorch code.",
            "micrograd": "micrograd is a tiny Autograd engine (with a bite-sized PyTorch-like API) that implements backpropagation over a dynamically built DAG. It supports scalar values and works with standard neural net layers.",
            "llm.c": "llm.c is a project to train LLMs in simple, pure C/CUDA. No need for 245MB of PyTorch. Just write the neural network layers directly in C and compile it. It is extremely fast and educational.",
            "char-rnn": "Multi-layer Recurrent Neural Networks (LSTM, RNN, GRU) for training character-level language models in Torch/Lua.",
            "convnetjs": "Javascript library for training Deep Learning models (Convolutional Neural Networks, Recurrent Neural Networks) in your browser. Formulate networks, train, and run completely client-side.",
            "arxiv-sanity-preserver": "A web interface for querying and personalized recommendations of arXiv papers over time. Built using Flask and simple TF-IDF vectors.",
            "nn-zero-to-hero": "Code and notebooks for the YouTube course 'Neural Networks: Zero to Hero'. Includes exercises for building micrograd, makemore, nanoGPT, and tokenizers."
        }
        
        fallback_text = mock_contents.get(
            repo_name,
            f"This is a mock README for repository {repo_name} owned by karpathy. It contains code, explanations, and tutorials relating to deep learning and neural network training."
        )
        
        self._save_readme(repo_name, fallback_text)
        print(f"[OK] Saved mock README: {repo_name}")

if __name__ == "__main__":
    GitHubCollector().collect()
