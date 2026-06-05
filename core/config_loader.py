import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration settings from YAML file"""
    path = Path(config_path)
    if not path.exists():
        # Look relative to project root
        project_root = Path(__file__).parent.parent
        path = project_root / config_path
        
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    print(f"[WARN] Config file '{config_path}' not found. Returning empty default settings.")
    return {}

def validate_env() -> None:
    """Validates required environment variables are set"""
    required_vars = ["GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment variables.")
        print("Creating .env file template...")
        
        template_path = Path(__file__).parent.parent / ".env.example"
        with open(template_path, "w", encoding="utf-8") as f:
            f.write("# Digital Twin - Environment Variables\n")
            f.write("# Copy this file to .env and fill in your values\n\n")
            f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
            f.write("# Optional: Add other API keys if needed\n")
            f.write("# YOUTUBE_API_KEY=your_youtube_api_key\n")
            f.write("# TWITTER_API_KEY=your_twitter_api_key\n")
        print(f"Created .env.example at {template_path.resolve()}")
        print("Please create .env file with your actual API keys.")

def load_env(env_path: str = ".env"):
    """
    Parses a local .env file and sets values in os.environ.
    This provides a zero-dependency dotenv alternative.
    """
    path = Path(env_path)
    if not path.exists():
        # Look relative to project root
        project_root = Path(__file__).parent.parent
        path = project_root / env_path
        
    if path.exists():
        print(f"Loading environment variables from {path.resolve()}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    # Strip quotes if present
                    val = val.strip().strip('"').strip("'")
                    os.environ[key.strip()] = val
    else:
        # Fallback print warning
        if not os.environ.get("GEMINI_API_KEY"):
            print("[WARN] .env file not found and GEMINI_API_KEY environment variable is not set.")
    
    # Validate environment variables after loading
    validate_env()
