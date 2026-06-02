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
