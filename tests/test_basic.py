
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.config_loader import load_config


def test_config_loader():
    """Test config loading (even if config doesn't exist)"""
    config = load_config()
    assert isinstance(config, dict)


def test_project_structure():
    """Test project structure exists"""
    assert project_root.exists()
    assert (project_root / "api").exists()
    assert (project_root / "agent").exists()
    assert (project_root / "core").exists()
    assert (project_root / "static").exists()
    assert (project_root / "requirements.txt").exists()


if __name__ == "__main__":
    test_config_loader()
    test_project_structure()
    print("✅ All basic tests passed!")
