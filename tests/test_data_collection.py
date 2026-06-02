import os
import shutil
import pytest
from pathlib import Path
from data_collection.collect_twitter import TwitterCollector
from data_collection.collect_blog import BlogCollector
from data_collection.collect_papers import PaperCollector
from data_collection.collect_github import GitHubCollector

def test_collectors_mock_execution(test_storage_dir):
    # Output folders for collectors inside the temp directory
    tw_dir = Path(test_storage_dir) / "twitter"
    bg_dir = Path(test_storage_dir) / "blog"
    pp_dir = Path(test_storage_dir) / "papers"
    gh_dir = Path(test_storage_dir) / "github"
    
    # Run Twitter Collector
    tw = TwitterCollector(output_dir=str(tw_dir))
    # Mock config to use 1 mock tweet
    tw.config = {"mock_tweets": [{"id": "tw1", "text": "Testing tweets.", "created_at": "2023-01-01"}]}
    tw.collect()
    
    # Run Blog Collector
    bg = BlogCollector(output_dir=str(bg_dir))
    bg.config = {"mock_posts": [{"title": "Test Blog", "content": "Testing blogs.", "url": "https://test.com", "date": "2023-01-01"}]}
    # Call internal save directly or collect to trigger fallback
    # Force failure to trigger mock save
    bg.collect()
    
    # Run Papers Collector
    pp = PaperCollector(output_dir=str(pp_dir))
    pp.config = {"mock_papers": [{"title": "Test Paper", "arxiv_id": "9999.9999", "date": "2023-01-01", "abstract": "Abstract"}]}
    pp.collect()
    
    # Run GitHub Collector
    gh = GitHubCollector(output_dir=str(gh_dir))
    gh.config = {"repos": ["nanoGPT"], "owner": "karpathy"}
    # Trigger fallback or fetch
    gh.collect()
    
    # Assertions
    assert len(list(tw_dir.glob("*.txt"))) > 0 or tw_dir.joinpath("tweets.jsonl").exists()
    assert len(list(bg_dir.glob("*.txt"))) > 0
    assert len(list(pp_dir.glob("*.txt"))) > 0
    assert len(list(gh_dir.glob("*.txt"))) > 0
