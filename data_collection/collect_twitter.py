import os
import json
import yaml
from pathlib import Path

class TwitterCollector:
    def __init__(self, output_dir="data/raw/twitter"):
        self.output_dir = output_dir
        base_dir = Path(__file__).parent
        config_path = base_dir / "config_sources.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f).get("twitter", {})
        else:
            self.config = {}

    def collect(self):
        """Try loading from HuggingFace dataset or fall back to config mock tweets"""
        os.makedirs(self.output_dir, exist_ok=True)
        dataset_name = self.config.get("huggingface_dataset", "dleemiller/karpathy_tweets")
        
        try:
            from datasets import load_dataset
            print(f"Attempting to download Twitter dataset '{dataset_name}' from HuggingFace...")
            dataset = load_dataset(dataset_name)
            tweets_data = dataset["train"]
            self._save_tweets(tweets_data)
            print(f"[OK] Twitter Collector finished using HuggingFace dataset.")
        except Exception as e:
            print(f"[WARN] HuggingFace Twitter dataset download failed: {e}. Using mock tweets from configuration.")
            mock_tweets = self.config.get("mock_tweets", [])
            self._save_tweets(mock_tweets)

    def _save_tweets(self, tweets_list):
        """Save as JSONL: one tweet per line, and also save a txt file representing each tweet for simpler RAG ingestion"""
        output_jsonl = f"{self.output_dir}/tweets.jsonl"
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for idx, tweet in enumerate(tweets_list):
                # Ensure fields are present
                clean_tweet = {
                    "id": tweet.get("id", f"mock_tweet_{idx}"),
                    "text": tweet.get("text", ""),
                    "created_at": tweet.get("created_at", "2023-01-01"),
                    "likes": tweet.get("likes", 100),
                    "retweets": tweet.get("retweets", 10),
                }
                json.dump(clean_tweet, f)
                f.write('\n')
                
                # Also save as a text file for simpler uniform ingestion in RAG
                txt_filename = f"{self.output_dir}/tweet_{clean_tweet['id']}.txt"
                with open(txt_filename, 'w', encoding='utf-8') as tf:
                    tf.write(f"# Tweet by @karpathy\n")
                    tf.write(f"Date: {clean_tweet['created_at']}\n")
                    tf.write(f"URL: https://twitter.com/karpathy/status/{clean_tweet['id']}\n")
                    tf.write(f"Likes: {clean_tweet['likes']}\n")
                    tf.write(f"Retweets: {clean_tweet['retweets']}\n\n")
                    tf.write(clean_tweet['text'])

if __name__ == "__main__":
    TwitterCollector().collect()
