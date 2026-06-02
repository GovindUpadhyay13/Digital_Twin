import csv
import json
import os
from pathlib import Path

def parse_k(val):
    if not val:
        return 0
    val = val.strip().upper()
    if val.endswith('K'):
        try:
            return int(float(val[:-1]) * 1000)
        except:
            return 0
    try:
        return int(val)
    except:
        return 0

def main():
    csv_path = Path("data/scraped_tweets.csv")
    if not csv_path.exists():
        print(f"[ERROR] Could not find CSV file at {csv_path}")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("[ERROR] CSV file is empty.")
            return

        tweets = []
        for idx, row in enumerate(reader):
            if not row:
                continue
            if len(row) < 8:
                continue
            date_str = row[0]
            content = row[2]
            likes_str = row[5]
            retweets_str = row[6]
            replies_str = row[7]

            likes = parse_k(likes_str)
            retweets = parse_k(retweets_str)
            replies = parse_k(replies_str)

            tweet_id = f"tweet_{idx}"
            tweets.append({
                "id": tweet_id,
                "text": content,
                "created_at": date_str.split('T')[0] if 'T' in date_str else date_str,
                "likes": likes,
                "retweets": retweets,
                "replies": replies
            })

    output_dir = Path("data/raw/twitter")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    output_jsonl = output_dir / "tweets.jsonl"
    with open(output_jsonl, 'w', encoding='utf-8') as jf:
        for t in tweets:
            json.dump(t, jf)
            jf.write('\n')

    # Delete old tweet text files
    for old_file in output_dir.glob("tweet_*.txt"):
        try:
            old_file.unlink()
        except:
            pass

    # Save individual TXT files for RAG
    for t in tweets:
        txt_filename = output_dir / f"tweet_{t['id']}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as tf:
            tf.write(f"# Tweet by @karpathy\n")
            tf.write(f"Date: {t['created_at']}\n")
            tf.write(f"URL: https://twitter.com/karpathy/status/{t['id']}\n")
            tf.write(f"Likes: {t['likes']}\n")
            tf.write(f"Retweets: {t['retweets']}\n\n")
            tf.write(t['text'])

    print(f"[OK] Successfully imported {len(tweets)} tweets to data/raw/twitter/!")

if __name__ == "__main__":
    main()
