import re
import json
import pandas as pd
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi

OUTPUT_DIR = Path("transcripts")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv("videos.csv")

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu.be/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract video id from {url}")

for _, row in df.iterrows():
    title = row["title"]
    url = row["url"]

    try:
        video_id = extract_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en"]
        )

        output = {
            "video_title": title,
            "video_id": video_id,
            "timestamp_start": transcript[0]["start"],
            "timestamp_end": transcript[-1]["start"] + transcript[-1]["duration"],
            "raw_text": " ".join([s["text"] for s in transcript])
        }

        safe_name = re.sub(
            r'[^a-zA-Z0-9_-]',
            '_',
            title
        )

        json_path = OUTPUT_DIR / f"{safe_name}.json"

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"Saved {title}")

    except Exception as e:
        print(f"Failed {title}: {e}")

print("Done")
