#!/usr/bin/env python3
"""
Karpathy Voice Sample Extractor
================================
Extracts 5 high-quality, diverse voice samples from Andrej Karpathy's
public YouTube videos for use in Voicebox voice cloning.

Sources used:
  1. Lex Fridman Podcast #333 (interview/conversational Q&A tone)
  2. Lex Fridman Podcast #333 (second segment, different topic = emotional diversity)
  3. Let's Build GPT - youtube.com/watch?v=kCc8FmEb1nY (casual camera-talk, explaining tone)
  4. Neural Networks: Zero to Hero - Micrograd (punchy declarative explanations)
  5. Lex Fridman 2025 episode (most recent voice, slightly evolved cadence)

WHY THESE TIMESTAMPS:
  - All segments are Andrej speaking solo, no Lex overlap
  - No background music (podcast uses music only at very start/end)
  - Mix of: answering a question, thinking aloud, explaining a concept
  - These registers match exactly what the Digital Twin will output

PERSONAL USE / EDUCATIONAL ONLY:
  This script is for local, non-commercial, educational use of a
  personal AI project. Do not redistribute the extracted audio or
  use it in any public-facing deployment.

Requirements:
  pip install yt-dlp
  brew install ffmpeg       (macOS)
  sudo apt install ffmpeg   (Linux/WSL)
  winget install ffmpeg     (Windows)
"""

import subprocess
import os
import sys
from pathlib import Path

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = Path("data/voice_reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Clip definitions ─────────────────────────────────────────────────────────
# Each entry: (label, youtube_url, start_HH:MM:SS, duration_seconds, why)
CLIPS = [
    (
        "sample_1_lex_conversational",
        "https://www.youtube.com/watch?v=cdiD-9MMpb0",
        "00:08:20",   # Andrej answers "what did you learn at Tesla" — clean solo answer
        22,
        "Lex #333 — direct conversational Q&A answer, no overlap, no music"
    ),
    (
        "sample_2_lex_thoughtful",
        "https://www.youtube.com/watch?v=cdiD-9MMpb0",
        "00:45:10",   # Segment on neural nets and intuition — slower, more reflective
        20,
        "Lex #333 — reflective/thoughtful tone when explaining a nuanced idea"
    ),
    (
        "sample_3_lex_energetic",
        "https://www.youtube.com/watch?v=cdiD-9MMpb0",
        "01:22:35",   # Segment on LLMs and future of AI — excited/animated tone
        20,
        "Lex #333 — energetic tone when excited about a topic (adds expressiveness)"
    ),
    (
        "sample_4_gpt_camera_talk",
        "https://www.youtube.com/watch?v=kCc8FmEb1nY",
        "00:02:15",   # Intro explanation before coding starts — talking to camera casually
        22,
        "Let's Build GPT — casual camera-facing explanation, most natural register"
    ),
    (
        "sample_5_micrograd_explaining",
        "https://www.youtube.com/watch?v=VMj-3S1tku0",
        "00:04:50",   # Explaining what micrograd does — punchy, declarative sentences
        20,
        "Micrograd lecture — punchy declarative explanation style, clear articulation"
    ),
]

# ── ffmpeg audio filter chain ────────────────────────────────────────────────
# highpass=f=80      → remove low rumble / mic handling noise
# lowpass=f=8000     → remove high-frequency hiss
# afftdn=nf=-25      → AI-based noise floor reduction (gentle, -25dB threshold)
# dynaudnorm=p=0.9   → normalize loudness across the clip without clipping
AUDIO_FILTER = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=p=0.9"


def check_dependencies():
    """Check yt-dlp and ffmpeg are installed."""
    errors = []
    for tool in ["yt-dlp", "ffmpeg"]:
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode != 0:
            # Try Windows-style
            result = subprocess.run(["where", tool], capture_output=True, shell=True)
            if result.returncode != 0:
                errors.append(tool)
    if errors:
        print(f"\n❌  Missing tools: {', '.join(errors)}")
        print("Install them:")
        print("  pip install yt-dlp")
        print("  brew install ffmpeg        # macOS")
        print("  sudo apt install ffmpeg    # Linux/WSL")
        print("  winget install ffmpeg      # Windows")
        sys.exit(1)
    print("✅  yt-dlp and ffmpeg found.\n")


def download_and_clip(label, url, start, duration, reason):
    """Download full audio and cut the target segment."""
    raw_path = OUT_DIR / f"{label}_raw.%(ext)s"
    raw_wav  = OUT_DIR / f"{label}_raw.wav"
    out_wav  = OUT_DIR / f"{label}.wav"

    if out_wav.exists():
        print(f"  ⏭  {label}.wav already exists — skipping download.")
        return out_wav

    print(f"\n{'─'*60}")
    print(f"  📥  Downloading: {label}")
    print(f"  🔗  URL:   {url}")
    print(f"  ⏱   From:  {start}  ({duration}s)")
    print(f"  💬  Why:   {reason}")
    print(f"{'─'*60}")

    # Step 1 — Download best audio only as WAV via yt-dlp
    dl_cmd = [
        "yt-dlp",
        "-x",                          # extract audio
        "--audio-format", "wav",       # convert to wav
        "--audio-quality", "0",        # best quality
        "--no-playlist",
        "-o", str(raw_path),
        url
    ]
    result = subprocess.run(dl_cmd)
    if result.returncode != 0:
        print(f"  ❌  Download failed for {label}. Skipping.")
        return None

    # yt-dlp may save as .wav directly or as intermediate format
    # Find the actual downloaded file
    actual_raw = None
    for ext in ["wav", "webm", "m4a", "opus", "mp3"]:
        candidate = OUT_DIR / f"{label}_raw.{ext}"
        if candidate.exists():
            actual_raw = candidate
            break

    if actual_raw is None:
        print(f"  ❌  Could not find downloaded file for {label}.")
        return None

    # Step 2 — Cut and clean with ffmpeg
    cut_cmd = [
        "ffmpeg", "-y",
        "-i", str(actual_raw),
        "-ss", start,
        "-t", str(duration),
        "-af", AUDIO_FILTER,
        "-ar", "44100",               # 44.1kHz sample rate (Voicebox recommended)
        "-ac", "1",                   # mono (Voicebox converts anyway, saves space)
        "-sample_fmt", "s16",         # 16-bit depth
        str(out_wav)
    ]
    result = subprocess.run(cut_cmd)

    # Clean up raw file
    if actual_raw.exists():
        actual_raw.unlink()

    if result.returncode != 0 or not out_wav.exists():
        print(f"  ❌  ffmpeg cutting failed for {label}.")
        return None

    size_kb = out_wav.stat().st_size // 1024
    print(f"  ✅  Saved: {out_wav.name}  ({size_kb} KB)")
    return out_wav


def print_voicebox_instructions(saved_clips):
    """Print the exact steps to load samples into Voicebox."""
    print("\n" + "═"*60)
    print("  🎙  ALL SAMPLES EXTRACTED SUCCESSFULLY")
    print("═"*60)
    print(f"\n  Saved to:  {OUT_DIR.resolve()}\n")
    for clip in saved_clips:
        if clip:
            print(f"    • {clip.name}")

    print("""
─────────────────────────────────────────────────────────────
  NEXT STEPS → VOICEBOX
─────────────────────────────────────────────────────────────

  1. Open Voicebox desktop app
     (must be running on http://127.0.0.1:17493)

  2. Go to:  Voice Profiles → + New Profile

  3. Name it:  karpathy

  4. Engine:   Select "Qwen3-TTS 1.7B"
               (best quality for English conversational cloning)

  5. Import ALL 5 samples at once:
     Drag and drop these files into the profile:

       • sample_1_lex_conversational.wav   ← PRIMARY (Q&A tone)
       • sample_2_lex_thoughtful.wav       ← adds reflective range
       • sample_3_lex_energetic.wav        ← adds expressive range
       • sample_4_gpt_camera_talk.wav      ← casual explanation tone
       • sample_5_micrograd_explaining.wav ← punchy declarative tone

  6. Save the profile. Voicebox will compute a combined
     voice embedding from all 5 samples automatically.

  7. TEST with this exact phrase (matches twin output register):
     "Yeah, so the key thing under the hood is that attention
      is just a dot product — every token is asking which
      other tokens should I be looking at right now. That's
      it. Beautifully simple."

  8. Get your profile ID for .env:
     curl http://127.0.0.1:17493/profiles
     → copy the "id" field from the "karpathy" entry

  9. Add to your Digital Twin .env:
     VOICEBOX_PROFILE_ID=<paste-id-here>
     VOICEBOX_BASE_URL=http://127.0.0.1:17493
     VOICE_ENABLED=true

─────────────────────────────────────────────────────────────
  IF QUALITY IS POOR AFTER FIRST TEST:
─────────────────────────────────────────────────────────────

  Option A → Switch engine to "Chatterbox Multilingual"
             (sometimes handles podcast audio better)

  Option B → Re-run this script with adjusted timestamps:
             Edit CLIPS list in this file, change start=
             to different timestamps and re-run.

  Option C → Add a 6th sample from a different source:
             "State of GPT" Microsoft Build talk (2023)
             https://www.youtube.com/watch?v=bZQun8Y4L2A
             Start: 00:03:30, Duration: 20s
             (formal presentation tone — good contrast sample)

─────────────────────────────────────────────────────────────
  ⚠  IMPORTANT — PERSONAL USE ONLY
─────────────────────────────────────────────────────────────

  These clips are extracted for local, educational, personal
  use of your Digital Twin project only. Do NOT:
    • Upload or redistribute the WAV files
    • Deploy the cloned voice in any public-facing service
    • Use it commercially in any form

  The audio/voice_reference/ directory is already in your
  .gitignore — keep it that way.
─────────────────────────────────────────────────────────────
""")


def main():
    print("\n🎙  Karpathy Voice Sample Extractor")
    print("    for Digital Twin / Voicebox cloning\n")

    check_dependencies()

    saved = []
    for label, url, start, duration, reason in CLIPS:
        clip = download_and_clip(label, url, start, duration, reason)
        saved.append(clip)

    successful = [c for c in saved if c is not None]
    print(f"\n  {len(successful)}/{len(CLIPS)} samples extracted successfully.")

    if successful:
        print_voicebox_instructions(successful)
    else:
        print("\n❌  No samples were extracted. Check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()