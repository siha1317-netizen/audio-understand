---
name: audio-understand
description: Lets Claude "hear" an audio or video file — splits it into speech / music / sound effects / silence, names the sounds (527 AudioSet classes), estimates mood (happy, sad, calm, cute, tense...), tempo, brightness, loudness and key, finds short sound events (impacts, laughter, paper tearing...) on a timeline, and ranks several music candidates against a wanted mood. Use when the user asks what a sound/music feels like, what is audible at a timestamp, whether a clip has speech, which background track fits a mood, or to check sound effects in an edit. Triggers include "listen to this", "what does this music feel like", "is this cute/sad", "which track is brighter", "find the laugh", "check the sound effects", "이 소리 들어 봐", "이 음악 어떤 느낌이야", "분위기 분석", "효과음 찾아 줘", "음악 후보 골라 줘".
---

# audio-understand

Claude cannot hear. This skill turns sound into numbers with three open models and reports what it probably sounds like. **Everything is a model estimate — say so, and tell the user to confirm by ear.**

| What | Model | License |
|---|---|---|
| 527 sound classes (speech, music, instruments, effects, mood labels) | AST `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause |
| Text ↔ audio match ("cute and playful", "the sound of laughing") | CLAP `laion/clap-htsat-unfused` | Apache-2.0 |
| Tempo, brightness, onset density, loudness, key | librosa | ISC |

## Setup (once)
Run `install.ps1` (Windows) or `install.sh` (macOS/Linux) in this folder. It creates `.venv`. Models (~1 GB) download on first run to the Hugging Face cache. ffmpeg is found on PATH, else via `imageio-ffmpeg`, else set `AUDIO_UNDERSTAND_FFMPEG`.

Python below means `<this folder>/.venv/Scripts/python.exe` on Windows or `<this folder>/.venv/bin/python` elsewhere.

## Commands
| Task | Command |
|---|---|
| Describe a file | `python scripts/analyze.py <file> [--seg 5] [--describe "cute ukulele"] [--out reports] [--lang ko\|en]` |
| Find short events | `python scripts/events.py <file> [--find "dog barking"] [--start 30 --end 60] [--lang ko\|en]` |
| Rank candidates | run `analyze.py --out reports` on each, then `python scripts/compare.py reports --want "cute and playful"` |

- Any audio or video format ffmpeg reads. Silent videos exit with "No audio track".
- `--lang ko` prints Korean. Use the user's language.
- CPU: ~30 s of audio takes tens of seconds for `analyze.py`; `events.py` takes ~1 s of compute per second of audio (cached on re-runs).

## How to report
1. One-line summary ("bright, playful ukulele track, ~120 BPM, no speech").
2. Segment table (time · kind · what is heard) or event list (time · labels).
3. Mood: CLAP ranking first (it compares well); AudioSet mood values are tiny (~0.00x) — use only to compare files.
4. Tempo / brightness / density / loudness / key. Tempo over 150 may be double-time — mention the half value.
5. End with: model estimates — confirm by ear.

## Known limits
- Does not transcribe words. It knows speech is present, not what is said.
- Short, quiet effects under music can be missed; use `events.py` (1 s windows) rather than `analyze.py --seg 5` for them.
- Mood models learned mostly from Western popular music; traditional or children's music may be off.
- Key detection is a simple estimate and often wrong.
- Licensed in-app music (e.g. Instagram's library) cannot be downloaded, so it cannot be analyzed.
