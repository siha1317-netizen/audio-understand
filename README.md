# audio-understand

**Give Claude (or any agent) ears.** A small Claude Code skill + three Python scripts that listen to an audio or video file and say, in plain words, what is in it:

- **What kind of sound, when** — speech / music / sound effects / silence per segment, and the top sound names (527 AudioSet classes: *Bird, Knock, Laughter, Glockenspiel, Rain...*).
- **How it feels** — mood ranking (*cute and playful, calm, warm, sad, tense, dreamy...*), plus tempo, brightness, density, loudness and a key guess.
- **Short events on a timeline** — impacts, laughter, paper tearing, heartbeat, rain... even under background music.
- **Pick a track** — rank several music candidates against the mood you want.

Runs locally on CPU, no API keys. English or Korean output (`--lang ko`).

> Everything here is a **model estimate**. It is very good at "is there speech?", "is it silent?", "what is the overall mood?" and loud events; it can miss short, quiet effects under music. Use it to narrow down, then listen.

## Install

```bash
git clone https://github.com/siha1317-netizen/audio-understand
cd audio-understand
# Windows
powershell -ExecutionPolicy Bypass -File install.ps1
# macOS / Linux
sh install.sh
```

Needs Python 3.10+. First run downloads the models (~1 GB) to the Hugging Face cache. ffmpeg is taken from PATH, or from the `imageio-ffmpeg` package, or from `AUDIO_UNDERSTAND_FFMPEG`.

**As a Claude Code skill:** put this folder in `~/.claude/skills/audio-understand/` and run the installer there. Claude will pick it up from `SKILL.md` when you ask things like "what does this track feel like?".

## Use

```bash
PY=.venv/bin/python            # Windows: .venv\Scripts\python.exe

# describe a file
$PY scripts/analyze.py song.mp3
$PY scripts/analyze.py clip.mp4 --seg 2 --lang ko

# find short events (1 s windows)
$PY scripts/events.py episode.wav --find "dog barking,keyboard typing"

# rank candidates for a mood
$PY scripts/analyze.py a.mp3 --out reports
$PY scripts/analyze.py b.mp3 --out reports
$PY scripts/compare.py reports --want "cute and playful"
```

Example (`analyze.py`, a playful CC0 tune):

```
[Mood — CLAP text match, sums to 1]
  funny and comical 0.80, cute and playful 0.08, cheerful and bright 0.03, ...
[Tempo & texture] ~122 BPM (fast) · brightness dark (729 Hz) · density busy (5.8 onsets/s) · ...
```

## How it works

| Step | Model |
|---|---|
| Sound classes per segment/window | [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) fine-tuned on AudioSet |
| Mood and named-sound search | [CLAP](https://huggingface.co/laion/clap-htsat-unfused) text–audio similarity |
| Tempo, brightness, onsets, key | librosa |

`events.py` slides a 1 s window (0.5 s hop). For AST it subtracts each class's median over the file, so only sounds that **stand out from the background** remain; classes active in more than 6 % of the file are treated as background, instrument-like classes are ignored while music plays, and windows quieter than −50 dBFS are skipped (both models hallucinate on silence). For CLAP it scores similarity to named sounds ("the sound of laughing") and keeps robust z-scores ≥ 3.

## Accuracy check

Tested on a 4.6-minute animated short's final mix with a known cue sheet (music box melody, ambiences, effects, dialogue, silences):

| | `analyze.py --seg 5` | `events.py` |
|---|---|---|
| Birds + wing flaps at the start | ✓ | ✓ |
| Music box melody / plucked strings | ✓ | (background) |
| Object dropping (0.5 s impact) | ✓ approx. | ✓ exact second |
| Paper tearing under music | ✗ | ✓ |
| Distant mocking laughter (pitched down) | ✗ | ✓ (~3 s late) |
| Heartbeat / rain | ✓ | ✓ |
| Fingernail scratch | ~ | ~ (3 s early) |
| Wing flap 0.45 s under music | ✗ | ✗ |
| Full silences | ✓ | (skipped) |

Also: a calm piano tune that a human listener called "too quiet, make it upbeat" was ranked *sad and melancholic* first by CLAP.

## Limits

- No transcription (it detects speech, not words). Pair with Whisper if you need words.
- Mood models are trained mostly on Western popular music.
- Key detection is a simple Krumhansl profile match — often wrong.
- Results are estimates. Listen before you decide.

## License

Scripts: MIT. Models and libraries keep their own licenses — see [NOTICE.md](NOTICE.md).

---

## 한국어 요약

Claude에게 "귀"를 달아 주는 스킬이에요. 소리나 영상 파일을 넣으면 이런 걸 말로 알려 줘요.
- 구간별 말소리, 음악, 효과음, 무음
- 들리는 소리 이름
- 분위기(귀여운, 잔잔한, 슬픈, 긴장되는 등), 빠르기, 밝기
- 짧은 효과음이 몇 초에 나오는지(웃음, 종이 찢기, 쿵 등)
- 원하는 느낌에 맞는 음악 후보 순위

컴퓨터 CPU로 돌아가고 API 키가 필요 없어요. 한국어로 보려면 `--lang ko`를 붙이세요. 결과는 모델이 추정한 값이니 최종 판단은 귀로 하세요.
