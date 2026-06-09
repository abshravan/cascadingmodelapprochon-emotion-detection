# Speech Emotion Recognition Pipeline

A production-ready pipeline that analyzes patient frustration, anxiety, and
sadness from audio clips using Google's **Gemini 2.0 Flash** model and logs
the structured output to a CSV file.

## Pipeline Architecture

```
Input → Preprocessing → Gemini Analysis → Response Parsing → CSV Logging
```

1. **Input layer** — accepts a single audio file or a folder of clips.
2. **Preprocessing** — validates format, converts to 16 kHz mono WAV, trims
   leading/trailing silence, and chunks long audio into 60 s segments with
   a 2 s overlap.
3. **Gemini analysis** — sends each chunk to `gemini-2.0-flash` with a
   structured clinical-analyst system prompt.
4. **Response parsing** — extracts the JSON object from the model response.
5. **CSV logging** — appends one row per chunk to `emotion_log.csv`.

## Setup

### 1. Prerequisites
- Python 3.10+
- `ffmpeg` available on `PATH` (required by `pydub` to decode mp3/m4a/ogg/flac).
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt-get install ffmpeg`
  - Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the **offline** backend (no API needed), also install:
```bash
pip install -r requirements-local.txt
```

### 3. Configure API key (Gemini backend only)
```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...
```
The key is loaded at runtime via `python-dotenv`; it is never hardcoded.
Skip this step if you only plan to use `--backend local`.

## Usage

### Single file
```bash
python pipeline.py --input ./samples/patient_001.wav
```

### Batch folder (recursive)
```bash
python pipeline.py --input ./samples --output results.csv --verbose
```

### CLI options
| Flag | Description | Default |
| --- | --- | --- |
| `--input` | Path to an audio file or folder | required |
| `--output` | Path to the output CSV | `emotion_log.csv` |
| `--backend` | `gemini` (cloud) or `local` (offline) | `gemini` |
| `--model` | Gemini model name (ignored for `local`) | `gemini-2.5-flash` |
| `--no-transcribe` | Skip Whisper transcription in local backend | off |
| `--rpm` | Max Gemini requests/minute (free-tier throttle) | `10` |
| `--max-requests` | Hard cap on Gemini calls this run | unlimited |
| `--no-resume` | Re-process chunks even if already logged | off |
| `--silence-dbfs` | dBFS threshold for skipping silent chunks | `-45.0` |
| `--verbose` | Print each row to the console as it is processed | off |

Supported input formats: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`.

### Switching backends
```bash
# Default Gemini model (gemini-2.5-flash)
python pipeline.py --input ./samples

# Pin to a different Gemini model (e.g. cheaper / more free quota)
python pipeline.py --input ./samples --model gemini-1.5-flash

# Fully offline — no API key, no network, no quota
python pipeline.py --input ./samples --backend local

# Offline + skip transcription for ~3x faster batches
python pipeline.py --input ./samples --backend local --no-transcribe
```

### Backend comparison
| | `--backend gemini` | `--backend local` |
| --- | --- | --- |
| Network required | yes | no |
| API key required | yes | no |
| Cost | free tier (rate-limited) | free, forever |
| Score quality | high (LLM reasoning over tone) | moderate (8-class softmax) |
| First-run latency | ~1 s upload | ~30 s model download (~1.5 GB) |
| Per-clip latency on CPU | 2–5 s | 1–3 s (emotion) + 5–15 s (transcript) |
| Transcript quality | high | depends on Whisper size |

The local backend uses
[`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
for emotion classification and `openai/whisper-small` for transcripts.
Its 8 classes (angry, calm, disgust, fearful, happy, neutral, sad, surprised)
are mapped into the same 0–10 frustration / anxiety / sadness scores so the
CSV schema is identical regardless of backend.

## CSV Output

`emotion_log.csv` is created automatically and appended to on subsequent
runs. Each row corresponds to one audio chunk.

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | ISO-8601 string | UTC time the row was written |
| `filename` | string | Source audio file name (no path) |
| `chunk_index` | int | 0-based chunk index within the file |
| `duration_seconds` | float | Chunk duration after silence trimming |
| `frustration_score` | int 0–10 (or -1) | Frustration intensity |
| `anxiety_score` | int 0–10 (or -1) | Anxiety intensity |
| `sadness_score` | int 0–10 (or -1) | Sadness intensity |
| `emotional_valence` | enum | `positive` / `neutral` / `negative` |
| `dominant_emotion` | string | Single dominant emotion word |
| `confidence` | enum | `low` / `medium` / `high` |
| `transcript_excerpt` | string | First ~100 chars of speech, or `unclear` |
| `reasoning` | string | 1–2 sentence justification for the scores |
| `raw_response` | string | Raw Gemini response text |
| `error` | string | Empty on success, otherwise the failure reason |

A score of `-1` means the model could not analyze the clip (silent, unclear,
or an API/parse error occurred).

## Free-Tier Tips

The pipeline ships with several optimizations specifically for the Gemini
free tier (roughly **10 RPM / 250 RPD** on `gemini-2.5-flash`, **15 RPM /
1,500 RPD** on `gemini-1.5-flash`):

- **Client-side throttling.** Requests are paced to `--rpm` so you don't
  trigger 429s in the first place. Default 10 RPM matches 2.5-flash.
- **Server-aware retry.** When a 429 does happen, the retry waits for the
  exact `retry_delay` Google returns rather than stacking blind exponential
  backoff on top of it.
- **Silent-chunk skipping.** Long-recording chunks under `--silence-dbfs`
  (default −45 dB) are scored locally as -1 / "silent" — no API call
  burned. Short user-submitted clips (< 10 s) are never auto-skipped.
- **Auto-resume.** Reruns read the existing CSV and skip chunks that
  already have a successful row. Crash, hit your daily quota, or Ctrl-C in
  the middle of a long batch → just rerun the same command and it picks
  up where it left off.
- **Daily budget cap.** `--max-requests N` stops the batch cleanly when
  you've used N calls, so you can split a large dataset across days.

Recommended free-tier invocation:
```bash
python pipeline.py \
  --input ./samples \
  --model gemini-2.5-flash \
  --rpm 10 \
  --max-requests 200 \
  --verbose
```
Tomorrow, run the exact same command — resume kicks in and only new
chunks are billed against your fresh daily quota.

## Error Handling

- **API failures** are retried up to 3 times with exponential backoff
  (2 s → 4 s → 8 s).
- **Parse failures** still write a row with `frustration/anxiety/sadness = -1`,
  the raw response in `raw_response`, and the failure reason in `error`.
- **File load failures** are logged to `errors.log` and skipped; the batch
  continues.
- All temporary WAV chunks created during preprocessing are deleted on exit.

## Known Limitations

- Gemini's audio understanding is probabilistic — scores can vary run to run.
- Long uploads (multi-minute clips) take noticeably longer; chunking helps
  but multiplies API calls.
- Only the standard REST SDK is used; the Live (WebSocket) API is **not**
  supported by design.
- Background noise, music, or multi-speaker audio can degrade scoring quality.
- The model returns English-centric reasoning; non-English transcript
  excerpts may still appear but the reasoning is in English.
- No PII redaction is performed on `transcript_excerpt` — handle the CSV as
  sensitive clinical data.
