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

### 3. Configure API key
```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...
```
The key is loaded at runtime via `python-dotenv`; it is never hardcoded.

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
| `--verbose` | Print each row to the console as it is processed | off |

Supported input formats: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`.

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
