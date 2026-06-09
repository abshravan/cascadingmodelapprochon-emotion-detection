"""Speech Emotion Recognition pipeline using Google Gemini 2.0 Flash.

Analyzes patient audio clips for frustration, anxiety, and sadness, then
appends structured results to a CSV file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from tqdm import tqdm

SUPPORTED_EXTS: set[str] = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

CSV_COLUMNS: list[str] = [
    "timestamp",
    "filename",
    "chunk_index",
    "duration_seconds",
    "frustration_score",
    "anxiety_score",
    "sadness_score",
    "emotional_valence",
    "dominant_emotion",
    "confidence",
    "transcript_excerpt",
    "reasoning",
    "raw_response",
    "error",
]

SYSTEM_INSTRUCTION: str = """
You are a clinical audio analyst specializing in patient emotional states.
Analyze the provided audio clip and return ONLY a valid JSON object with
no additional text, markdown, or explanation.

The JSON must follow this exact schema:
{
  "frustration_score": <integer 0-10>,
  "anxiety_score": <integer 0-10>,
  "sadness_score": <integer 0-10>,
  "emotional_valence": <"positive" | "neutral" | "negative">,
  "dominant_emotion": <string, one word>,
  "confidence": <"low" | "medium" | "high">,
  "transcript_excerpt": <string, first 100 chars of speech or "unclear">,
  "reasoning": <string, 1-2 sentences explaining the scores>
}

Scoring guide:
- 0 = completely absent, 10 = extremely intense
- Base scores on vocal tone, pacing, pitch variation, and word choice
- If audio is unclear or silent, return all scores as -1 and confidence as "low"
""".strip()

USER_PROMPT: str = "Analyze the attached audio clip and return the JSON object."

GEMINI_MODEL: str = "gemini-2.5-flash"

LOCAL_EMOTION_MODEL: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
LOCAL_WHISPER_MODEL: str = "openai/whisper-small"

logger = logging.getLogger("emotion_pipeline")


class AudioPreprocessor:
    """Loads audio, normalizes to 16kHz mono WAV, trims silence, and chunks."""

    def __init__(self, target_sr: int = 16_000) -> None:
        self.target_sr = target_sr
        self._temp_files: list[Path] = []

    def load_and_convert(self, file_path: Path) -> AudioSegment:
        """Load any supported audio file and normalize to 16kHz mono."""
        audio = AudioSegment.from_file(str(file_path))
        audio = audio.set_channels(1).set_frame_rate(self.target_sr)
        return self._trim_silence(audio)

    @staticmethod
    def _trim_silence(audio: AudioSegment, silence_thresh_db: int = -40) -> AudioSegment:
        """Strip leading and trailing silence."""
        start = detect_leading_silence(audio, silence_threshold=silence_thresh_db)
        end = detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh_db)
        duration = len(audio)
        if start + end >= duration:
            return audio
        return audio[start: duration - end]

    @staticmethod
    def chunk_audio(
        audio: AudioSegment,
        chunk_ms: int = 60_000,
        overlap_ms: int = 2_000,
    ) -> list[AudioSegment]:
        """Split into overlapping chunks if the clip exceeds chunk_ms."""
        if len(audio) <= chunk_ms:
            return [audio]
        step = chunk_ms - overlap_ms
        chunks: list[AudioSegment] = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_ms, len(audio))
            chunks.append(audio[start:end])
            if end == len(audio):
                break
            start += step
        return chunks

    def export_temp(self, audio_segment: AudioSegment) -> Path:
        """Write an AudioSegment to a temp WAV file and track it for cleanup."""
        fd, raw_path = tempfile.mkstemp(suffix=".wav", prefix="emotion_chunk_")
        os.close(fd)
        path = Path(raw_path)
        audio_segment.export(str(path), format="wav")
        self._temp_files.append(path)
        return path

    def cleanup(self) -> None:
        """Remove all temp files created by this preprocessor."""
        for path in self._temp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove temp file %s: %s", path, exc)
        self._temp_files.clear()


class GeminiAnalyzer:
    """Wraps Gemini 2.0 Flash audio analysis with retry and JSON parsing."""

    _JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, api_key: str, model_name: str = GEMINI_MODEL) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def analyze(self, audio_path: Path) -> dict[str, Any]:
        """Upload and analyze a single audio clip. Returns parsed dict + raw text."""
        def _call() -> Any:
            uploaded = genai.upload_file(path=str(audio_path))
            try:
                return self._model.generate_content(
                    [USER_PROMPT, uploaded],
                    generation_config={"response_mime_type": "application/json"},
                )
            finally:
                try:
                    genai.delete_file(uploaded.name)
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    logger.debug("Could not delete uploaded file %s: %s", uploaded.name, exc)

        response = self._retry_with_backoff(_call)
        raw_text = getattr(response, "text", "") or ""
        parsed = self._parse_response(raw_text)
        parsed["raw_response"] = raw_text
        return parsed

    @classmethod
    def _parse_response(cls, raw: str) -> dict[str, Any]:
        """Extract a JSON object from the model's response."""
        text = raw.strip()
        if not text:
            raise ValueError("Empty response from Gemini")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = cls._JSON_BLOCK.search(text)
            if not match:
                raise ValueError(f"No JSON object found in response: {text[:200]}")
            return json.loads(match.group(0))

    @staticmethod
    def _retry_with_backoff(fn: Callable[[], Any], retries: int = 3) -> Any:
        """Run fn up to `retries` times with 2s, 4s, 8s backoff."""
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - surface to caller after retries
                last_exc = exc
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1, retries, exc, wait,
                )
                if attempt < retries - 1:
                    time.sleep(wait)
        assert last_exc is not None
        raise last_exc


class LocalAnalyzer:
    """Fully offline emotion analyzer using HuggingFace audio models.

    Uses a wav2vec2 8-class speech emotion classifier and (optionally) Whisper
    for transcript excerpts. Maps the classifier's softmax probabilities into
    the same JSON schema produced by GeminiAnalyzer so the CSV output is
    backend-agnostic.
    """

    def __init__(
        self,
        emotion_model_name: str = LOCAL_EMOTION_MODEL,
        whisper_model_name: str = LOCAL_WHISPER_MODEL,
        transcribe: bool = True,
    ) -> None:
        import numpy as np
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioClassification,
        )

        self._np = np
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading local emotion model: %s", emotion_model_name)
        self._extractor = AutoFeatureExtractor.from_pretrained(emotion_model_name)
        self._model = (
            AutoModelForAudioClassification.from_pretrained(emotion_model_name)
            .to(self._device)
            .eval()
        )
        self._id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}

        self._asr = None
        if transcribe:
            logger.info("Loading local ASR model: %s", whisper_model_name)
            from transformers import pipeline as hf_pipeline

            self._asr = hf_pipeline(
                "automatic-speech-recognition",
                model=whisper_model_name,
                device=0 if self._device == "cuda" else -1,
                chunk_length_s=30,
            )

    def analyze(self, audio_path: Path) -> dict[str, Any]:
        """Run emotion classification and (optional) transcription on one chunk."""
        audio = (
            AudioSegment.from_file(str(audio_path))
            .set_channels(1)
            .set_frame_rate(16_000)
        )
        samples = (
            self._np.array(audio.get_array_of_samples()).astype(self._np.float32) / 32_768.0
        )

        inputs = self._extractor(
            samples, sampling_rate=16_000, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = self._torch.softmax(logits, dim=-1).cpu().numpy()
        prob_map: dict[str, float] = {
            self._id2label[i]: float(probs[i]) for i in range(len(probs))
        }

        transcript = "unclear"
        if self._asr is not None:
            try:
                result = self._asr(samples.copy(), generate_kwargs={"language": "en"})
                text = (result.get("text") or "").strip()
                transcript = text[:100] if text else "unclear"
            except Exception as exc:  # noqa: BLE001 - transcript is best-effort
                logger.warning("Local ASR failed: %s", exc)

        return self._map_to_schema(prob_map, transcript)

    @staticmethod
    def _map_to_schema(probs: dict[str, float], transcript: str) -> dict[str, Any]:
        """Convert wav2vec2 class probabilities into the Gemini-shaped schema."""
        p = lambda key: probs.get(key, 0.0)  # noqa: E731

        frustration = round(10 * min(1.0, p("angry") + 0.5 * p("disgust")))
        anxiety = round(10 * min(1.0, p("fearful") + 0.3 * p("surprised")))
        sadness = round(10 * p("sad"))

        dominant_label, dominant_p = max(probs.items(), key=lambda kv: kv[1])
        negative = p("angry") + p("sad") + p("fearful") + p("disgust")
        positive = p("happy") + 0.5 * p("calm")
        if positive > 0.5:
            valence = "positive"
        elif negative > 0.5:
            valence = "negative"
        else:
            valence = "neutral"

        if dominant_p >= 0.6:
            confidence = "high"
        elif dominant_p >= 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        reasoning = (
            f"Local wav2vec2 classifier: dominant={dominant_label} ({dominant_p:.2f}). "
            f"Top probabilities: " + ", ".join(f"{k}={v:.2f}" for k, v in top3) + "."
        )

        return {
            "frustration_score": int(frustration),
            "anxiety_score": int(anxiety),
            "sadness_score": int(sadness),
            "emotional_valence": valence,
            "dominant_emotion": dominant_label,
            "confidence": confidence,
            "transcript_excerpt": transcript,
            "reasoning": reasoning,
            "raw_response": json.dumps(probs, sort_keys=True),
        }


class CSVLogger:
    """Append-only CSV writer with a fixed schema."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._rows_written = 0
        self._errors = 0
        if not self.output_path.exists():
            pd.DataFrame(columns=CSV_COLUMNS).to_csv(self.output_path, index=False)

    def log(self, row: dict[str, Any]) -> None:
        """Append a single row, filling missing columns with empty strings."""
        clean = {col: row.get(col, "") for col in CSV_COLUMNS}
        if clean.get("error"):
            self._errors += 1
        pd.DataFrame([clean], columns=CSV_COLUMNS).to_csv(
            self.output_path, mode="a", header=False, index=False
        )
        self._rows_written += 1

    def finalize(self) -> dict[str, int]:
        """Return summary statistics for the run."""
        return {
            "rows_written": self._rows_written,
            "errors": self._errors,
            "successes": self._rows_written - self._errors,
        }


def _iter_audio_files(input_path: Path) -> Iterable[Path]:
    """Yield supported audio files from a file or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTS:
            yield input_path
        return
    if input_path.is_dir():
        for path in sorted(input_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                yield path
        return
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_error_row(
    filename: str,
    chunk_index: int,
    duration_seconds: float,
    error_msg: str,
    raw_response: str = "",
) -> dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "filename": filename,
        "chunk_index": chunk_index,
        "duration_seconds": round(duration_seconds, 3),
        "frustration_score": -1,
        "anxiety_score": -1,
        "sadness_score": -1,
        "emotional_valence": "",
        "dominant_emotion": "",
        "confidence": "low",
        "transcript_excerpt": "",
        "reasoning": "",
        "raw_response": raw_response,
        "error": error_msg,
    }


def _build_success_row(
    filename: str,
    chunk_index: int,
    duration_seconds: float,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "filename": filename,
        "chunk_index": chunk_index,
        "duration_seconds": round(duration_seconds, 3),
        "frustration_score": parsed.get("frustration_score", -1),
        "anxiety_score": parsed.get("anxiety_score", -1),
        "sadness_score": parsed.get("sadness_score", -1),
        "emotional_valence": parsed.get("emotional_valence", ""),
        "dominant_emotion": parsed.get("dominant_emotion", ""),
        "confidence": parsed.get("confidence", ""),
        "transcript_excerpt": parsed.get("transcript_excerpt", ""),
        "reasoning": parsed.get("reasoning", ""),
        "raw_response": parsed.get("raw_response", ""),
        "error": "",
    }


def _build_analyzer(
    backend: str,
    *,
    model: str,
    api_key: str | None,
    transcribe: bool,
) -> GeminiAnalyzer | LocalAnalyzer:
    """Instantiate the analyzer for the requested backend."""
    if backend == "gemini":
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file or use --backend local."
            )
        return GeminiAnalyzer(api_key=api_key, model_name=model)
    if backend == "local":
        return LocalAnalyzer(transcribe=transcribe)
    raise ValueError(f"Unknown backend: {backend!r}")


def run_pipeline(
    input_path: Path,
    output_csv: Path,
    *,
    backend: str = "gemini",
    model: str = GEMINI_MODEL,
    transcribe: bool = True,
    verbose: bool = False,
    api_key: str | None = None,
) -> dict[str, int]:
    """End-to-end run: discover files, preprocess, analyze, log."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")

    files = list(_iter_audio_files(input_path))
    if not files:
        logger.warning("No supported audio files found at %s", input_path)
        return {"rows_written": 0, "errors": 0, "successes": 0}

    analyzer = _build_analyzer(
        backend, model=model, api_key=api_key, transcribe=transcribe
    )
    csv_logger = CSVLogger(output_csv)
    preprocessor = AudioPreprocessor()

    try:
        for file_path in tqdm(files, desc="Analyzing", unit="file"):
            try:
                audio = preprocessor.load_and_convert(file_path)
            except Exception as exc:  # noqa: BLE001 - log + continue batch
                logger.error("Failed to load %s: %s", file_path, exc)
                csv_logger.log(
                    _build_error_row(
                        filename=file_path.name,
                        chunk_index=0,
                        duration_seconds=0.0,
                        error_msg=f"load_error: {exc}",
                    )
                )
                continue

            chunks = preprocessor.chunk_audio(audio)
            for idx, chunk in enumerate(chunks):
                duration_s = len(chunk) / 1000.0
                temp_path: Path | None = None
                try:
                    temp_path = preprocessor.export_temp(chunk)
                    parsed = analyzer.analyze(temp_path)
                    row = _build_success_row(file_path.name, idx, duration_s, parsed)
                except (ValueError, json.JSONDecodeError) as exc:
                    raw = getattr(exc, "raw_response", "")
                    row = _build_error_row(
                        filename=file_path.name,
                        chunk_index=idx,
                        duration_seconds=duration_s,
                        error_msg=f"parse_error: {exc}",
                        raw_response=str(raw),
                    )
                except Exception as exc:  # noqa: BLE001 - already retried
                    row = _build_error_row(
                        filename=file_path.name,
                        chunk_index=idx,
                        duration_seconds=duration_s,
                        error_msg=f"api_error: {exc}",
                    )

                csv_logger.log(row)
                if verbose:
                    tqdm.write(
                        f"[{file_path.name} chunk {idx}] "
                        f"frustration={row['frustration_score']} "
                        f"anxiety={row['anxiety_score']} "
                        f"sadness={row['sadness_score']} "
                        f"emotion={row['dominant_emotion']} "
                        f"error={row['error'] or 'none'}"
                    )
    finally:
        preprocessor.cleanup()

    summary = csv_logger.finalize()
    print(
        f"Done. Rows written: {summary['rows_written']} "
        f"(successes: {summary['successes']}, errors: {summary['errors']}). "
        f"Output: {output_csv}"
    )
    return summary


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    file_handler = logging.FileHandler("errors.log")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition pipeline using Gemini 2.0 Flash."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to an audio file or a directory of audio files.",
    )
    parser.add_argument(
        "--output",
        default=Path("emotion_log.csv"),
        type=Path,
        help="Path to the output CSV (default: emotion_log.csv).",
    )
    parser.add_argument(
        "--backend",
        choices=("gemini", "local"),
        default="gemini",
        help="Inference backend: 'gemini' (cloud, needs API key) or 'local' "
        "(offline HuggingFace models). Default: gemini.",
    )
    parser.add_argument(
        "--model",
        default=GEMINI_MODEL,
        help=f"Gemini model name (ignored for --backend local). Default: {GEMINI_MODEL}.",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip Whisper transcription in local backend (faster, no transcript_excerpt).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each result to the console as it is processed.",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    _configure_logging(args.verbose)

    try:
        run_pipeline(
            args.input,
            args.output,
            backend=args.backend,
            model=args.model,
            transcribe=not args.no_transcribe,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level guard
        logger.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
