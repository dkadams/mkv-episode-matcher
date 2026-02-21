from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor
from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.misalignment import MisalignmentPolicy
from mkv_episode_matcher.series import Series


class SegmentTranscriber:
    def __init__(self, config: Configuration, series: Series,
        model_name: str, transcriber,
        misalignment_policy: Optional[MisalignmentPolicy] = None,
        variant_id: Optional[str] = None,
        output_dir: Optional[Path] = None):
        self.config = config
        self.series = series
        self.transcriber = self._init_transcriber(model_name, transcriber)
        self.misalignment_policy = misalignment_policy
        self.variant_id = variant_id or "aligned"
        self.output_dir = output_dir or series.ensure_transcription_text_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log_dir = self.output_dir.parent / "failure-logs"
        self.failure_log_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log_path = self.failure_log_dir / (
            f"transcription-failures-{os.getpid()}.jsonl"
        )

    @staticmethod
    def _init_transcriber(model_name, transcriber):
        if hasattr(transcriber, "transcribe") and not isinstance(transcriber, type):
            return transcriber
        if callable(transcriber):
            try:
                return transcriber(model_name)
            except TypeError:
                return transcriber()
        raise TypeError("transcriber must be a callable or object with a transcribe() method")

    @staticmethod
    def _normalize_transcript(transcript):
        if transcript is None:
            return None

        # TODO some of this is specific to the transcriber used and should be
        # pushed down.
        if isinstance(transcript, dict):
            text = transcript.get("text")
            if text:
                return str(text)
            segments = transcript.get("segments")
            if isinstance(segments, list):
                texts = [
                    str(segment.get("text", "")).strip()
                    for segment in segments
                    if isinstance(segment, dict) and segment.get("text")
                ]
                if texts:
                    return " ".join(filter(None, texts))
            files = transcript.get("files")
            if files:
                return SegmentTranscriber._extract_text(transcript)
            return None

        if isinstance(transcript, (list, tuple)):
            texts = [str(item).strip() for item in transcript if item]
            return " ".join(texts) if texts else None

        return (str(transcript)
                .strip()
                # whisper-cli inserts a marker when there's no audio for some
                # period of time.
                .replace("[BLANK_AUDIO]", " "))

    def execute(self, input: list[tuple[Path, list[int]]]) -> dict[Path, Path]:
        if hasattr(self.transcriber, "transcribe_many"):
            return self._execute_batch(input)
        return {path: self.transcribe(path, chunk_indexes)
                for path, chunk_indexes in input}

    def _execute_batch(self, inputs: list[tuple[Path, list[int]]]) -> dict[Path, Path]:
        duration = self.series.segment_duration
        outputs = {path: self.series.transcription_file(path) for path, _ in inputs}

        pending: list[tuple[Path, int, Path]] = []
        total_extract_time = 0.0
        with AudioChunkExtractor() as audio_extractor:
            for path, chunk_indexes in inputs:
                logger.info(f"Transcribing {path} chunks: {chunk_indexes}")
                for index in chunk_indexes:
                    base_offset = float(index * duration)
                    offset = base_offset
                    if self.misalignment_policy:
                        video_id = f"{path.resolve()}|{self.variant_id}"
                        offset += self.misalignment_policy.offset_for(video_id, index)
                    before = time.time()
                    try:
                        chunk_path = audio_extractor.extract(path, offset, duration)
                    except Exception as exc:  # noqa: BLE001
                        self._write_failure({
                            "failure_type": "audio_extract_exception",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "video_path": str(path),
                            "segment_index": index,
                            "base_offset_seconds": base_offset,
                            "effective_offset_seconds": max(0.0, offset),
                            "duration_seconds": duration,
                        })
                        logger.error(f"Audio extraction failed for {path} segment {index}: {exc}")
                        continue
                    total_extract_time += time.time() - before
                    pending.append((path, index, chunk_path))

            raw_results: list[Any] = []
            total_transcribe_time = 0.0
            if pending:
                before = time.time()
                try:
                    raw_results = self.transcriber.transcribe_many(
                        [chunk_path for _, _, chunk_path in pending]
                    )
                except Exception as exc:  # noqa: BLE001
                    for path, index, chunk_path in pending:
                        self._write_failure({
                            "failure_type": "batch_transcribe_exception",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "video_path": str(path),
                            "segment_index": index,
                            "chunk_path": str(chunk_path),
                            "duration_seconds": duration,
                        })
                    logger.error(f"Batch transcriber failed: {exc}")
                    raw_results = [None] * len(pending)
                total_transcribe_time += time.time() - before

            if pending and len(raw_results) != len(pending):
                logger.warning(
                    "Batch transcriber returned {} results for {} chunks",
                    len(raw_results),
                    len(pending),
                )
                if len(raw_results) < len(pending):
                    raw_results = raw_results + [None] * (len(pending) - len(raw_results))
                else:
                    raw_results = raw_results[:len(pending)]

            transcribed_by_path: dict[Path, dict[int, str]] = {path: {} for path, _ in inputs}
            for (path, index, chunk_path), raw_transcript in zip(pending, raw_results):
                text = self._normalize_transcript(raw_transcript)
                if text:
                    transcribed_by_path[path][index] = text
                else:
                    raw_preview = str(raw_transcript)
                    if len(raw_preview) > 300:
                        raw_preview = raw_preview[:300] + "...[truncated]"
                    self._write_failure({
                        "failure_type": "empty_transcript",
                        "video_path": str(path),
                        "segment_index": index,
                        "chunk_path": str(chunk_path),
                        "raw_transcript_type": type(raw_transcript).__name__,
                        "raw_transcript_preview": raw_preview,
                        "duration_seconds": duration,
                    })
                    logger.warning(f"Failed to transcribe {chunk_path}")

        logger.info(
            f"Extracted {len(pending)} audio chunks in {total_extract_time:.2f}s, "
            f"transcribed in {total_transcribe_time:.2f}s"
        )

        for path, output in outputs.items():
            transcribed = transcribed_by_path.get(path, {})
            self._write_transcript(output, transcribed)

        return outputs

    def transcribe(self, path: Path, chunk_indexes: list[int]) -> Path:
        logger.info(f"Transcribing {path} chunks: {chunk_indexes}")
        duration = self.series.segment_duration

        if self.output_dir == self.series.transcriptions_text_dir:
            output = self.series.transcription_file(path)
        else:
            output = self.output_dir / self.series.transcription_file_name(path)

        transcribed = {}
        total_extract_time = 0
        total_transcribe_time = 0
        failure_count = 0
        with AudioChunkExtractor() as audio_extractor:
            for index in chunk_indexes:
                base_offset = float(index * duration)
                offset = base_offset
                if self.misalignment_policy:
                    video_id = f"{path.resolve()}|{self.variant_id}"
                    offset += self.misalignment_policy.offset_for(video_id, index)

                before = time.time()
                try:
                    chunk_path = audio_extractor.extract(path, offset, duration)
                except Exception as exc:  # noqa: BLE001
                    failure_count += 1
                    self._write_failure({
                        "failure_type": "audio_extract_exception",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "video_path": str(path),
                        "segment_index": index,
                        "base_offset_seconds": base_offset,
                        "effective_offset_seconds": max(0.0, offset),
                        "duration_seconds": duration,
                    })
                    logger.error(f"Audio extraction failed for {path} segment {index}: {exc}")
                    continue
                total_extract_time += time.time() - before

                before = time.time()
                try:
                    raw_transcript = self.transcriber.transcribe(chunk_path)
                except Exception as exc:  # noqa: BLE001
                    failure_count += 1
                    self._write_failure({
                        "failure_type": "transcribe_exception",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "video_path": str(path),
                        "segment_index": index,
                        "chunk_path": str(chunk_path),
                        "base_offset_seconds": base_offset,
                        "effective_offset_seconds": max(0.0, offset),
                        "duration_seconds": duration,
                    })
                    logger.error(f"Transcription backend failed for {path} segment {index}: {exc}")
                    continue
                text = self._normalize_transcript(raw_transcript)
                total_transcribe_time += time.time() - before

                if text:
                    transcribed[index] = text
                else:
                    failure_count += 1
                    raw_preview = str(raw_transcript)
                    if len(raw_preview) > 300:
                        raw_preview = raw_preview[:300] + "...[truncated]"
                    self._write_failure({
                        "failure_type": "empty_transcript",
                        "video_path": str(path),
                        "segment_index": index,
                        "chunk_path": str(chunk_path),
                        "base_offset_seconds": base_offset,
                        "effective_offset_seconds": max(0.0, offset),
                        "duration_seconds": duration,
                        "raw_transcript_type": type(raw_transcript).__name__,
                        "raw_transcript_preview": raw_preview,
                    })
                    logger.warning(f"Failed to transcribe {chunk_path}")

        logger.info(f"Extracted {len(transcribed)} audio chunks "
                    f"in {total_extract_time:.2f}s, transcribed "
                    f"in {total_transcribe_time:.2f}s")
        if failure_count:
            logger.warning(
                f"Transcription had {failure_count} failures for {path}. "
                f"Failure log: {self.failure_log_path}"
            )

        self._write_transcript(output, transcribed)
        return output

    @staticmethod
    def _write_transcript(output: Path, transcribed: dict[int, str]) -> None:
        existing_transcript = {}
        if output.exists():
            with open(output) as existing:
                existing_transcript = json.load(existing)

        with open(output, "w") as json_out:
            full_transcript = existing_transcript | transcribed
            json.dump(full_transcript, json_out)

    @staticmethod
    def _extract_text(payload: dict) -> str | None:
        """
        Reduce the structured whisperkit-cli payload to raw text.

        Prefers segmented transcripts when available; falls back to the top-level
        plain text field.
        """
        files = payload.get("files") if isinstance(payload, dict) else None
        if not files:
            return None

        for file_entry in files:
            segments = file_entry.get("segments") if isinstance(file_entry, dict) else None
            if segments:
                texts = [
                    segment.get("text", "").strip()
                    for segment in segments
                    if isinstance(segment, dict) and segment.get("text")
                ]
                filtered = [text for text in texts if text]
                if filtered:
                    return " ".join(filtered)

            text = file_entry.get("text") if isinstance(file_entry, dict) else None
            if text:
                return str(text).strip()

        return None

    def _write_failure(self, payload: dict):
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "variant_id": self.variant_id,
            "misalignment_profile": (
                self.misalignment_policy.profile if self.misalignment_policy else "aligned"
            ),
            "misalignment_min_seconds": (
                self.misalignment_policy.min_seconds if self.misalignment_policy else None
            ),
            "misalignment_max_seconds": (
                self.misalignment_policy.max_seconds if self.misalignment_policy else None
            ),
            "misalignment_seed": (
                self.misalignment_policy.seed if self.misalignment_policy else None
            ),
        } | payload
        with self.failure_log_path.open("a", encoding="utf-8") as log_out:
            log_out.write(json.dumps(entry, ensure_ascii=False))
            log_out.write("\n")
