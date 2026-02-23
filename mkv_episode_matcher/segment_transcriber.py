from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor
from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.misalignment import MisalignmentPolicy
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcription_pipeline import (
    FilePipelineStats,
    SegmentSpec,
    TranscriptionPipeline,
)


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
        return self._execute_pipeline(input)

    def _execute_pipeline(self, inputs: list[tuple[Path, list[int]]]) -> dict[Path, Path]:
        duration = self.series.segment_duration
        outputs = {path: self._transcription_output(path) for path, _ in inputs}
        specs_by_path: dict[Path, list[SegmentSpec]] = {}

        for path, chunk_indexes in inputs:
            specs_by_path[path] = [
                self._build_spec(path, index, duration)
                for index in chunk_indexes
            ]

        with AudioChunkExtractor() as audio_extractor:
            pipeline = TranscriptionPipeline(
                audio_extractor=audio_extractor,
                transcriber=self.transcriber,
                normalize_transcript=self._normalize_transcript,
                write_failure=self._write_failure,
                io_workers=self._io_workers(),
                transcribe_workers=self._transcribe_workers(),
            )
            transcribed_by_path, per_path_stats = pipeline.run(specs_by_path)

        total_extract_time = sum(stats.extract_seconds for stats in per_path_stats.values())
        total_transcribe_time = sum(stats.transcribe_seconds for stats in per_path_stats.values())
        total_transcribed = sum(stats.segments_transcribed for stats in per_path_stats.values())
        total_failed = sum(stats.segments_failed for stats in per_path_stats.values())
        logger.info(
            f"Extracted {total_transcribed} audio chunks in {total_extract_time:.2f}s, "
            f"transcribed in {total_transcribe_time:.2f}s"
        )
        if total_failed:
            logger.warning(
                f"Transcription had {total_failed} failures across {len(inputs)} file(s). "
                f"Failure log: {self.failure_log_path}"
            )

        for path, output in outputs.items():
            transcribed = transcribed_by_path.get(path, {})
            stats = per_path_stats.get(path, FilePipelineStats())
            self._write_transcript(output, transcribed)
            self._write_metrics(
                output,
                {
                    "extract_seconds": stats.extract_seconds,
                    "transcribe_seconds": stats.transcribe_seconds,
                    "segments_attempted": stats.segments_attempted,
                    "segments_transcribed": stats.segments_transcribed,
                },
            )

        return outputs

    def transcribe(self, path: Path, chunk_indexes: list[int]) -> Path:
        return self.execute([(path, chunk_indexes)])[path]

    def _build_spec(self, path: Path, index: int, duration: int) -> SegmentSpec:
        base_offset = float(index * duration)
        offset = base_offset
        if self.misalignment_policy:
            video_id = f"{path.resolve()}|{self.variant_id}"
            offset += self.misalignment_policy.offset_for(video_id, index)
        return SegmentSpec(
            video_path=path,
            segment_index=index,
            base_offset_seconds=base_offset,
            effective_offset_seconds=max(0.0, offset),
            duration_seconds=duration,
        )

    def _transcription_output(self, path: Path) -> Path:
        if self.output_dir == self.series.transcriptions_text_dir:
            return self.series.transcription_file(path)
        return self.output_dir / self.series.transcription_file_name(path)

    @staticmethod
    def _io_workers() -> int:
        return max(1, int(os.environ.get("MEM_PIPELINE_IO_WORKERS", "2")))

    @staticmethod
    def _transcribe_workers() -> int:
        return max(1, int(os.environ.get("MEM_PIPELINE_TRANSCRIBE_WORKERS", "1")))

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

    @staticmethod
    def _metrics_path(output: Path) -> Path:
        return output.with_suffix(".metrics.json")

    @classmethod
    def _write_metrics(cls, output: Path, payload: dict) -> None:
        metrics_path = cls._metrics_path(output)
        with metrics_path.open("w", encoding="utf-8") as metrics_out:
            json.dump(payload, metrics_out)
