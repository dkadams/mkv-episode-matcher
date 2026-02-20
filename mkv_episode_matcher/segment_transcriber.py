from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

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
        return {path: self.transcribe(path, chunk_indexes)
                for path, chunk_indexes in input}

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
        with AudioChunkExtractor() as audio_extractor:
            for index in chunk_indexes:
                offset = float(index * duration)
                if self.misalignment_policy:
                    video_id = f"{path.resolve()}|{self.variant_id}"
                    offset += self.misalignment_policy.offset_for(video_id, index)

                before = time.time()
                chunk_path = audio_extractor.extract(path, offset, duration)
                total_extract_time += time.time() - before

                before = time.time()
                raw_transcript = self.transcriber.transcribe(chunk_path)
                text = self._normalize_transcript(raw_transcript)
                total_transcribe_time += time.time() - before

                if text:
                    transcribed[index] = text
                else:
                    logger.warning(f"Failed to transcribe {chunk_path}")

        logger.info(f"Extracted {len(transcribed)} audio chunks "
                    f"in {total_extract_time:.2f}s, transcribed "
                    f"in {total_transcribe_time:.2f}s")

        existing_transcript = {}
        if output.exists():
            existing_transcript = json.load(open(output))

        with open(output, "w") as json_out:
            full_transcript = existing_transcript | transcribed
            json.dump(full_transcript, json_out)

        return output

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
