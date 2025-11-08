from __future__ import annotations

import math
import random
import time

from loguru import logger

from mkv_episode_matcher import video_helper as get_video_duration
from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor


class SegmentTranscriber:
    def __init__(self, model_name, transcriber):
        self.transcriber = self._init_transcriber(model_name, transcriber)

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

        return str(transcript)

    def get_random_segments(self, path, duration, count):
        total_duration = get_video_duration.get_video_duration(path)
        chunks_per_file = math.ceil(total_duration / duration)
        count = min(chunks_per_file, count)

        # use a fixed seed so that we choose the same chunks for each file
        random.seed(12345)

        # TODO bias this towards the middle of the file?
        chunk_indexes = random.sample(range(chunks_per_file), count)

        results = []
        total_extract_time = 0
        total_transcribe_time = 0
        with AudioChunkExtractor() as audio_extractor:
            for index in chunk_indexes:
                offset = index * duration

                before = time.time()
                chunk_path = audio_extractor.extract(path, offset, duration)
                total_extract_time += time.time() - before

                before = time.time()
                raw_transcript = self.transcriber.transcribe(chunk_path)
                text = self._normalize_transcript(raw_transcript)
                total_transcribe_time += time.time() - before

                if text:
                    results.append((index, text))
                else:
                    logger.warning(f"Failed to transcribe {chunk_path}")

        logger.info(f"Extracted {count} audio chunks "
                    f"in {total_extract_time:.2f}s, transcribed "
                    f"in {total_transcribe_time:.2f}s")
        return results

    def get_text_segments(self, path, duration=30, count=10):
        segments = self.get_random_segments(path, duration, count)
        return {index: text for index, text in segments}

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
