from pathlib import Path
from typing import Optional

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.segment_transcriber import SegmentTranscriber
from mkv_episode_matcher.series import Series

_PROCESS_TEXT_EXTRACTOR: Optional[SegmentTranscriber] = None

def _init_transcription_worker(config: Configuration, series: Series,
    transcriber: type, model_name: str):
    """Initializer for the process pool so Whisper loads only in child processes."""
    global _PROCESS_TEXT_EXTRACTOR

    if _PROCESS_TEXT_EXTRACTOR is None:
        _PROCESS_TEXT_EXTRACTOR = SegmentTranscriber(config, series,
                                                     model_name, transcriber)

def _extract_text_segments_worker(inputs: list[tuple[Path, list[int]]]) -> dict[Path, Path]:
    """Extract text segments for a single file inside a worker process."""
    return _PROCESS_TEXT_EXTRACTOR.execute(inputs)
