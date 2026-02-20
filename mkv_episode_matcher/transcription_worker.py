from pathlib import Path
from typing import Optional

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.misalignment import MisalignmentPolicy
from mkv_episode_matcher.segment_transcriber import SegmentTranscriber
from mkv_episode_matcher.series import Series

_PROCESS_TEXT_EXTRACTOR: Optional[SegmentTranscriber] = None

def _init_transcription_worker(config: Configuration, series: Series,
    transcriber: type, model_name: str,
    misalignment_policy: Optional[MisalignmentPolicy] = None,
    variant_id: Optional[str] = None,
    output_dir: Optional[Path] = None):
    """Initializer for the process pool so Whisper loads only in child processes."""
    global _PROCESS_TEXT_EXTRACTOR

    # Always reinitialize when a new executor/pool is created so each benchmark
    # backend uses its own transcriber implementation.
    _PROCESS_TEXT_EXTRACTOR = SegmentTranscriber(config, series,
                                                 model_name, transcriber,
                                                 misalignment_policy=misalignment_policy,
                                                 variant_id=variant_id,
                                                 output_dir=output_dir)

def _extract_text_segments_worker(inputs: list[tuple[Path, list[int]]]) -> dict[Path, Path]:
    """Extract text segments for a single file inside a worker process."""
    return _PROCESS_TEXT_EXTRACTOR.execute(inputs)
