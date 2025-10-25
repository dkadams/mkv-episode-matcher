from pathlib import Path
from typing import Optional, List, Tuple

from mkv_episode_matcher.text_segment_extractor import TextSegmentExtractor

_PROCESS_TEXT_EXTRACTOR: Optional[TextSegmentExtractor] = None
_DURATION = None
_COUNT = None
def _init_text_extractor_worker(model_name: str, duration: int, count: int):
    """Initializer for the process pool so Whisper loads only in child processes."""
    global _PROCESS_TEXT_EXTRACTOR, _DURATION, _COUNT
    if _PROCESS_TEXT_EXTRACTOR is None:
        _PROCESS_TEXT_EXTRACTOR = TextSegmentExtractor(model_name)

    _DURATION = duration
    _COUNT = count

def _extract_text_segments_worker(file: Path) -> List[Tuple[int, str]]:
    """Extract text segments for a single file inside a worker process."""
    return _PROCESS_TEXT_EXTRACTOR.get_text_segments(file, _DURATION, _COUNT)
