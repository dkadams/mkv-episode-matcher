import json
import hashlib
from pathlib import Path
from typing import Optional

from mkv_episode_matcher.segment_transcriber import SegmentTranscriber
from mkv_episode_matcher.series import Series

_PROCESS_TEXT_EXTRACTOR: Optional[SegmentTranscriber] = None
_DURATION = None
_COUNT = None

_OUTPUT_DIR: Optional[Path] = None
def _init_transcription_worker(series: Series,
    transcriber: type, model_name: str,
    duration: int, count: int):
    """Initializer for the process pool so Whisper loads only in child processes."""
    global _PROCESS_TEXT_EXTRACTOR, _DURATION, _COUNT, _OUTPUT_DIR

    if _PROCESS_TEXT_EXTRACTOR is None:
        _PROCESS_TEXT_EXTRACTOR = SegmentTranscriber(model_name, transcriber)

    _DURATION = duration
    _COUNT = count

    _OUTPUT_DIR = series.ensure_transcription_text_dir(duration, count)

def _extract_text_segments_worker(inputs: list[Path]) -> dict[Path, Path]:
    """Extract text segments for a single file inside a worker process."""
    return {input: transcribe(input) for input in inputs}

def transcribe(input: Path) -> Path:
    segments = _PROCESS_TEXT_EXTRACTOR.get_text_segments(input, _DURATION,
                                                         _COUNT)
    output = _OUTPUT_DIR / transcription_file_name(input)
    if not output.exists():
        with open(output, "w") as json_out:
            json.dump(segments, json_out)

    return output

def transcription_file_name(input: Path) -> str:
    path = input.resolve()

    path_bytes = str(path).encode("utf-8")
    path_hash = hashlib.sha256(path_bytes).hexdigest()

    # The parent should enough to uniquely identify the file, but including the
    # path hash ensures uniqueness. Adding the parent directory name helps in
    # identifying the original file path.
    return f"{path_hash}_{input.parent.name}_{input.with_suffix('.json').name}"
