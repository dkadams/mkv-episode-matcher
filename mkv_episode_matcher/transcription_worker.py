import time
import traceback
from pathlib import Path
from typing import Optional

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.misalignment import MisalignmentPolicy
from mkv_episode_matcher.pipeline_types import ChunkTask, SampleTask, TranscriptionResultEvent
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


def _transcribe_segment_task_worker(
    task: ChunkTask | SampleTask,
) -> TranscriptionResultEvent:
    if isinstance(task, SampleTask):
        return TranscriptionResultEvent(
            video_path=task.video_path,
            output_path=task.output_path,
            segment_index=task.segment_index,
            text=None,
            transcribe_seconds=0.0,
            failure={
                "failure_type": "sample_task_not_supported",
                "video_path": str(task.video_path),
                "segment_index": task.segment_index,
                "duration_seconds": task.duration_seconds,
            },
        )

    before = time.perf_counter()
    try:
        raw = _PROCESS_TEXT_EXTRACTOR.transcriber.transcribe(task.chunk_path)
    except Exception as exc:  # noqa: BLE001
        return TranscriptionResultEvent(
            video_path=task.video_path,
            output_path=task.output_path,
            segment_index=task.segment_index,
            text=None,
            transcribe_seconds=time.perf_counter() - before,
            failure={
                "failure_type": "transcribe_exception",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "video_path": str(task.video_path),
                "segment_index": task.segment_index,
                "chunk_path": str(task.chunk_path),
                "base_offset_seconds": task.base_offset_seconds,
                "effective_offset_seconds": task.effective_offset_seconds,
                "duration_seconds": task.duration_seconds,
            },
        )

    elapsed = time.perf_counter() - before
    text = SegmentTranscriber._normalize_transcript(raw)
    if text:
        return TranscriptionResultEvent(
            video_path=task.video_path,
            output_path=task.output_path,
            segment_index=task.segment_index,
            text=text,
            transcribe_seconds=elapsed,
        )

    raw_preview = str(raw)
    if len(raw_preview) > 300:
        raw_preview = raw_preview[:300] + "...[truncated]"
    return TranscriptionResultEvent(
        video_path=task.video_path,
        output_path=task.output_path,
        segment_index=task.segment_index,
        text=None,
        transcribe_seconds=elapsed,
        failure={
            "failure_type": "empty_transcript",
            "video_path": str(task.video_path),
            "segment_index": task.segment_index,
            "chunk_path": str(task.chunk_path),
            "base_offset_seconds": task.base_offset_seconds,
            "effective_offset_seconds": task.effective_offset_seconds,
            "duration_seconds": task.duration_seconds,
            "raw_transcript_type": type(raw).__name__,
            "raw_transcript_preview": raw_preview,
        },
    )
