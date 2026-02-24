from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SegmentRequest:
    video_path: Path
    output_path: Path
    segment_index: int
    base_offset_seconds: float
    effective_offset_seconds: float
    duration_seconds: int


@dataclass(frozen=True)
class ChunkTask:
    kind: str
    video_path: Path
    output_path: Path
    segment_index: int
    chunk_path: Path
    base_offset_seconds: float
    effective_offset_seconds: float
    duration_seconds: int


@dataclass(frozen=True)
class SampleTask:
    kind: str
    video_path: Path
    output_path: Path
    segment_index: int
    sample_rate: int
    samples: Any
    base_offset_seconds: float
    effective_offset_seconds: float
    duration_seconds: int


@dataclass(frozen=True)
class TranscriptionResultEvent:
    video_path: Path
    output_path: Path
    segment_index: int
    text: str | None
    transcribe_seconds: float
    failure: dict | None = None


@dataclass
class FilePipelineMetrics:
    extract_seconds: float = 0.0
    transcribe_seconds: float = 0.0
    segments_attempted: int = 0
    segments_transcribed: int = 0
    stage_a_queue_wait_seconds: float = 0.0
    stage_b_submit_seconds: float = 0.0
    stage_b_result_wait_seconds: float = 0.0
    delete_seconds: float = 0.0
    deleted_chunk_count: int = 0
    delete_failures: int = 0


@dataclass
class PipelineRunResult:
    outputs: dict[Path, Path] = field(default_factory=dict)
    transcripts_by_output: dict[Path, dict[int, str]] = field(default_factory=dict)
    metrics_by_output: dict[Path, FilePipelineMetrics] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
