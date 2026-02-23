from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class SegmentSpec:
    video_path: Path
    segment_index: int
    base_offset_seconds: float
    effective_offset_seconds: float
    duration_seconds: int


@dataclass(frozen=True)
class SegmentTask:
    video_path: Path
    segment_index: int
    chunk_path: Path
    base_offset_seconds: float
    effective_offset_seconds: float
    duration_seconds: int


@dataclass
class FilePipelineStats:
    extract_seconds: float = 0.0
    transcribe_seconds: float = 0.0
    segments_attempted: int = 0
    segments_transcribed: int = 0
    segments_failed: int = 0


class TranscriptionPipeline:
    def __init__(
        self,
        *,
        audio_extractor,
        transcriber,
        normalize_transcript: Callable[[object], str | None],
        write_failure: Callable[[dict], None],
        io_workers: int = 1,
        transcribe_workers: int = 1,
        queue_size: int = 32,
        batch_size: int | None = None,
    ):
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.normalize_transcript = normalize_transcript
        self.write_failure = write_failure
        self.io_workers = max(1, int(io_workers))
        self.transcribe_workers = max(1, int(transcribe_workers))
        self.queue_size = max(1, int(queue_size))
        if batch_size is None:
            batch_size = int(os.environ.get("MEM_PIPELINE_BATCH_SIZE", "8"))
        self.batch_size = max(1, int(batch_size))

        self._per_path_stats: dict[Path, FilePipelineStats] = {}
        self._transcribed_by_path: dict[Path, dict[int, str]] = {}
        self._lock = threading.Lock()

    def run(self, specs_by_path: dict[Path, list[SegmentSpec]]) -> tuple[dict[Path, dict[int, str]], dict[Path, FilePipelineStats]]:
        ordered_specs: list[SegmentSpec] = []
        for path, specs in specs_by_path.items():
            self._per_path_stats[path] = FilePipelineStats(segments_attempted=len(specs))
            self._transcribed_by_path[path] = {}
            ordered_specs.extend(sorted(specs, key=lambda spec: spec.effective_offset_seconds))

        spec_queue: queue.Queue[SegmentSpec | None] = queue.Queue()
        task_queue: queue.Queue[SegmentTask | None] = queue.Queue(maxsize=self.queue_size)

        for spec in ordered_specs:
            spec_queue.put(spec)
        for _ in range(self.io_workers):
            spec_queue.put(None)

        io_threads = [
            threading.Thread(target=self._io_worker, args=(spec_queue, task_queue), daemon=True)
            for _ in range(self.io_workers)
        ]
        tx_threads = [
            threading.Thread(target=self._transcribe_worker, args=(task_queue,), daemon=True)
            for _ in range(self.transcribe_workers)
        ]

        for thread in io_threads:
            thread.start()
        for thread in tx_threads:
            thread.start()

        for thread in io_threads:
            thread.join()

        for _ in range(self.transcribe_workers):
            task_queue.put(None)
        for thread in tx_threads:
            thread.join()

        return self._transcribed_by_path, self._per_path_stats

    def _io_worker(self, spec_queue: queue.Queue[SegmentSpec | None], task_queue: queue.Queue[SegmentTask | None]) -> None:
        while True:
            spec = spec_queue.get()
            if spec is None:
                return
            before = time.perf_counter()
            try:
                chunk_path = self.audio_extractor.extract(
                    spec.video_path,
                    spec.effective_offset_seconds,
                    spec.duration_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                self._record_failure({
                    "failure_type": "audio_extract_exception",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "video_path": str(spec.video_path),
                    "segment_index": spec.segment_index,
                    "base_offset_seconds": spec.base_offset_seconds,
                    "effective_offset_seconds": spec.effective_offset_seconds,
                    "duration_seconds": spec.duration_seconds,
                })
                continue

            extract_elapsed = time.perf_counter() - before
            with self._lock:
                stats = self._per_path_stats[spec.video_path]
                stats.extract_seconds += extract_elapsed

            task_queue.put(
                SegmentTask(
                    video_path=spec.video_path,
                    segment_index=spec.segment_index,
                    chunk_path=chunk_path,
                    base_offset_seconds=spec.base_offset_seconds,
                    effective_offset_seconds=spec.effective_offset_seconds,
                    duration_seconds=spec.duration_seconds,
                )
            )

    def _transcribe_worker(self, task_queue: queue.Queue[SegmentTask | None]) -> None:
        use_batch = hasattr(self.transcriber, "transcribe_many")
        while True:
            first = task_queue.get()
            if first is None:
                return
            batch = [first]
            if use_batch:
                while len(batch) < self.batch_size:
                    try:
                        maybe = task_queue.get_nowait()
                    except queue.Empty:
                        break
                    if maybe is None:
                        task_queue.put(None)
                        break
                    batch.append(maybe)
                self._transcribe_batch(batch)
            else:
                self._transcribe_single(first)

    def _transcribe_batch(self, batch: list[SegmentTask]) -> None:
        before = time.perf_counter()
        try:
            raw_results = self.transcriber.transcribe_many([task.chunk_path for task in batch])
        except Exception as exc:  # noqa: BLE001
            for task in batch:
                self._record_failure({
                    "failure_type": "batch_transcribe_exception",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "video_path": str(task.video_path),
                    "segment_index": task.segment_index,
                    "chunk_path": str(task.chunk_path),
                    "duration_seconds": task.duration_seconds,
                })
            raw_results = [None] * len(batch)

        elapsed = time.perf_counter() - before
        for task, raw in zip(batch, raw_results):
            self._record_transcribe(task, raw, elapsed / max(1, len(batch)))

    def _transcribe_single(self, task: SegmentTask) -> None:
        before = time.perf_counter()
        try:
            raw = self.transcriber.transcribe(task.chunk_path)
        except Exception as exc:  # noqa: BLE001
            self._record_failure({
                "failure_type": "transcribe_exception",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "video_path": str(task.video_path),
                "segment_index": task.segment_index,
                "chunk_path": str(task.chunk_path),
                "base_offset_seconds": task.base_offset_seconds,
                "effective_offset_seconds": task.effective_offset_seconds,
                "duration_seconds": task.duration_seconds,
            })
            return
        elapsed = time.perf_counter() - before
        self._record_transcribe(task, raw, elapsed)

    def _record_transcribe(self, task: SegmentTask, raw: object, elapsed: float) -> None:
        text = self.normalize_transcript(raw)
        with self._lock:
            self._per_path_stats[task.video_path].transcribe_seconds += elapsed
        if text:
            with self._lock:
                self._transcribed_by_path[task.video_path][task.segment_index] = text
                self._per_path_stats[task.video_path].segments_transcribed += 1
            return

        raw_preview = str(raw)
        if len(raw_preview) > 300:
            raw_preview = raw_preview[:300] + "...[truncated]"
        self._record_failure({
            "failure_type": "empty_transcript",
            "video_path": str(task.video_path),
            "segment_index": task.segment_index,
            "chunk_path": str(task.chunk_path),
            "base_offset_seconds": task.base_offset_seconds,
            "effective_offset_seconds": task.effective_offset_seconds,
            "duration_seconds": task.duration_seconds,
            "raw_transcript_type": type(raw).__name__,
            "raw_transcript_preview": raw_preview,
        })

    def _record_failure(self, payload: dict) -> None:
        self.write_failure(payload)
        path = payload.get("video_path")
        if not path:
            return
        with self._lock:
            self._per_path_stats[Path(path)].segments_failed += 1
