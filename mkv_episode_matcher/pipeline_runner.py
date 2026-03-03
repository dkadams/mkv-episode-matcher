from __future__ import annotations

import json
import multiprocessing
import os
import queue
import threading
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor
from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.misalignment import MisalignmentPolicy
from mkv_episode_matcher.pipeline_types import (
    ChunkTask,
    FilePipelineMetrics,
    PipelineRunResult,
    SegmentRequest,
    TranscriptionResultEvent,
)
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcribers import (
    SubprocessTranscriber,
)
from mkv_episode_matcher.transcription_worker import (
    _init_transcription_worker,
    _transcribe_segment_batch_task_worker,
)


class PipelineRunner:
    def __init__(
        self,
        *,
        config: Configuration,
        series: Series,
        transcriber_type: type,
        model_name: str,
        output_dir: Path,
        misalignment_policy: Optional[MisalignmentPolicy] = None,
        variant_id: Optional[str] = None,
    ):
        self.config = config
        self.series = series
        self.transcriber_type = transcriber_type
        self.model_name = model_name
        self.output_dir = output_dir
        self.misalignment_policy = misalignment_policy
        self.variant_id = variant_id or "aligned"
        self._lock = threading.Lock()
        self._failure_log_path = self._build_failure_log_path(output_dir)

    @staticmethod
    def _build_failure_log_path(output_dir: Path) -> Path:
        failure_log_dir = output_dir.parent / "failure-logs"
        failure_log_dir.mkdir(parents=True, exist_ok=True)
        return failure_log_dir / f"transcription-failures-{os.getpid()}.jsonl"

    def run(self, segments_to_transcribe: dict[Path, list[int]]) -> PipelineRunResult:
        requests_by_file = self._build_requests_by_file(segments_to_transcribe)
        result = PipelineRunResult()
        result.outputs = {
            path: self._output_path(path)
            for path in segments_to_transcribe
        }
        for output_path in result.outputs.values():
            result.transcripts_by_output[output_path] = {}
        for requests in requests_by_file.values():
            for request in requests:
                metrics = result.metrics_by_output.setdefault(
                    request.output_path,
                    FilePipelineMetrics(),
                )
                metrics.segments_attempted += 1

        if not requests_by_file:
            return result

        io_workers = max(1, self._configured_workers("io_workers", "MEM_IO_WORKERS", 2))
        transcribe_workers = max(
            1,
            self._configured_workers("transcribe_workers", "MEM_TRANSCRIBE_WORKERS", 4),
        )
        queue_size = max(1, io_workers * 8)
        file_queue: queue.Queue[list[SegmentRequest] | None] = queue.Queue()
        task_queue: queue.Queue[ChunkTask | None] = queue.Queue(maxsize=queue_size)
        result_queue: queue.Queue[tuple[Future, list[ChunkTask]] | None] = queue.Queue()
        delete_queue: queue.Queue[tuple[Path, Path] | None] = queue.Queue(maxsize=queue_size)
        chunk_refcounts: dict[Path, int] = {}
        chunk_output_paths: dict[Path, Path] = {}
        refcount_lock = threading.Lock()

        file_jobs = sorted(requests_by_file.values(), key=len, reverse=True)
        for file_requests in file_jobs:
            file_queue.put(file_requests)
        for _ in range(io_workers):
            file_queue.put(None)

        with AudioChunkExtractor(manage_lifecycle=False) as audio_extractor:
            with self._make_transcribe_executor(transcribe_workers) as transcribers:
                io_threads = [
                    threading.Thread(
                        target=self._io_worker,
                        args=(
                            file_queue,
                            task_queue,
                            audio_extractor,
                            result,
                            chunk_refcounts,
                            chunk_output_paths,
                            refcount_lock,
                        ),
                        daemon=True,
                    )
                    for _ in range(io_workers)
                ]
                dispatcher = threading.Thread(
                    target=self._dispatch_transcribe_tasks,
                    args=(task_queue, result_queue, transcribers, io_workers, result),
                    daemon=True,
                )
                delete_thread = threading.Thread(
                    target=self._delete_worker,
                    args=(delete_queue, result),
                    daemon=True,
                )
                for thread in io_threads:
                    thread.start()
                dispatcher.start()
                delete_thread.start()

                completed = 0
                while True:
                    wait_before = time.perf_counter()
                    done_item = result_queue.get()
                    wait_elapsed = time.perf_counter() - wait_before
                    if done_item is None:
                        break
                    future, batch_tasks = done_item
                    completed += len(batch_tasks)
                    try:
                        events = future.result()
                    except Exception as exc:  # noqa: BLE001
                        self._append_failure(
                            result,
                            self._base_failure_payload() | {
                                "failure_type": "transcribe_future_exception",
                                "error": str(exc),
                                "batch_size": len(batch_tasks),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        for task in batch_tasks:
                            self._release_chunk(
                                chunk_refcounts=chunk_refcounts,
                                chunk_output_paths=chunk_output_paths,
                                refcount_lock=refcount_lock,
                                chunk_path=task.chunk_path,
                                output_path=task.output_path,
                                delete_queue=delete_queue,
                            )
                        continue

                    if len(events) != len(batch_tasks):
                        self._append_failure(
                            result,
                            self._base_failure_payload() | {
                                "failure_type": "batch_result_mismatch",
                                "expected_results": len(batch_tasks),
                                "actual_results": len(events),
                            },
                        )
                        if len(events) < len(batch_tasks):
                            missing = batch_tasks[len(events):]
                            events.extend(
                                [
                                    TranscriptionResultEvent(
                                        video_path=task.video_path,
                                        output_path=task.output_path,
                                        segment_index=task.segment_index,
                                        text=None,
                                        transcribe_seconds=0.0,
                                        failure={
                                            "failure_type": "batch_result_mismatch",
                                            "video_path": str(task.video_path),
                                            "segment_index": task.segment_index,
                                            "chunk_path": str(task.chunk_path),
                                            "base_offset_seconds": task.base_offset_seconds,
                                            "effective_offset_seconds": task.effective_offset_seconds,
                                            "duration_seconds": task.duration_seconds,
                                        },
                                    )
                                    for task in missing
                                ]
                            )
                        else:
                            events = events[:len(batch_tasks)]

                    for task, event in zip(batch_tasks, events):
                        metrics = result.metrics_by_output[event.output_path]
                        metrics.transcribe_seconds += event.transcribe_seconds
                        metrics.stage_b_result_wait_seconds += wait_elapsed
                        if event.text:
                            result.transcripts_by_output[event.output_path][event.segment_index] = event.text
                            metrics.segments_transcribed += 1
                        if event.failure:
                            self._append_failure(result, event.failure)
                        self._release_chunk(
                            chunk_refcounts=chunk_refcounts,
                            chunk_output_paths=chunk_output_paths,
                            refcount_lock=refcount_lock,
                            chunk_path=task.chunk_path,
                            output_path=task.output_path,
                            delete_queue=delete_queue,
                        )

                for thread in io_threads:
                    thread.join()
                dispatcher.join()
                with refcount_lock:
                    leftovers = list(chunk_output_paths.items())
                    chunk_refcounts.clear()
                    chunk_output_paths.clear()
                for chunk_path, output_path in leftovers:
                    delete_queue.put((chunk_path, output_path))
                delete_queue.put(None)
                delete_thread.join()

        if completed == 0:
            logger.warning("Pipeline completed without transcription futures")
        return result

    def _build_requests_by_file(
        self,
        segments_to_transcribe: dict[Path, list[int]],
    ) -> dict[Path, list[SegmentRequest]]:
        duration = self.series.segment_duration
        grouped: dict[Path, list[SegmentRequest]] = {}
        for path, segment_indexes in segments_to_transcribe.items():
            output_path = self._output_path(path)
            requests: list[SegmentRequest] = []
            for index in segment_indexes:
                base_offset = float(index * duration)
                offset = base_offset
                if self.misalignment_policy:
                    video_id = f"{path.resolve()}|{self.variant_id}"
                    offset += self.misalignment_policy.offset_for(video_id, index)
                requests.append(
                    SegmentRequest(
                        video_path=path,
                        output_path=output_path,
                        segment_index=index,
                        base_offset_seconds=base_offset,
                        effective_offset_seconds=max(0.0, offset),
                        duration_seconds=duration,
                    )
                )
            requests.sort(key=lambda req: req.effective_offset_seconds)
            grouped[path] = requests
        return grouped

    def _io_worker(
        self,
        file_queue: queue.Queue[list[SegmentRequest] | None],
        task_queue: queue.Queue[ChunkTask | None],
        extractor: AudioChunkExtractor,
        result: PipelineRunResult,
        chunk_refcounts: dict[Path, int],
        chunk_output_paths: dict[Path, Path],
        refcount_lock: threading.Lock,
    ) -> None:
        while True:
            file_requests = file_queue.get()
            if file_requests is None:
                task_queue.put(None)
                return

            for request in file_requests:
                before = time.perf_counter()
                try:
                    chunk_path = extractor.extract(
                        request.video_path,
                        request.effective_offset_seconds,
                        request.duration_seconds,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._append_failure(
                        result,
                        self._base_failure_payload() | {
                            "failure_type": "audio_extract_exception",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "video_path": str(request.video_path),
                            "segment_index": request.segment_index,
                            "base_offset_seconds": request.base_offset_seconds,
                            "effective_offset_seconds": request.effective_offset_seconds,
                            "duration_seconds": request.duration_seconds,
                        },
                    )
                    continue
                extract_elapsed = time.perf_counter() - before
                task = ChunkTask(
                    kind="chunk_path",
                    video_path=request.video_path,
                    output_path=request.output_path,
                    segment_index=request.segment_index,
                    chunk_path=chunk_path,
                    base_offset_seconds=request.base_offset_seconds,
                    effective_offset_seconds=request.effective_offset_seconds,
                    duration_seconds=request.duration_seconds,
                )
                metrics = result.metrics_by_output[request.output_path]
                with self._lock:
                    metrics.extract_seconds += extract_elapsed
                self._retain_chunk(
                    chunk_refcounts=chunk_refcounts,
                    chunk_output_paths=chunk_output_paths,
                    refcount_lock=refcount_lock,
                    chunk_path=chunk_path,
                    output_path=request.output_path,
                )
                queue_before = time.perf_counter()
                task_queue.put(task)
                queue_elapsed = time.perf_counter() - queue_before
                with self._lock:
                    metrics.stage_a_queue_wait_seconds += queue_elapsed

    def _dispatch_transcribe_tasks(
        self,
        task_queue: queue.Queue[ChunkTask | None],
        result_queue: queue.Queue[tuple[Future, list[ChunkTask]] | None],
        transcribers,
        io_workers: int,
        result: PipelineRunResult,
    ) -> None:
        sentinels = 0
        pending: set[Future] = set()
        pending_lock = threading.Lock()
        batch_size = self._microbatch_size()
        batch_wait_seconds = self._microbatch_max_wait_seconds()
        batch_buffer: list[ChunkTask] = []
        batch_started_at: float | None = None

        def submit_batch(tasks: list[ChunkTask]) -> None:
            if not tasks:
                return
            submit_before = time.perf_counter()
            try:
                future = transcribers.submit(_transcribe_segment_batch_task_worker, tasks)
            except Exception as exc:  # noqa: BLE001
                failed = Future()
                failed.set_exception(exc)
                result_queue.put((failed, list(tasks)))
                return
            submit_elapsed = time.perf_counter() - submit_before
            with pending_lock:
                pending.add(future)

            def _on_done(done: Future, submitted_tasks: list[ChunkTask] = list(tasks)) -> None:
                with pending_lock:
                    pending.discard(done)
                result_queue.put((done, submitted_tasks))

            future.add_done_callback(_on_done)
            per_task_submit = submit_elapsed / len(tasks)
            with self._lock:
                for task in tasks:
                    result.metrics_by_output[task.output_path].stage_b_submit_seconds += per_task_submit

            # keep the submit elapsed in logs for observability.
            if submit_elapsed > 0.05:
                logger.debug(f"Stage B submit took {submit_elapsed:.3f}s for batch of {len(tasks)}")

        while sentinels < io_workers or batch_buffer:
            if not batch_buffer:
                task = task_queue.get()
            else:
                deadline = (batch_started_at or time.perf_counter()) + batch_wait_seconds
                timeout = max(0.0, deadline - time.perf_counter())
                try:
                    task = task_queue.get(timeout=timeout)
                except queue.Empty:
                    submit_batch(batch_buffer)
                    batch_buffer = []
                    batch_started_at = None
                    continue

            if task is None:
                sentinels += 1
                if sentinels >= io_workers:
                    submit_batch(batch_buffer)
                    batch_buffer = []
                    batch_started_at = None
                continue

            if not batch_buffer:
                batch_started_at = time.perf_counter()
            batch_buffer.append(task)
            if len(batch_buffer) >= batch_size:
                submit_batch(batch_buffer)
                batch_buffer = []
                batch_started_at = None

        while True:
            with pending_lock:
                if not pending:
                    break
            time.sleep(0.01)
        result_queue.put(None)

    def _retain_chunk(
        self,
        *,
        chunk_refcounts: dict[Path, int],
        chunk_output_paths: dict[Path, Path],
        refcount_lock: threading.Lock,
        chunk_path: Path,
        output_path: Path,
    ) -> None:
        with refcount_lock:
            chunk_refcounts[chunk_path] = chunk_refcounts.get(chunk_path, 0) + 1
            chunk_output_paths.setdefault(chunk_path, output_path)

    def _release_chunk(
        self,
        *,
        chunk_refcounts: dict[Path, int],
        chunk_output_paths: dict[Path, Path],
        refcount_lock: threading.Lock,
        chunk_path: Path,
        output_path: Path,
        delete_queue: queue.Queue[tuple[Path, Path] | None],
    ) -> None:
        should_delete = False
        resolved_output = output_path
        with refcount_lock:
            remaining = chunk_refcounts.get(chunk_path, 0) - 1
            if remaining <= 0:
                chunk_refcounts.pop(chunk_path, None)
                resolved_output = chunk_output_paths.pop(chunk_path, output_path)
                should_delete = True
            else:
                chunk_refcounts[chunk_path] = remaining
        if should_delete:
            delete_queue.put((chunk_path, resolved_output))

    def _delete_worker(
        self,
        delete_queue: queue.Queue[tuple[Path, Path] | None],
        result: PipelineRunResult,
    ) -> None:
        while True:
            item = delete_queue.get()
            if item is None:
                return
            chunk_path, output_path = item
            before = time.perf_counter()
            try:
                chunk_path.unlink(missing_ok=True)
                elapsed = time.perf_counter() - before
                with self._lock:
                    metrics = result.metrics_by_output.get(output_path)
                    if metrics is not None:
                        metrics.delete_seconds += elapsed
                        metrics.deleted_chunk_count += 1
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - before
                with self._lock:
                    metrics = result.metrics_by_output.get(output_path)
                    if metrics is not None:
                        metrics.delete_seconds += elapsed
                        metrics.delete_failures += 1
                self._append_failure(
                    result,
                    self._base_failure_payload() | {
                        "failure_type": "chunk_delete_exception",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "chunk_path": str(chunk_path),
                        "output_path": str(output_path),
                    },
                )

    def write_outputs(self, result: PipelineRunResult) -> dict[Path, Path]:
        for output_path, transcribed in result.transcripts_by_output.items():
            self._write_transcript(output_path, transcribed)
            metrics = result.metrics_by_output.get(output_path, FilePipelineMetrics())
            self._write_metrics(
                output_path,
                {
                    "extract_seconds": metrics.extract_seconds,
                    "transcribe_seconds": metrics.transcribe_seconds,
                    "segments_attempted": metrics.segments_attempted,
                    "segments_transcribed": metrics.segments_transcribed,
                },
            )
        return {path: output for path, output in result.outputs.items()}

    def _output_path(self, path: Path) -> Path:
        if self.output_dir == self.series.transcriptions_text_dir:
            return self.series.transcription_file(path)
        return self.output_dir / self.series.transcription_file_name(path)

    def _append_failure(self, result: PipelineRunResult, payload: dict) -> None:
        with self._lock:
            result.failures.append(payload)
        self._write_failure(payload)

    def _write_failure(self, payload: dict) -> None:
        entry = self._base_failure_payload() | payload
        with self._failure_log_path.open("a", encoding="utf-8") as log_out:
            log_out.write(json.dumps(entry, ensure_ascii=False))
            log_out.write("\n")

    def _base_failure_payload(self) -> dict:
        return {
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
        }

    def _make_transcribe_executor(self, transcribe_workers: int):
        initargs = (
            self.config,
            self.series,
            self.transcriber_type,
            self.model_name,
            self.misalignment_policy,
            self.variant_id,
            self.output_dir,
        )
        if issubclass(self.transcriber_type, SubprocessTranscriber):
            return ThreadPoolExecutor(
                max_workers=transcribe_workers,
                initializer=_init_transcription_worker,
                initargs=initargs,
            )
        return ProcessPoolExecutor(
            max_workers=transcribe_workers,
            initializer=_init_transcription_worker,
            initargs=initargs,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def _configured_workers(self, arg_name: str, env_name: str, default_value: int) -> int:
        args = getattr(self.config, "args", None)
        if args is not None:
            configured = getattr(args, arg_name, None)
            if configured is not None:
                return int(configured)
        return int(os.environ.get(env_name, str(default_value)))

    @staticmethod
    def _positive_int_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        value = str(raw).strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError:
            logger.warning(f"Ignoring {name}: expected integer, got '{raw}'")
            return None
        if parsed < 1:
            logger.warning(f"Ignoring {name}: expected value >= 1, got '{parsed}'")
            return None
        return parsed

    def _microbatch_size(self) -> int:
        override = self._positive_int_env("MEM_TRANSCRIBE_MICROBATCH_SIZE")
        if override is not None:
            return override

        default_size = getattr(self.transcriber_type, "DEFAULT_MICROBATCH_SIZE", 1)
        try:
            resolved = int(default_size)
        except (TypeError, ValueError):
            resolved = 1
        return max(1, resolved)

    @staticmethod
    def _microbatch_max_wait_seconds() -> float:
        raw = str(os.environ.get("MEM_TRANSCRIBE_MICROBATCH_MAX_WAIT_MS", "15")).strip()
        if not raw:
            return 0.015
        try:
            ms = float(raw)
        except ValueError:
            logger.warning(
                f"Ignoring MEM_TRANSCRIBE_MICROBATCH_MAX_WAIT_MS: expected number, got '{raw}'"
            )
            return 0.015
        if ms < 0:
            logger.warning(
                "Ignoring MEM_TRANSCRIBE_MICROBATCH_MAX_WAIT_MS: expected value >= 0, got '{}'",
                ms,
            )
            return 0.015
        return ms / 1000.0

    @staticmethod
    def _write_transcript(output: Path, transcribed: dict[int, str]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        existing_transcript = {}
        if output.exists():
            with output.open("r", encoding="utf-8") as existing:
                existing_transcript = json.load(existing)
        with output.open("w", encoding="utf-8") as json_out:
            full_transcript = existing_transcript | transcribed
            json.dump(full_transcript, json_out)

    @staticmethod
    def _write_metrics(output: Path, payload: dict) -> None:
        metrics_path = output.with_suffix(".metrics.json")
        with metrics_path.open("w", encoding="utf-8") as metrics_out:
            json.dump(payload, metrics_out)
