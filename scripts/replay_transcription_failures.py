#!/usr/bin/env python3
from __future__ import annotations

import argparse
from argparse import Namespace
from collections import defaultdict
import json
import multiprocessing
import sys
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.segment_transcriber import SegmentTranscriber
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcribers import (
    FasterWhisperTranscriber,
    ParakeetMlxCliTranscriber,
    ParakeetMlxGenerateBatchTranscriber,
    SubprocessTranscriber,
    WhisperKitCliTranscriber,
    WhisperTranscriber,
    WhispercppCliTranscriber,
)
from mkv_episode_matcher.transcription_worker import (
    _extract_text_segments_worker,
    _init_transcription_worker,
)
from mkv_episode_matcher.utils import unique_filename

TRANSCRIBERS = {
    "whisper": WhisperTranscriber,
    "faster-whisper": FasterWhisperTranscriber,
    "whispercpp-cli": WhispercppCliTranscriber,
    "whisperkit-cli": WhisperKitCliTranscriber,
    "parakeet-mlx": ParakeetMlxCliTranscriber,
    "parakeet-mlx-batch": ParakeetMlxGenerateBatchTranscriber,
}


@dataclass(frozen=True)
class ReplayTask:
    video_path: Path
    segment_index: int
    duration_seconds: float
    effective_offset_seconds: float
    profile: str
    variant_id: str
    misalign_min_seconds: float | None
    misalign_max_seconds: float | None
    misalign_seed: int | None
    failure_type: str
    source_log: Path
    source_line: int

    @property
    def base_offset_seconds(self) -> float:
        return float(self.segment_index) * float(self.duration_seconds)

    @property
    def replay_offset_seconds(self) -> float:
        return float(self.effective_offset_seconds) - self.base_offset_seconds


@dataclass(frozen=True)
class PolicySignature:
    variant_id: str
    profile: str
    min_seconds: float | None
    max_seconds: float | None
    seed: int | None


@dataclass
class ReplayMisalignmentPolicy:
    profile: str
    min_seconds: float | None
    max_seconds: float | None
    seed: int | None
    offset_by_video_segment: dict[tuple[str, int], float]

    def offset_for(self, video_id: str, segment_index: int) -> float:
        video_path = video_id.split("|", 1)[0]
        return self.offset_by_video_segment.get((video_path, segment_index), 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay failed transcription intervals using production transcription code paths."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory that contains .work/**/failure-logs/*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for replay logs and transcripts.",
    )
    parser.add_argument(
        "--logs-glob",
        default=".work/**/failure-logs/transcription-failures-*.jsonl",
        help="Glob pattern (relative to dataset-dir) for failure logs.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Optional profile filter. Example: --profiles left right random",
    )
    parser.add_argument(
        "--failure-types",
        nargs="+",
        default=["empty_transcript", "transcribe_exception", "batch_transcribe_exception"],
        help="Failure types to replay.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of tasks to replay.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["single", "mp1", "mp"],
        default=["single", "mp1", "mp"],
        help="Replay execution modes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker count for mp mode.",
    )
    parser.add_argument(
        "--transcriber",
        choices=sorted(TRANSCRIBERS.keys()),
        default="parakeet-mlx-batch",
        help="Transcriber backend to replay with.",
    )
    parser.add_argument(
        "--model-name",
        default="small.en",
        help="Model name passed to transcriber class (same behavior as production).",
    )
    parser.add_argument(
        "--replay-log",
        type=Path,
        default=None,
        help=(
            "Optional prior replay stdout/stderr log file. "
            "When set with --replay-log-error-types, only tasks with matching "
            "chunk-level errors are replayed."
        ),
    )
    parser.add_argument(
        "--replay-log-error-types",
        nargs="+",
        choices=["metal_malloc", "load_audio_invalid", "empty_text"],
        default=None,
        help=(
            "Error types to select from --replay-log. "
            "metal_malloc=MLX allocation errors, "
            "load_audio_invalid=Failed to load audio/invalid input, "
            "empty_text=AlignedResult text is empty."
        ),
    )
    return parser.parse_args()


def _chunked(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _extract_effective_offset_seconds(payload: dict) -> float | None:
    value = payload.get("effective_offset_seconds")
    if isinstance(value, (float, int)):
        return max(0.0, float(value))

    chunk_path = payload.get("chunk_path")
    if isinstance(chunk_path, str):
        start_idx = chunk_path.rfind(".AT")
        end_idx = chunk_path.rfind("ms.wav")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx + 3:
            ms_str = chunk_path[start_idx + 3:end_idx]
            if ms_str.isdigit():
                return int(ms_str) / 1000.0

    segment_index = payload.get("segment_index")
    duration_seconds = payload.get("duration_seconds")
    if isinstance(segment_index, int) and isinstance(duration_seconds, (float, int)):
        return max(0.0, float(segment_index) * float(duration_seconds))
    return None


def _build_tasks(
    dataset_dir: Path,
    logs_glob: str,
    profiles: set[str] | None,
    failure_types: set[str],
) -> tuple[list[ReplayTask], list[str]]:
    tasks: list[ReplayTask] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, int]] = set()

    for log_path in sorted(dataset_dir.glob(logs_glob)):
        if not log_path.is_file():
            continue
        with log_path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(f"{log_path}:{line_num}: invalid json ({exc})")
                    continue

                failure_type = str(payload.get("failure_type") or "")
                if failure_type not in failure_types:
                    continue

                profile = str(payload.get("misalignment_profile") or "aligned")
                if profiles and profile not in profiles:
                    continue

                video = payload.get("video_path")
                segment_index = payload.get("segment_index")
                duration_seconds = payload.get("duration_seconds")
                if not isinstance(video, str) or not isinstance(segment_index, int):
                    warnings.append(f"{log_path}:{line_num}: missing video or segment index")
                    continue
                if not isinstance(duration_seconds, (float, int)) or duration_seconds <= 0:
                    warnings.append(f"{log_path}:{line_num}: invalid duration_seconds")
                    continue
                effective_offset = _extract_effective_offset_seconds(payload)
                if effective_offset is None:
                    warnings.append(f"{log_path}:{line_num}: could not determine effective offset")
                    continue

                variant_id = str(payload.get("variant_id") or profile)
                key = (str(Path(video).resolve()), variant_id, segment_index)
                if key in seen:
                    continue
                seen.add(key)

                tasks.append(
                    ReplayTask(
                        video_path=Path(video),
                        segment_index=segment_index,
                        duration_seconds=float(duration_seconds),
                        effective_offset_seconds=float(effective_offset),
                        profile=profile,
                        variant_id=variant_id,
                        misalign_min_seconds=(
                            float(payload["misalignment_min_seconds"])
                            if isinstance(payload.get("misalignment_min_seconds"), (float, int))
                            else None
                        ),
                        misalign_max_seconds=(
                            float(payload["misalignment_max_seconds"])
                            if isinstance(payload.get("misalignment_max_seconds"), (float, int))
                            else None
                        ),
                        misalign_seed=(
                            int(payload["misalignment_seed"])
                            if isinstance(payload.get("misalignment_seed"), int)
                            else None
                        ),
                        failure_type=failure_type,
                        source_log=log_path,
                        source_line=line_num,
                    )
                )
    return tasks, warnings


def _policy_groups(tasks: list[ReplayTask]) -> dict[PolicySignature, list[ReplayTask]]:
    by_sig: dict[PolicySignature, list[ReplayTask]] = defaultdict(list)
    for task in tasks:
        sig = PolicySignature(
            variant_id=task.variant_id,
            profile=task.profile,
            min_seconds=task.misalign_min_seconds,
            max_seconds=task.misalign_max_seconds,
            seed=task.misalign_seed,
        )
        by_sig[sig].append(task)
    return by_sig


def _build_policy(sig: PolicySignature, tasks: list[ReplayTask]) -> ReplayMisalignmentPolicy:
    mapping = {
        (str(task.video_path.resolve()), task.segment_index): task.replay_offset_seconds
        for task in tasks
    }
    return ReplayMisalignmentPolicy(
        profile=sig.profile,
        min_seconds=sig.min_seconds,
        max_seconds=sig.max_seconds,
        seed=sig.seed,
        offset_by_video_segment=mapping,
    )


def _build_config(transcriber) -> Configuration:
    args = Namespace(
        no_transcription_cache=False,
        transcriber=transcriber,
    )
    return Configuration(args=args, stored=ConfigParser(interpolation=None))


def _load_series(dataset_dir: Path) -> Series:
    meta_path = dataset_dir / "meta.json"
    segment_duration = 30
    random_seed = 12345
    series_name = dataset_dir.name
    source_series_dir = str(dataset_dir)
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        segment_duration = int(meta.get("segment_duration") or segment_duration)
        random_seed = int(meta.get("random_seed") or random_seed)
        series_name = str(meta.get("series_name") or series_name)
        source_series_dir = str(meta.get("source_series_dir") or source_series_dir)
    return Series(
        dir=Path(source_series_dir),
        detail={},
        name=series_name,
        segment_duration=segment_duration,
        random_seed=random_seed,
    )


def _group_inputs(tasks: list[ReplayTask]) -> list[tuple[Path, list[int]]]:
    by_path: dict[Path, set[int]] = defaultdict(set)
    for task in tasks:
        by_path[task.video_path].add(task.segment_index)
    return [(path, sorted(indexes)) for path, indexes in by_path.items()]


def _run_single_mode(
    tasks: list[ReplayTask],
    config: Configuration,
    series: Series,
    transcriber,
    model_name: str,
    mode_dir: Path,
) -> dict[Path, Path]:
    outputs: dict[Path, Path] = {}
    by_sig = _policy_groups(tasks)
    for sig, sig_tasks in by_sig.items():
        policy = _build_policy(sig, sig_tasks)
        transcriber_impl = SegmentTranscriber(
            config=config,
            series=series,
            model_name=model_name,
            transcriber=transcriber,
            misalignment_policy=policy,
            variant_id=sig.variant_id,
            output_dir=mode_dir / "transcriptions",
            debug_log_path=mode_dir / "debug-events.jsonl",
        )
        outputs.update(transcriber_impl.execute(_group_inputs(sig_tasks)))
    return outputs


def _run_pool_mode(
    tasks: list[ReplayTask],
    config: Configuration,
    series: Series,
    transcriber,
    model_name: str,
    mode_dir: Path,
    workers: int,
) -> dict[Path, Path]:
    outputs: dict[Path, Path] = {}
    by_sig = _policy_groups(tasks)
    for sig, sig_tasks in by_sig.items():
        policy = _build_policy(sig, sig_tasks)
        initargs = (
            config,
            series,
            transcriber,
            model_name,
            policy,
            sig.variant_id,
            mode_dir / "transcriptions",
            mode_dir / "debug-events.jsonl",
        )

        inputs = _group_inputs(sig_tasks)
        if issubclass(transcriber, SubprocessTranscriber):
            executor_cls = ThreadPoolExecutor
            executor_kwargs = {"max_workers": workers}
        else:
            executor_cls = ProcessPoolExecutor
            executor_kwargs = {
                "max_workers": workers,
                "mp_context": multiprocessing.get_context("spawn"),
            }
        with executor_cls(
            initializer=_init_transcription_worker,
            initargs=initargs,
            **executor_kwargs,
        ) as executor:
            futures = (
                executor.submit(_extract_text_segments_worker, chunk)
                for chunk in _chunked(inputs, 3)
            )
            for future in as_completed(futures):
                outputs.update(future.result())
    return outputs


def _read_text_by_segment(transcript_path: Path) -> dict[int, str]:
    with transcript_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for key, value in raw.items():
        try:
            segment_index = int(key)
        except (TypeError, ValueError):
            continue
        result[segment_index] = value
    return result


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False))
            out.write("\n")


def _chunk_name_for_task(task: ReplayTask) -> str:
    start_ms = int(round(float(task.effective_offset_seconds) * 1000))
    duration = int(round(float(task.duration_seconds)))
    return unique_filename(task.video_path, f".{duration}S.AT{start_ms}ms.wav")


def _parse_chunks_from_replay_log(
    replay_log_path: Path,
    selected_error_types: set[str],
) -> set[str]:
    chunk_names: set[str] = set()
    if not replay_log_path.exists():
        raise FileNotFoundError(f"Replay log not found: {replay_log_path}")

    for line in replay_log_path.open("r", encoding="utf-8", errors="ignore"):
        error_type = None
        if (
            "parakeet-mlx failed for " in line
            and "metal::malloc" in line
        ):
            error_type = "metal_malloc"
        elif (
            "parakeet-mlx failed for " in line
            and "Failed to load audio" in line
        ):
            error_type = "load_audio_invalid"
        elif "parakeet-mlx returned empty text for " in line:
            error_type = "empty_text"

        if error_type not in selected_error_types:
            continue

        marker = " for "
        if marker not in line:
            continue
        path_start = line.split(marker, 1)[1]
        wav_idx = path_start.find(".wav")
        if wav_idx == -1:
            continue
        wav_path = path_start[: wav_idx + 4]
        chunk_names.add(Path(wav_path).name)
    return chunk_names


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset_dir / "debug" / "replay-failures" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = set(args.profiles) if args.profiles else None
    failure_types = set(args.failure_types)
    tasks, warnings = _build_tasks(
        dataset_dir=dataset_dir,
        logs_glob=args.logs_glob,
        profiles=profiles,
        failure_types=failure_types,
    )
    missing_video_count = 0
    filtered_tasks = []
    for task in tasks:
        if not task.video_path.exists():
            missing_video_count += 1
            continue
        filtered_tasks.append(task)
    tasks = filtered_tasks
    if args.replay_log_error_types:
        if not args.replay_log:
            raise SystemExit(
                "--replay-log-error-types requires --replay-log"
            )
        selected_error_types = set(args.replay_log_error_types)
        chunk_names = _parse_chunks_from_replay_log(
            replay_log_path=args.replay_log.expanduser().resolve(),
            selected_error_types=selected_error_types,
        )
        task_by_chunk = { _chunk_name_for_task(task): task for task in tasks }
        tasks = [task_by_chunk[name] for name in sorted(chunk_names) if name in task_by_chunk]

    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]

    print(f"dataset: {dataset_dir}")
    print(f"output:  {output_dir}")
    print(f"tasks:   {len(tasks)}")
    print(f"missing videos skipped: {missing_video_count}")
    if args.replay_log_error_types:
        print(f"replay_log: {args.replay_log}")
        print(f"replay_log_error_types: {sorted(args.replay_log_error_types)}")
    if warnings:
        print(f"parse warnings: {len(warnings)}")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  - ... {len(warnings) - 10} more")

    if not tasks:
        print("No replay tasks found.")
        return 0

    transcriber = TRANSCRIBERS[args.transcriber]
    config = _build_config(transcriber)
    series = _load_series(dataset_dir)

    records: list[dict] = []
    summary = {"total_tasks": len(tasks), "modes": {}, "warnings_count": len(warnings)}
    mode_outputs: dict[str, dict[Path, Path]] = {}
    mode_order = []
    for mode in args.modes:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"replaying mode={mode} ...")
        if mode == "single":
            outputs = _run_single_mode(
                tasks=tasks,
                config=config,
                series=series,
                transcriber=transcriber,
                model_name=args.model_name,
                mode_dir=mode_dir,
            )
        elif mode == "mp1":
            outputs = _run_pool_mode(
                tasks=tasks,
                config=config,
                series=series,
                transcriber=transcriber,
                model_name=args.model_name,
                mode_dir=mode_dir,
                workers=1,
            )
        else:
            outputs = _run_pool_mode(
                tasks=tasks,
                config=config,
                series=series,
                transcriber=transcriber,
                model_name=args.model_name,
                mode_dir=mode_dir,
                workers=max(1, args.workers),
            )
        mode_outputs[mode] = outputs
        mode_order.append(mode)

        texts_by_path = {}
        success_count = 0
        for task in tasks:
            transcript_path = outputs.get(task.video_path)
            text = None
            if transcript_path and transcript_path.exists():
                cache_key = str(transcript_path)
                if cache_key not in texts_by_path:
                    texts_by_path[cache_key] = _read_text_by_segment(transcript_path)
                text = texts_by_path[cache_key].get(task.segment_index)

            success = bool(isinstance(text, str) and text.strip())
            if success:
                success_count += 1
            records.append(
                {
                    "mode": mode,
                    "video_path": str(task.video_path),
                    "segment_index": task.segment_index,
                    "profile": task.profile,
                    "variant_id": task.variant_id,
                    "failure_type": task.failure_type,
                    "effective_offset_seconds": task.effective_offset_seconds,
                    "duration_seconds": task.duration_seconds,
                    "success": success,
                    "text_length": len(text.strip()) if success else 0,
                    "transcript_path": str(transcript_path) if transcript_path else None,
                    "source_log": str(task.source_log),
                    "source_line": task.source_line,
                }
            )
        summary["modes"][mode] = {
            "success_count": success_count,
            "failure_count": len(tasks) - success_count,
            "success_rate": (success_count / len(tasks)) if tasks else 0.0,
        }

    by_task_mode: dict[tuple[str, int, str], dict[str, bool]] = defaultdict(dict)
    for record in records:
        task_key = (record["video_path"], int(record["segment_index"]), record["variant_id"])
        by_task_mode[task_key][record["mode"]] = bool(record["success"])

    disagreements = 0
    for outcome_by_mode in by_task_mode.values():
        outcomes = [outcome_by_mode.get(mode) for mode in mode_order]
        if None not in outcomes and len(set(outcomes)) > 1:
            disagreements += 1
    summary["cross_mode_disagreements"] = disagreements

    _write_jsonl(output_dir / "attempts.jsonl", records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2)

    print("done")
    print(f"attempts: {output_dir / 'attempts.jsonl'}")
    print(f"summary:  {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
