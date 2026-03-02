from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from loguru import logger
from rich.console import Console
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.indexed_episode_matcher import (
    IndexedEpisodeMatcher,
    VideoInfo,
)
from mkv_episode_matcher.pipeline_runner import PipelineRunner
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcribers import (
    FasterWhisperTranscriber,
    ParakeetMlxTranscriber,
    WhispercppTranscriber,
)
from mkv_episode_matcher.video_helper import get_video_duration_seconds

console = Console()

DEFAULT_SEGMENT_DURATION = 30
DEFAULT_RANDOM_SEED = 12345
DEFAULT_TEXT_EXTRACTOR_MODEL = "small.en"

BENCHMARK_BACKENDS: dict[str, type] = {
    "whispercpp": WhispercppTranscriber,
    "faster-whisper": FasterWhisperTranscriber,
    "parakeet-mlx": ParakeetMlxTranscriber,
}
BENCHMARK_BACKEND_CHOICES = tuple(BENCHMARK_BACKENDS.keys())


@dataclass(frozen=True)
class BenchmarkGroup:
    key: str
    files: list[Path]
    source_series: Series | None
    segment_duration: int
    random_seed: int


@dataclass
class BenchmarkResult:
    backend: str
    files_attempted: int = 0
    files_succeeded: int = 0
    sampled_segments: int = 0
    transcribed_segments: int = 0
    sampled_audio_minutes: float = 0.0
    wall_seconds: float = 0.0
    extract_seconds: float = 0.0
    transcribe_seconds: float = 0.0
    errors: list[str] = dataclasses.field(default_factory=list)

    @property
    def status(self) -> str:
        if self.files_attempted == 0:
            return "failed"
        if self.files_succeeded == self.files_attempted and not self.errors:
            return "ok"
        if self.files_succeeded > 0:
            return "partial"
        return "failed"

    @property
    def sec_per_media_min(self) -> float | None:
        if self.sampled_audio_minutes <= 0:
            return None
        return self.wall_seconds / self.sampled_audio_minutes

    @property
    def extract_sec_per_media_min(self) -> float | None:
        if self.sampled_audio_minutes <= 0:
            return None
        return self.extract_seconds / self.sampled_audio_minutes

    @property
    def transcribe_sec_per_media_min(self) -> float | None:
        if self.sampled_audio_minutes <= 0:
            return None
        return self.transcribe_seconds / self.sampled_audio_minutes

    def add_error(self, message: str):
        message = str(message).strip()
        if message and message not in self.errors:
            self.errors.append(message)


def benchmark_transcribers(config: Configuration) -> None:
    paths = [Path(path) for path in config.args.video_files]
    extensions = _normalize_extensions(config.args.extension)
    files = list(_collect_files(paths, extensions))
    if not files:
        console.print("[orange1]No input files found to benchmark.")
        raise SystemExit(1)

    groups = _group_files(files, config.args.segment_duration, config.args.random_seed)
    backend_names = tuple(config.args.backend or BENCHMARK_BACKEND_CHOICES)

    results: list[BenchmarkResult] = []
    for backend_name in backend_names:
        result = _run_backend(config, backend_name, groups)
        results.append(result)

    _display_results(results)
    if all(result.status == "failed" for result in results):
        raise SystemExit(1)


def _run_backend(
    config: Configuration, backend_name: str, groups: list[BenchmarkGroup]
) -> BenchmarkResult:
    transcriber_type = BENCHMARK_BACKENDS[backend_name]
    result = BenchmarkResult(backend=backend_name)

    started = time.perf_counter()
    for group in groups:
        with tempfile.TemporaryDirectory(prefix=f"benchmark-{backend_name}-") as tmpdir:
            try:
                group_result = _benchmark_group(
                    config,
                    transcriber_type,
                    group,
                    Path(tmpdir),
                )
            except Exception as exc:  # noqa: BLE001
                result.files_attempted += len(group.files)
                result.add_error(f"{group.key}: {exc}")
                continue

            result.files_attempted += group_result.files_attempted
            result.files_succeeded += group_result.files_succeeded
            result.sampled_segments += group_result.sampled_segments
            result.transcribed_segments += group_result.transcribed_segments
            result.sampled_audio_minutes += group_result.sampled_audio_minutes
            result.extract_seconds += group_result.extract_seconds
            result.transcribe_seconds += group_result.transcribe_seconds
            for error in group_result.errors:
                result.add_error(f"{group.key}: {error}")

    result.wall_seconds = time.perf_counter() - started
    return result


def _benchmark_group(
    config: Configuration,
    transcriber_type: type,
    group: BenchmarkGroup,
    temp_root: Path,
) -> BenchmarkResult:
    bench_series = _build_ephemeral_series(group, temp_root)
    benchmark_args = _build_benchmark_args(config, transcriber_type)
    benchmark_config = Configuration(args=benchmark_args, stored=config.stored)

    file_infos = _collect_video_infos(group.files, bench_series.segment_duration)
    segment_indexes = _get_segment_indexes(benchmark_config, bench_series, file_infos.values())

    segments_per_minute = benchmark_config.args.segments_per_minute
    segments_to_transcribe: dict[Path, list[int]] = {}
    sampled_segments = 0
    sampled_audio_minutes = 0.0
    for path, video_info in file_infos.items():
        segment_target = math.ceil(video_info.minutes * segments_per_minute)
        segment_selection = segment_indexes[:segment_target]
        sampled_segments += len(segment_selection)
        sampled_audio_minutes += (len(segment_selection) * bench_series.segment_duration) / 60.0
        if segment_selection:
            segments_to_transcribe[path] = segment_selection

    transcribed: dict[Path, Path] = {}
    errors: list[str] = []
    if segments_to_transcribe:
        transcribed, errors = _transcribe_segments(
            benchmark_config,
            bench_series,
            transcriber_type,
            segments_to_transcribe,
        )

    files_succeeded = 0
    transcribed_segments = 0
    extract_seconds = 0.0
    transcribe_seconds = 0.0
    for path in group.files:
        selected = segments_to_transcribe.get(path, [])
        if not selected:
            files_succeeded += 1
            continue

        transcript_path = transcribed.get(path)
        if not transcript_path or not transcript_path.exists():
            errors.append(f"{path}: missing transcript output")
            continue

        try:
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: invalid transcript JSON ({exc})")
            continue

        if not isinstance(payload, dict):
            errors.append(f"{path}: transcript output is not a dictionary")
            continue

        chunk_count = len(payload)
        transcribed_segments += chunk_count
        metrics_path = transcript_path.with_suffix(".metrics.json")
        if metrics_path.exists():
            try:
                metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                extract_seconds += float(metrics_payload.get("extract_seconds", 0.0))
                transcribe_seconds += float(metrics_payload.get("transcribe_seconds", 0.0))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: invalid metrics JSON ({exc})")
        if chunk_count > 0:
            files_succeeded += 1
        else:
            errors.append(f"{path}: no chunks transcribed")

    result = BenchmarkResult(
        backend="",
        files_attempted=len(group.files),
        files_succeeded=files_succeeded,
        sampled_segments=sampled_segments,
        transcribed_segments=transcribed_segments,
        sampled_audio_minutes=sampled_audio_minutes,
        extract_seconds=extract_seconds,
        transcribe_seconds=transcribe_seconds,
    )
    for error in errors:
        result.add_error(error)
    return result


def _transcribe_segments(
    config: Configuration,
    series: Series,
    transcriber_type: type,
    segments_to_transcribe: dict[Path, list[int]],
) -> tuple[dict[Path, Path], list[str]]:
    if not segments_to_transcribe:
        return {}, []

    runner = PipelineRunner(
        config=config,
        series=series,
        transcriber_type=transcriber_type,
        model_name=DEFAULT_TEXT_EXTRACTOR_MODEL,
        output_dir=series.ensure_transcription_text_dir(),
    )
    pipeline_result = runner.run(segments_to_transcribe)
    outputs = runner.write_outputs(pipeline_result)
    errors: list[str] = []
    for failure in pipeline_result.failures:
        video_path = failure.get("video_path")
        failure_type = failure.get("failure_type", "unknown_failure")
        if video_path:
            errors.append(f"{video_path}: {failure_type}")
        else:
            errors.append(failure_type)
    logger.debug(f"Completed transcription run for {transcriber_type.__name__}")
    return outputs, errors


def _build_benchmark_args(config: Configuration, transcriber_type: type) -> Namespace:
    values = vars(config.args).copy()
    values["transcriber"] = transcriber_type
    values["no_transcription_cache"] = True
    return Namespace(**values)


def _collect_video_infos(files: list[Path], segment_duration: int) -> dict[Path, VideoInfo]:
    video_infos: dict[Path, VideoInfo] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_video_info, path, segment_duration): path for path in files}
        for future in as_completed(futures):
            path, info = future.result()
            video_infos[path] = info
    return video_infos


def _video_info(path: Path, segment_duration: int) -> tuple[Path, VideoInfo]:
    resolved = path.resolve()
    seconds = get_video_duration_seconds(resolved)
    minutes = seconds / 60.0
    segments = math.ceil(seconds / segment_duration)
    return path, VideoInfo(
        byte_count=resolved.stat().st_size,
        minutes=minutes,
        segments=segments,
    )


def _get_segment_indexes(
    config: Configuration, series: Series, video_infos: Iterable[VideoInfo]
) -> list[int]:
    pseudo_matcher = SimpleNamespace(config=config, series=series)
    return IndexedEpisodeMatcher.get_segment_selection(pseudo_matcher, video_infos)


def _build_ephemeral_series(group: BenchmarkGroup, temp_root: Path) -> Series:
    series_name = group.source_series.name if group.source_series else "adhoc"
    detail = group.source_series.detail if group.source_series else {"name": "adhoc"}
    series_dir = temp_root / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    return Series(
        dir=series_dir,
        detail=detail,
        name=series_name,
        segment_duration=group.segment_duration,
        random_seed=group.random_seed,
    )


def _group_files(
    files: list[Path], segment_duration_override: int | None, random_seed_override: int | None
) -> list[BenchmarkGroup]:
    grouped: dict[str, list[Path]] = {}
    series_by_key: dict[str, Series | None] = {}
    for file in files:
        source_series = _detect_series(file)
        key = str(source_series.dir) if source_series else "__adhoc__"
        grouped.setdefault(key, []).append(file)
        series_by_key[key] = source_series

    groups: list[BenchmarkGroup] = []
    for key, grouped_files in grouped.items():
        source_series = series_by_key[key]
        segment_duration = segment_duration_override
        if segment_duration is None:
            segment_duration = (
                source_series.segment_duration if source_series else DEFAULT_SEGMENT_DURATION
            )
        random_seed = random_seed_override
        if random_seed is None:
            random_seed = source_series.random_seed if source_series else DEFAULT_RANDOM_SEED

        groups.append(
            BenchmarkGroup(
                key=key,
                files=sorted(grouped_files),
                source_series=source_series,
                segment_duration=segment_duration,
                random_seed=random_seed,
            )
        )
    return groups


def _detect_series(path: Path) -> Series | None:
    search_dirs = [path] if path.is_dir() else []
    search_dirs.extend(path.parents)
    for directory in search_dirs:
        if (directory / ".mkv-episode-matcher").is_dir():
            return Series.from_dir(directory)
    return None


def _normalize_extensions(extensions: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(normalized or [".mkv"])


def _collect_files(paths: Iterable[Path], extensions: tuple[str, ...]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            if path.suffix.lower() in extensions:
                yield path
            continue

        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in extensions:
                yield candidate


def _display_results(results: list[BenchmarkResult]):
    table = Table(title="Transcriber Benchmark")
    table.add_column("Backend")
    table.add_column("Status")
    table.add_column("Wall (s)", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Sampled Segments", justify="right")
    table.add_column("Transcribed Segments", justify="right")
    table.add_column("Sampled Audio (min)", justify="right")
    table.add_column("Sec / Media Min", justify="right")
    table.add_column("Extract (s)", justify="right")
    table.add_column("Transcribe (s)", justify="right")
    table.add_column("Extract / Media Min", justify="right")
    table.add_column("Transcribe / Media Min", justify="right")
    table.add_column("Errors")

    for result in results:
        sec_per_media = result.sec_per_media_min
        extract_per_media = result.extract_sec_per_media_min
        transcribe_per_media = result.transcribe_sec_per_media_min
        errors = "; ".join(result.errors[:3])
        if len(result.errors) > 3:
            errors = f"{errors}; +{len(result.errors) - 3} more"
        table.add_row(
            result.backend,
            result.status,
            f"{result.wall_seconds:.2f}",
            f"{result.files_succeeded}/{result.files_attempted}",
            str(result.sampled_segments),
            str(result.transcribed_segments),
            f"{result.sampled_audio_minutes:.2f}",
            "-" if sec_per_media is None else f"{sec_per_media:.2f}",
            f"{result.extract_seconds:.2f}",
            f"{result.transcribe_seconds:.2f}",
            "-" if extract_per_media is None else f"{extract_per_media:.2f}",
            "-" if transcribe_per_media is None else f"{transcribe_per_media:.2f}",
            errors,
        )
    console.print(table)
