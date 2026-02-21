from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path

import pytest

from mkv_episode_matcher.args import build_args_parser
from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.indexed_episode_matcher import VideoInfo
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcriber_benchmark import (
    BENCHMARK_BACKEND_CHOICES,
    BenchmarkGroup,
    BenchmarkResult,
    _build_ephemeral_series,
    _get_segment_indexes,
    _make_executor,
    benchmark_transcribers,
)
from mkv_episode_matcher.transcribers import (
    SubprocessTranscriber,
    WhisperTranscriber,
    WhispercppCliTranscriber,
)


def _config_from_args(args: Namespace) -> Configuration:
    return Configuration(args=args, stored=ConfigParser())


def test_benchmark_parser_defaults():
    parser = build_args_parser()
    args = parser.parse_args(["benchmark-transcribers", "video.mkv"])

    assert args.func is benchmark_transcribers
    assert args.video_files == ["video.mkv"]
    assert args.extension == [".mkv"]
    assert args.segments_per_minute == 0.5
    assert args.segment_duration is None
    assert args.random_seed is None
    assert args.thread_workers == 10
    assert args.process_workers == 8
    assert args.backend is None


def test_benchmark_parser_backend_filter():
    parser = build_args_parser()
    args = parser.parse_args(
        [
            "benchmark-transcribers",
            "video.mkv",
            "--backend",
            "whispercpp-cli",
            "--backend",
            "parakeet-mlx",
        ]
    )
    assert args.backend == ["whispercpp-cli", "parakeet-mlx"]


def test_benchmark_parser_accepts_parakeet_batch_backend():
    parser = build_args_parser()
    args = parser.parse_args(
        [
            "benchmark-transcribers",
            "video.mkv",
            "--backend",
            "parakeet-mlx-batch",
        ]
    )
    assert args.backend == ["parakeet-mlx-batch"]


def test_benchmark_uses_all_backends_by_default(monkeypatch, tmp_path):
    parser = build_args_parser()
    args = parser.parse_args(["benchmark-transcribers", str(tmp_path / "in")])
    config = _config_from_args(args)

    video = tmp_path / "one.mkv"
    video.write_text("x", encoding="utf-8")

    group = BenchmarkGroup(
        key="__adhoc__",
        files=[video],
        source_series=None,
        segment_duration=30,
        random_seed=12345,
    )

    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._collect_files",
        lambda _paths, _ext: [video],
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._group_files",
        lambda _files, _segment_duration, _random_seed: [group],
    )

    seen = []

    def fake_run_backend(_config, backend_name, _groups):
        seen.append(backend_name)
        return BenchmarkResult(backend=backend_name, files_attempted=1, files_succeeded=1)

    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark._run_backend", fake_run_backend)
    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark._display_results", lambda _results: None)

    benchmark_transcribers(config)
    assert tuple(seen) == BENCHMARK_BACKEND_CHOICES


def test_benchmark_continues_after_backend_failure(monkeypatch, tmp_path):
    parser = build_args_parser()
    args = parser.parse_args(
        [
            "benchmark-transcribers",
            str(tmp_path / "in"),
            "--backend",
            "whisper",
            "--backend",
            "whispercpp-cli",
        ]
    )
    config = _config_from_args(args)

    video = tmp_path / "one.mkv"
    video.write_text("x", encoding="utf-8")

    group = BenchmarkGroup(
        key="__adhoc__",
        files=[video],
        source_series=None,
        segment_duration=30,
        random_seed=12345,
    )

    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._collect_files",
        lambda _paths, _ext: [video],
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._group_files",
        lambda _files, _segment_duration, _random_seed: [group],
    )
    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark._display_results", lambda _results: None)

    def fake_run_backend(_config, backend_name, _groups):
        if backend_name == "whisper":
            return BenchmarkResult(
                backend=backend_name,
                files_attempted=1,
                files_succeeded=0,
                errors=["boom"],
            )
        return BenchmarkResult(backend=backend_name, files_attempted=1, files_succeeded=1)

    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark._run_backend", fake_run_backend)

    benchmark_transcribers(config)


def test_benchmark_raises_when_all_backends_fail(monkeypatch, tmp_path):
    parser = build_args_parser()
    args = parser.parse_args(
        ["benchmark-transcribers", str(tmp_path / "in"), "--backend", "whisper"]
    )
    config = _config_from_args(args)

    video = tmp_path / "one.mkv"
    video.write_text("x", encoding="utf-8")

    group = BenchmarkGroup(
        key="__adhoc__",
        files=[video],
        source_series=None,
        segment_duration=30,
        random_seed=12345,
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._collect_files",
        lambda _paths, _ext: [video],
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._group_files",
        lambda _files, _segment_duration, _random_seed: [group],
    )
    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark._display_results", lambda _results: None)
    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._run_backend",
        lambda _config, backend_name, _groups: BenchmarkResult(
            backend=backend_name, files_attempted=1, files_succeeded=0, errors=["boom"]
        ),
    )

    with pytest.raises(SystemExit):
        benchmark_transcribers(config)


def test_ephemeral_series_uses_temp_dir(tmp_path):
    source = Series(
        dir=tmp_path / "source-series",
        detail={"name": "Source"},
        name="Source",
        segment_duration=22,
        random_seed=7,
    )
    group = BenchmarkGroup(
        key=str(source.dir),
        files=[],
        source_series=source,
        segment_duration=22,
        random_seed=7,
    )
    temp_root = tmp_path / "temp-root"
    built = _build_ephemeral_series(group, temp_root)
    assert built.dir == temp_root / "series"
    assert built.dir != source.dir


def test_segment_selection_is_deterministic():
    args = Namespace(segments_per_minute=0.5)
    config = Configuration(args=args, stored=ConfigParser())
    series = Series(
        dir=Path("/tmp/nonexistent"),
        detail={"name": "x"},
        name="x",
        segment_duration=30,
        random_seed=12345,
    )
    infos = [
        VideoInfo(full_path_str="/a", byte_count=1, minutes=22.0, segments=44),
        VideoInfo(full_path_str="/b", byte_count=1, minutes=44.0, segments=88),
    ]

    first = _get_segment_indexes(config, series, infos)
    second = _get_segment_indexes(config, series, infos)
    assert first == second


def test_make_executor_uses_threads_for_subprocess(monkeypatch):
    seen = {}

    class DummyThreadExecutor:
        def __init__(self, **kwargs):
            seen["thread"] = kwargs

    class DummyProcessExecutor:
        def __init__(self, **kwargs):
            seen["process"] = kwargs

    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark.ThreadPoolExecutor", DummyThreadExecutor)
    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark.ProcessPoolExecutor", DummyProcessExecutor)

    args = Namespace(segments_per_minute=0.5, thread_workers=1, process_workers=1)
    config = Configuration(args=args, stored=ConfigParser())
    series = Series(Path("/tmp/s"), {"name": "s"}, "s", 30, 1)

    _make_executor(WhispercppCliTranscriber, 3, 2, config, series)
    assert "thread" in seen
    assert "process" not in seen


def test_make_executor_uses_process_for_python_transcriber(monkeypatch):
    seen = {}

    class DummyThreadExecutor:
        def __init__(self, **kwargs):
            seen["thread"] = kwargs

    class DummyProcessExecutor:
        def __init__(self, **kwargs):
            seen["process"] = kwargs

    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark.ThreadPoolExecutor", DummyThreadExecutor)
    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark.ProcessPoolExecutor", DummyProcessExecutor)
    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark.multiprocessing.get_context", lambda _name: "ctx")

    args = Namespace(segments_per_minute=0.5, thread_workers=1, process_workers=1)
    config = Configuration(args=args, stored=ConfigParser())
    series = Series(Path("/tmp/s"), {"name": "s"}, "s", 30, 1)

    class NonSubprocessTranscriber(WhisperTranscriber):
        pass

    assert not issubclass(NonSubprocessTranscriber, SubprocessTranscriber)
    _make_executor(NonSubprocessTranscriber, 3, 2, config, series)
    assert "process" in seen
    assert "thread" not in seen
