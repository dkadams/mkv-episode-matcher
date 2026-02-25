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
    _benchmark_group,
    _build_ephemeral_series,
    _get_segment_indexes,
    _transcribe_segments,
    benchmark_transcribers,
)
from mkv_episode_matcher.transcribers import (
    WhispercppTranscriber,
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
    assert args.transcribe_workers == 4
    assert args.io_workers == 2
    assert args.backend is None


def test_benchmark_parser_accepts_xscribe_workers_alias():
    parser = build_args_parser()
    args = parser.parse_args(
        ["benchmark-transcribers", "video.mkv", "--xscribe-workers", "7", "--io-workers", "3"]
    )
    assert args.transcribe_workers == 7
    assert args.io_workers == 3


def test_benchmark_parser_backend_filter():
    parser = build_args_parser()
    args = parser.parse_args(
        [
            "benchmark-transcribers",
            "video.mkv",
            "--backend",
            "whispercpp",
            "--backend",
            "parakeet-mlx",
        ]
    )
    assert args.backend == ["whispercpp", "parakeet-mlx"]


def test_benchmark_parser_accepts_parakeet_backend():
    parser = build_args_parser()
    args = parser.parse_args(
        [
            "benchmark-transcribers",
            "video.mkv",
            "--backend",
            "parakeet-mlx",
        ]
    )
    assert args.backend == ["parakeet-mlx"]


def test_benchmark_parser_rejects_removed_backend():
    parser = build_args_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "benchmark-transcribers",
                "video.mkv",
                "--backend",
                "whispercpp-cli",
            ]
        )


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
            "parakeet-mlx",
            "--backend",
            "whispercpp",
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
        if backend_name == "parakeet-mlx":
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
        ["benchmark-transcribers", str(tmp_path / "in"), "--backend", "whispercpp"]
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


def test_transcribe_segments_uses_pipeline_runner(monkeypatch, tmp_path):
    args = Namespace(segments_per_minute=0.5, transcribe_workers=2, io_workers=1)
    config = Configuration(args=args, stored=ConfigParser())
    series = Series(Path("/tmp/s"), {"name": "s"}, "s", 30, 1)
    video = tmp_path / "ep.mkv"

    seen = {}

    class DummyRunner:
        def __init__(self, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

        def run(self, segments_to_transcribe):
            seen["segments"] = segments_to_transcribe
            output = series.transcription_file(video)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"0":"ok"}', encoding="utf-8")
            output.with_suffix(".metrics.json").write_text(
                '{"extract_seconds": 1.0, "transcribe_seconds": 2.0}',
                encoding="utf-8",
            )
            return type(
                "DummyResult",
                (),
                {"outputs": {video: output}, "failures": []},
            )()

        def write_outputs(self, run_result):
            return run_result.outputs

    monkeypatch.setattr("mkv_episode_matcher.transcriber_benchmark.PipelineRunner", DummyRunner)
    outputs, errors = _transcribe_segments(
        config=config,
        series=series,
        transcriber_type=WhispercppTranscriber,
        segments_to_transcribe={video: [0, 1]},
    )
    assert errors == []
    assert video in outputs
    assert seen["segments"] == {video: [0, 1]}


def test_benchmark_group_aggregates_extract_and_transcribe_metrics(monkeypatch, tmp_path):
    parser = build_args_parser()
    args = parser.parse_args(["benchmark-transcribers", str(tmp_path / "in")])
    config = _config_from_args(args)

    video = tmp_path / "episode.mkv"
    video.write_text("x", encoding="utf-8")

    group = BenchmarkGroup(
        key="__adhoc__",
        files=[video],
        source_series=None,
        segment_duration=30,
        random_seed=12345,
    )

    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._collect_video_infos",
        lambda _files, _duration: {
            video: VideoInfo(
                full_path_str=str(video.resolve()),
                byte_count=video.stat().st_size,
                minutes=1.0,
                segments=2,
            )
        },
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._get_segment_indexes",
        lambda _cfg, _series, _infos: [0],
    )

    def fake_transcribe_segments(_cfg, series, _transcriber_type, segments_to_transcribe):
        outputs = {}
        for path in segments_to_transcribe:
            output = series.transcription_file(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"0": "hello"}', encoding="utf-8")
            output.with_suffix(".metrics.json").write_text(
                '{"extract_seconds": 3.5, "transcribe_seconds": 1.25}',
                encoding="utf-8",
            )
            outputs[path] = output
        return outputs, []

    monkeypatch.setattr(
        "mkv_episode_matcher.transcriber_benchmark._transcribe_segments",
        fake_transcribe_segments,
    )

    result = _benchmark_group(config, WhispercppTranscriber, group, tmp_path / "bench-root")
    assert result.files_succeeded == 1
    assert result.transcribed_segments == 1
    assert result.extract_seconds == 3.5
    assert result.transcribe_seconds == 1.25
