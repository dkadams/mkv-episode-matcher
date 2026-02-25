from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path
from types import SimpleNamespace

from rich.progress import Progress

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.indexed_episode_matcher import IndexedEpisodeMatcher
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcribers import WhispercppTranscriber


def test_get_transcriptions_uses_pipeline_runner(monkeypatch, tmp_path):
    video = tmp_path / "episode.mkv"
    video.write_text("x", encoding="utf-8")

    args = Namespace(
        index_type=SimpleNamespace(reader_type=object),
        no_transcription_cache=True,
        segments_per_minute=0.5,
        transcriber=WhispercppTranscriber,
        transcribe_workers=2,
        io_workers=1,
    )
    config = Configuration(args=args, stored=ConfigParser())
    series = Series(
        dir=tmp_path / "series",
        detail={"name": "series"},
        name="series",
        segment_duration=30,
        random_seed=12345,
    )
    matcher = IndexedEpisodeMatcher(config, series)

    monkeypatch.setattr(
        "mkv_episode_matcher.indexed_episode_matcher.get_video_duration_seconds",
        lambda _path: 60.0,
    )

    seen = {}

    class DummyRunner:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def run(self, segments_to_transcribe):
            seen["segments"] = segments_to_transcribe
            return type("DummyResult", (), {"outputs": {video: series.transcription_file(video)}, "failures": []})()

        def write_outputs(self, run_result):
            output = series.transcription_file(video)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"0":"ok"}', encoding="utf-8")
            output.with_suffix(".metrics.json").write_text(
                '{"extract_seconds": 1.0, "transcribe_seconds": 2.0, "segments_attempted": 1, "segments_transcribed": 1}',
                encoding="utf-8",
            )
            return run_result.outputs

    monkeypatch.setattr("mkv_episode_matcher.indexed_episode_matcher.PipelineRunner", DummyRunner)

    with Progress(disable=True) as progress:
        _, transcriptions = matcher.get_transcriptions(progress, [video], no_transcription_cache=True)

    assert transcriptions[video] == series.transcription_file(video)
    assert seen["segments"][video]
