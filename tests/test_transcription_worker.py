from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path

import mkv_episode_matcher.transcription_worker as worker
from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.series import Series


def _config() -> Configuration:
    return Configuration(args=Namespace(), stored=ConfigParser())


def _series() -> Series:
    return Series(
        dir=Path("/tmp/series"),
        detail={"name": "series"},
        name="series",
        segment_duration=30,
        random_seed=12345,
    )


def test_init_transcription_worker_reinitializes_each_time(monkeypatch):
    created = []

    class DummySegmentTranscriber:
        def __init__(self, config, series, model_name, transcriber, **kwargs):
            created.append((model_name, transcriber))

        def execute(self, inputs):
            return {}

    class FirstTranscriber:
        pass

    class SecondTranscriber:
        pass

    monkeypatch.setattr(worker, "SegmentTranscriber", DummySegmentTranscriber)
    monkeypatch.setattr(worker, "_PROCESS_TEXT_EXTRACTOR", None)

    worker._init_transcription_worker(_config(), _series(), FirstTranscriber, "m1")
    worker._init_transcription_worker(_config(), _series(), SecondTranscriber, "m2")

    assert created == [("m1", FirstTranscriber), ("m2", SecondTranscriber)]
