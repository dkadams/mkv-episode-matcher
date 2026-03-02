from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path
from types import SimpleNamespace

import mkv_episode_matcher.transcription_worker as worker
from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.pipeline_types import ChunkTask
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


def _chunk_task(index: int) -> ChunkTask:
    chunk_path = Path(f"/tmp/chunk-{index}.wav")
    return ChunkTask(
        kind="chunk_path",
        video_path=Path("/tmp/video.mkv"),
        output_path=Path("/tmp/out.json"),
        segment_index=index,
        chunk_path=chunk_path,
        base_offset_seconds=index * 30.0,
        effective_offset_seconds=index * 30.0,
        duration_seconds=30,
    )


def test_batch_worker_returns_events_on_success(monkeypatch):
    class DummyTranscriber:
        def transcribe_many(self, paths):
            return [f"text-{Path(path).stem}" for path in paths]

    monkeypatch.setattr(
        worker,
        "_PROCESS_TEXT_EXTRACTOR",
        SimpleNamespace(transcriber=DummyTranscriber()),
    )

    tasks = [_chunk_task(0), _chunk_task(1)]
    events = worker._transcribe_segment_batch_task_worker(tasks)

    assert [event.segment_index for event in events] == [0, 1]
    assert [event.text for event in events] == ["text-chunk-0", "text-chunk-1"]
    assert all(event.failure is None for event in events)
    assert all(event.transcribe_seconds > 0 for event in events)


def test_batch_worker_falls_back_when_result_count_mismatch(monkeypatch):
    calls = {"single": 0}

    class DummyTranscriber:
        def transcribe_many(self, _paths):
            return ["only-one"]

        def transcribe(self, path):
            calls["single"] += 1
            return f"single-{Path(path).stem}"

    monkeypatch.setattr(
        worker,
        "_PROCESS_TEXT_EXTRACTOR",
        SimpleNamespace(transcriber=DummyTranscriber()),
    )

    tasks = [_chunk_task(0), _chunk_task(1)]
    events = worker._transcribe_segment_batch_task_worker(tasks)

    assert calls["single"] == 2
    assert [event.text for event in events] == ["single-chunk-0", "single-chunk-1"]
    assert all(event.failure is None for event in events)


def test_batch_worker_falls_back_when_batch_raises(monkeypatch):
    calls = {"single": 0}

    class DummyTranscriber:
        def transcribe_many(self, _paths):
            raise RuntimeError("batch boom")

        def transcribe(self, path):
            calls["single"] += 1
            if Path(path).stem.endswith("1"):
                raise RuntimeError("single boom")
            return "ok"

    monkeypatch.setattr(
        worker,
        "_PROCESS_TEXT_EXTRACTOR",
        SimpleNamespace(transcriber=DummyTranscriber()),
    )

    tasks = [_chunk_task(0), _chunk_task(1)]
    events = worker._transcribe_segment_batch_task_worker(tasks)

    assert calls["single"] == 2
    assert events[0].text == "ok"
    assert events[0].failure is None
    assert events[1].text is None
    assert events[1].failure["failure_type"] == "transcribe_exception"
