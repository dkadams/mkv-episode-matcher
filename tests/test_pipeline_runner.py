from argparse import Namespace
from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.pipeline_runner import PipelineRunner
from mkv_episode_matcher.pipeline_types import TranscriptionResultEvent
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcribers import WhispercppCliTranscriber


def _config(io_workers: int = 1, transcribe_workers: int = 1) -> Configuration:
    return Configuration(
        args=Namespace(io_workers=io_workers, transcribe_workers=transcribe_workers),
        stored=ConfigParser(),
    )


def _series(tmp_path: Path) -> Series:
    return Series(
        dir=tmp_path / "series",
        detail={"name": "series"},
        name="series",
        segment_duration=30,
        random_seed=12345,
    )


def test_pipeline_runner_sorts_extracts_and_preserves_segment_indexes(monkeypatch, tmp_path):
    extract_offsets: list[float] = []
    created_chunks: list[Path] = []

    def fake_extract(self, file_path, start_time, duration):  # noqa: ANN001
        extract_offsets.append(start_time)
        out = tmp_path / f"{file_path.stem}-{int(start_time)}-{duration}.wav"
        out.write_bytes(b"RIFF")
        created_chunks.append(out)
        return out

    def fake_worker(task):
        return TranscriptionResultEvent(
            video_path=task.video_path,
            output_path=task.output_path,
            segment_index=task.segment_index,
            text=f"text-{task.segment_index}",
            transcribe_seconds=0.25,
        )

    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner.AudioChunkExtractor.extract", fake_extract)
    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner._transcribe_segment_task_worker", fake_worker)
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner.PipelineRunner._make_transcribe_executor",
        lambda _self, workers: ThreadPoolExecutor(max_workers=workers),
    )

    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=WhispercppCliTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    video = tmp_path / "episode.mkv"
    result = runner.run({video: [3, 1]})
    outputs = runner.write_outputs(result)
    payload = outputs[video].read_text(encoding="utf-8")

    assert extract_offsets == [30.0, 90.0]
    assert '"1": "text-1"' in payload
    assert '"3": "text-3"' in payload
    metrics = result.metrics_by_output[outputs[video]]
    assert metrics.segments_attempted == 2
    assert metrics.segments_transcribed == 2
    assert metrics.deleted_chunk_count == 2
    assert metrics.delete_failures == 0
    assert all(not chunk.exists() for chunk in created_chunks)


def test_pipeline_runner_records_extract_failures(monkeypatch, tmp_path):
    def fake_extract(self, file_path, start_time, duration):  # noqa: ANN001
        if start_time <= 0:
            raise RuntimeError("extract failed")
        out = tmp_path / f"{file_path.stem}-{int(start_time)}-{duration}.wav"
        out.write_bytes(b"RIFF")
        return out

    def fake_worker(task):
        return TranscriptionResultEvent(
            video_path=task.video_path,
            output_path=task.output_path,
            segment_index=task.segment_index,
            text="ok",
            transcribe_seconds=0.1,
        )

    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner.AudioChunkExtractor.extract", fake_extract)
    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner._transcribe_segment_task_worker", fake_worker)
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner.PipelineRunner._make_transcribe_executor",
        lambda _self, workers: ThreadPoolExecutor(max_workers=workers),
    )

    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=WhispercppCliTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    video = tmp_path / "episode.mkv"
    result = runner.run({video: [0, 1]})
    outputs = runner.write_outputs(result)

    metrics = result.metrics_by_output[outputs[video]]
    assert metrics.segments_attempted == 2
    assert metrics.segments_transcribed == 1
    assert any(failure["failure_type"] == "audio_extract_exception" for failure in result.failures)
