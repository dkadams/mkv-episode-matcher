from argparse import Namespace
from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.pipeline_runner import PipelineRunner
from mkv_episode_matcher.pipeline_types import TranscriptionResultEvent
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcribers import WhispercppTranscriber


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

    def fake_batch_worker(tasks):
        return [
            TranscriptionResultEvent(
                video_path=task.video_path,
                output_path=task.output_path,
                segment_index=task.segment_index,
                text=f"text-{task.segment_index}",
                transcribe_seconds=0.25,
            )
            for task in tasks
        ]

    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner.AudioChunkExtractor.extract", fake_extract)
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner._transcribe_segment_batch_task_worker",
        fake_batch_worker,
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner.PipelineRunner._make_transcribe_executor",
        lambda _self, workers: ThreadPoolExecutor(max_workers=workers),
    )

    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=WhispercppTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    video = tmp_path / "episode.mkv"
    video.write_bytes(b"dummy")
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

    def fake_batch_worker(tasks):
        return [
            TranscriptionResultEvent(
                video_path=task.video_path,
                output_path=task.output_path,
                segment_index=task.segment_index,
                text="ok",
                transcribe_seconds=0.1,
            )
            for task in tasks
        ]

    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner.AudioChunkExtractor.extract", fake_extract)
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner._transcribe_segment_batch_task_worker",
        fake_batch_worker,
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner.PipelineRunner._make_transcribe_executor",
        lambda _self, workers: ThreadPoolExecutor(max_workers=workers),
    )

    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=WhispercppTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    video = tmp_path / "episode.mkv"
    video.write_bytes(b"dummy")
    result = runner.run({video: [0, 1]})
    outputs = runner.write_outputs(result)

    metrics = result.metrics_by_output[outputs[video]]
    assert metrics.segments_attempted == 2
    assert metrics.segments_transcribed == 1
    assert any(failure["failure_type"] == "audio_extract_exception" for failure in result.failures)


def test_pipeline_runner_batches_across_files(monkeypatch, tmp_path):
    class BatchTranscriber:
        DEFAULT_MICROBATCH_SIZE = 2

    seen_batches: list[list[Path]] = []

    def fake_extract(self, file_path, start_time, duration):  # noqa: ANN001
        out = tmp_path / f"{file_path.stem}-{int(start_time)}-{duration}.wav"
        out.write_bytes(b"RIFF")
        return out

    def fake_batch_worker(tasks):
        seen_batches.append([task.video_path for task in tasks])
        return [
            TranscriptionResultEvent(
                video_path=task.video_path,
                output_path=task.output_path,
                segment_index=task.segment_index,
                text=f"text-{task.segment_index}",
                transcribe_seconds=0.1,
            )
            for task in tasks
        ]

    monkeypatch.delenv("MEM_TRANSCRIBE_MICROBATCH_SIZE", raising=False)
    monkeypatch.delenv("MEM_TRANSCRIBE_MICROBATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner.AudioChunkExtractor.extract", fake_extract)
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner._transcribe_segment_batch_task_worker",
        fake_batch_worker,
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner.PipelineRunner._make_transcribe_executor",
        lambda _self, workers: ThreadPoolExecutor(max_workers=workers),
    )

    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=BatchTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    video1 = tmp_path / "episode1.mkv"
    video2 = tmp_path / "episode2.mkv"
    video1.write_bytes(b"dummy")
    video2.write_bytes(b"dummy")

    result = runner.run({video1: [0], video2: [0]})
    runner.write_outputs(result)
    total_transcribed = sum(
        metrics.segments_transcribed
        for metrics in result.metrics_by_output.values()
    )
    assert total_transcribed == 2
    assert any(len(batch) == 2 for batch in seen_batches)
    assert any(set(batch) == {video1, video2} for batch in seen_batches)


def test_pipeline_runner_uses_backend_default_microbatch_size(monkeypatch, tmp_path):
    class BatchTranscriber:
        DEFAULT_MICROBATCH_SIZE = 3

    batch_sizes: list[int] = []

    def fake_extract(self, file_path, start_time, duration):  # noqa: ANN001
        out = tmp_path / f"{file_path.stem}-{int(start_time)}-{duration}.wav"
        out.write_bytes(b"RIFF")
        return out

    def fake_batch_worker(tasks):
        batch_sizes.append(len(tasks))
        return [
            TranscriptionResultEvent(
                video_path=task.video_path,
                output_path=task.output_path,
                segment_index=task.segment_index,
                text=f"text-{task.segment_index}",
                transcribe_seconds=0.1,
            )
            for task in tasks
        ]

    monkeypatch.delenv("MEM_TRANSCRIBE_MICROBATCH_SIZE", raising=False)
    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner.AudioChunkExtractor.extract", fake_extract)
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner._transcribe_segment_batch_task_worker",
        fake_batch_worker,
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner.PipelineRunner._make_transcribe_executor",
        lambda _self, workers: ThreadPoolExecutor(max_workers=workers),
    )

    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=BatchTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    video = tmp_path / "episode.mkv"
    video.write_bytes(b"dummy")
    runner.run({video: [0, 1, 2, 3]})
    assert batch_sizes == [3, 1]


def test_pipeline_runner_flushes_partial_batch_on_sentinel(monkeypatch, tmp_path):
    class BatchTranscriber:
        DEFAULT_MICROBATCH_SIZE = 3

    batch_sizes: list[int] = []

    def fake_extract(self, file_path, start_time, duration):  # noqa: ANN001
        out = tmp_path / f"{file_path.stem}-{int(start_time)}-{duration}.wav"
        out.write_bytes(b"RIFF")
        return out

    def fake_batch_worker(tasks):
        batch_sizes.append(len(tasks))
        return [
            TranscriptionResultEvent(
                video_path=task.video_path,
                output_path=task.output_path,
                segment_index=task.segment_index,
                text="ok",
                transcribe_seconds=0.1,
            )
            for task in tasks
        ]

    monkeypatch.delenv("MEM_TRANSCRIBE_MICROBATCH_SIZE", raising=False)
    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner.AudioChunkExtractor.extract", fake_extract)
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner._transcribe_segment_batch_task_worker",
        fake_batch_worker,
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner.PipelineRunner._make_transcribe_executor",
        lambda _self, workers: ThreadPoolExecutor(max_workers=workers),
    )

    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=BatchTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    video = tmp_path / "episode.mkv"
    video.write_bytes(b"dummy")
    runner.run({video: [0, 1]})
    assert batch_sizes == [2]


def test_pipeline_runner_flushes_on_batch_wait_timeout(monkeypatch, tmp_path):
    class BatchTranscriber:
        DEFAULT_MICROBATCH_SIZE = 2

    batch_sizes: list[int] = []

    def fake_extract(self, file_path, start_time, duration):  # noqa: ANN001
        if start_time > 0:
            time.sleep(0.03)
        out = tmp_path / f"{file_path.stem}-{int(start_time)}-{duration}.wav"
        out.write_bytes(b"RIFF")
        return out

    def fake_batch_worker(tasks):
        batch_sizes.append(len(tasks))
        return [
            TranscriptionResultEvent(
                video_path=task.video_path,
                output_path=task.output_path,
                segment_index=task.segment_index,
                text="ok",
                transcribe_seconds=0.1,
            )
            for task in tasks
        ]

    monkeypatch.setenv("MEM_TRANSCRIBE_MICROBATCH_MAX_WAIT_MS", "5")
    monkeypatch.delenv("MEM_TRANSCRIBE_MICROBATCH_SIZE", raising=False)
    monkeypatch.setattr("mkv_episode_matcher.pipeline_runner.AudioChunkExtractor.extract", fake_extract)
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner._transcribe_segment_batch_task_worker",
        fake_batch_worker,
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.pipeline_runner.PipelineRunner._make_transcribe_executor",
        lambda _self, workers: ThreadPoolExecutor(max_workers=workers),
    )

    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=BatchTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    video = tmp_path / "episode.mkv"
    video.write_bytes(b"dummy")
    runner.run({video: [0, 1]})
    assert batch_sizes == [1, 1]


def test_pipeline_runner_uses_whispercpp_default_microbatch_size(monkeypatch, tmp_path):
    monkeypatch.delenv("MEM_TRANSCRIBE_MICROBATCH_SIZE", raising=False)
    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=WhispercppTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    assert runner._microbatch_size() == 8


def test_pipeline_runner_allows_microbatch_override_for_whispercpp(monkeypatch, tmp_path):
    monkeypatch.setenv("MEM_TRANSCRIBE_MICROBATCH_SIZE", "1")
    series = _series(tmp_path)
    runner = PipelineRunner(
        config=_config(),
        series=series,
        transcriber_type=WhispercppTranscriber,
        model_name="unused",
        output_dir=series.ensure_transcription_text_dir(),
    )
    assert runner._microbatch_size() == 1
