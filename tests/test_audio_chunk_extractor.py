import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor
from mkv_episode_matcher.utils import unique_filename


def test_extract_uses_millisecond_start_in_chunk_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mkv_episode_matcher.audio_chunk_extractor.tempfile.gettempdir",
        lambda: str(tmp_path),
    )

    ffmpeg_outputs = []

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        output_path = Path(cmd[-1])
        ffmpeg_outputs.append(output_path)
        output_path.write_bytes(b"RIFF")
        return SimpleNamespace(returncode=0, stderr="")

    with patch("mkv_episode_matcher.audio_chunk_extractor.subprocess.run",
               side_effect=fake_run):
        extractor = AudioChunkExtractor()
        input_video = tmp_path / "episode.mkv"
        input_video.write_bytes(b"0" * 200_000)
        first = extractor.extract(input_video, start_time=1.25, duration=30)
        second = extractor.extract(input_video, start_time=2.75, duration=30)
        assert "AT1250ms" in first.name
        assert "AT2750ms" in second.name
        assert first != second
        assert len(ffmpeg_outputs) == 2
        assert all(path != first and path != second for path in ffmpeg_outputs)
        assert all(".tmp." in path.name for path in ffmpeg_outputs)
        assert all(not path.exists() for path in ffmpeg_outputs)
        assert first.exists()
        assert second.exists()


def test_extract_ffmpeg_failure_does_not_leave_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mkv_episode_matcher.audio_chunk_extractor.tempfile.gettempdir",
        lambda: str(tmp_path),
    )

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        output_path = Path(cmd[-1])
        output_path.write_bytes(b"PARTIAL")
        return SimpleNamespace(returncode=1, stderr="ffmpeg failed")

    input_video = tmp_path / "episode.mkv"
    input_video.write_bytes(b"0" * 200_000)
    expected_name = unique_filename(input_video, ".30S.AT1250ms.wav")
    expected_chunk_path = (
        tmp_path / "mkv-episode-matcher-audio-chunks" / expected_name
    )

    with patch("mkv_episode_matcher.audio_chunk_extractor.subprocess.run",
               side_effect=fake_run):
        extractor = AudioChunkExtractor()
        with pytest.raises(RuntimeError, match="ffmpeg failed extracting chunk"):
            extractor.extract(input_video, start_time=1.25, duration=30)

    assert not expected_chunk_path.exists()


def test_extract_ffmpeg_failure_does_not_clobber_existing_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mkv_episode_matcher.audio_chunk_extractor.tempfile.gettempdir",
        lambda: str(tmp_path),
    )

    input_video = tmp_path / "episode.mkv"
    input_video.write_bytes(b"0" * 200_000)
    expected_name = unique_filename(input_video, ".30S.AT1250ms.wav")
    expected_chunk_path = (
        tmp_path / "mkv-episode-matcher-audio-chunks" / expected_name
    )
    expected_chunk_path.parent.mkdir(parents=True, exist_ok=True)
    original_bytes = b"KNOWN_GOOD"
    expected_chunk_path.write_bytes(original_bytes)

    original_exists = Path.exists

    def fake_exists(path_self):  # noqa: ANN001
        if os.fspath(path_self) == os.fspath(expected_chunk_path):
            return False
        return original_exists(path_self)

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        output_path = Path(cmd[-1])
        output_path.write_bytes(b"PARTIAL")
        return SimpleNamespace(returncode=1, stderr="ffmpeg failed")

    monkeypatch.setattr(Path, "exists", fake_exists)
    with patch("mkv_episode_matcher.audio_chunk_extractor.subprocess.run",
               side_effect=fake_run):
        extractor = AudioChunkExtractor()
        with pytest.raises(RuntimeError, match="ffmpeg failed extracting chunk"):
            extractor.extract(input_video, start_time=1.25, duration=30)

    assert expected_chunk_path.read_bytes() == original_bytes


def test_effective_start_time_is_clamped_to_zero():
    assert AudioChunkExtractor._effective_start_seconds(-2.5) == 0.0
