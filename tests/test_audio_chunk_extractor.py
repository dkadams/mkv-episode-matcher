from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor


def test_extract_uses_millisecond_start_in_chunk_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mkv_episode_matcher.audio_chunk_extractor.tempfile.gettempdir",
        lambda: str(tmp_path),
    )

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        chunk_path = Path(cmd[-1])
        chunk_path.write_bytes(b"RIFF")
        return SimpleNamespace(returncode=0, stderr="")

    with patch("mkv_episode_matcher.audio_chunk_extractor.subprocess.run",
               side_effect=fake_run):
        extractor = AudioChunkExtractor()
        first = extractor.extract(Path("episode.mkv"), start_time=1.25, duration=30)
        second = extractor.extract(Path("episode.mkv"), start_time=2.75, duration=30)
        assert "AT1250ms" in first.name
        assert "AT2750ms" in second.name
        assert first != second


def test_effective_start_time_is_clamped_to_zero():
    assert AudioChunkExtractor._effective_start_seconds(-2.5) == 0.0
