from types import SimpleNamespace

import pytest

from mkv_episode_matcher import transcribers
from mkv_episode_matcher.transcribers import FasterWhisperTranscriber


def test_is_faster_whisper_supported_missing_dependency(monkeypatch):
    monkeypatch.setattr(transcribers.importlib.util, "find_spec", lambda _name: None)
    supported, reason = transcribers.is_faster_whisper_supported()
    assert supported is False
    assert "not installed" in reason


def test_faster_whisper_raises_clear_error_when_unsupported(monkeypatch):
    monkeypatch.setattr(
        transcribers,
        "is_faster_whisper_supported",
        lambda: (False, transcribers.FASTER_WHISPER_UNSUPPORTED_MESSAGE),
    )
    with pytest.raises(RuntimeError, match="not installed"):
        FasterWhisperTranscriber("small.en")


def test_faster_whisper_transcribe_returns_first_item(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribers, "is_faster_whisper_supported", lambda: (True, None))

    class DummyPipeline:
        def transcribe(self, audio_paths, batch_size):
            assert len(audio_paths) == 1
            assert batch_size == 1
            return [SimpleNamespace(text="hello world")], None

    monkeypatch.setattr(
        FasterWhisperTranscriber,
        "_load_pipeline",
        staticmethod(lambda *_args, **_kwargs: DummyPipeline()),
    )

    transcriber = FasterWhisperTranscriber(None)
    text = transcriber.transcribe(tmp_path / "chunk.wav")
    assert text == "hello world"


def test_faster_whisper_batch_transcribe_many_splits_batches(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribers, "is_faster_whisper_supported", lambda: (True, None))
    monkeypatch.setenv("FASTER_WHISPER_BATCH_SIZE", "2")
    calls = []

    class DummyPipeline:
        def transcribe(self, audio_paths, batch_size):
            calls.append((list(audio_paths), batch_size))
            rows = [[SimpleNamespace(text=f"text-{idx}")] for idx, _ in enumerate(audio_paths)]
            return rows, None

    monkeypatch.setattr(
        FasterWhisperTranscriber,
        "_load_pipeline",
        staticmethod(lambda *_args, **_kwargs: DummyPipeline()),
    )

    transcriber = FasterWhisperTranscriber(None)
    paths = [tmp_path / f"chunk_{i}.wav" for i in range(5)]
    texts = transcriber.transcribe_many(paths)

    assert calls == [
        ([str(paths[0]), str(paths[1])], 2),
        ([str(paths[2]), str(paths[3])], 2),
        ([str(paths[4])], 1),
    ]
    assert texts == ["text-0", "text-1", "text-0", "text-1", "text-0"]


def test_faster_whisper_batch_transcribe_many_handles_batch_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribers, "is_faster_whisper_supported", lambda: (True, None))
    monkeypatch.setenv("FASTER_WHISPER_BATCH_SIZE", "2")
    calls = []

    class DummyPipeline:
        def transcribe(self, audio_paths, batch_size):
            calls.append((list(audio_paths), batch_size))
            if len(calls) == 2:
                raise RuntimeError("boom")
            return [[SimpleNamespace(text=f"text-{i}")] for i, _ in enumerate(audio_paths)], None

    monkeypatch.setattr(
        FasterWhisperTranscriber,
        "_load_pipeline",
        staticmethod(lambda *_args, **_kwargs: DummyPipeline()),
    )

    transcriber = FasterWhisperTranscriber(None)
    paths = [tmp_path / f"chunk_{i}.wav" for i in range(4)]
    with pytest.raises(RuntimeError, match="batch_id=1"):
        transcriber.transcribe_many(paths)
