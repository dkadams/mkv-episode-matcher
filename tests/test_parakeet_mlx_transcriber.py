from types import SimpleNamespace

import pytest

from mkv_episode_matcher.transcribers import (
    ParakeetMlxCliTranscriber,
    ParakeetMlxGenerateBatchTranscriber,
)


def test_parakeet_mlx_uses_default_model_for_whisper_name(monkeypatch):
    monkeypatch.setattr(
        ParakeetMlxCliTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: object()),
    )
    transcriber = ParakeetMlxCliTranscriber("small.en")
    assert transcriber.model_name == "mlx-community/parakeet-tdt-0.6b-v3"


def test_parakeet_mlx_uses_explicit_model_name(monkeypatch):
    monkeypatch.setattr(
        ParakeetMlxCliTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: object()),
    )
    transcriber = ParakeetMlxCliTranscriber("mlx-community/parakeet-tdt-0.6b-v3")
    assert transcriber.model_name == "mlx-community/parakeet-tdt-0.6b-v3"


def test_parakeet_mlx_transcribe_returns_text(monkeypatch, tmp_path):
    class DummyModel:
        def transcribe(self, _audio, chunk_duration, overlap_duration):
            assert chunk_duration == 120.0
            assert overlap_duration == 15.0
            return SimpleNamespace(text="hello world")

    monkeypatch.setattr(
        ParakeetMlxCliTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: DummyModel()),
    )
    transcriber = ParakeetMlxCliTranscriber(None)
    text = transcriber.transcribe(tmp_path / "chunk.wav")
    assert text == "hello world"


def test_parakeet_mlx_transcribe_handles_exception(monkeypatch, tmp_path):
    class DummyModel:
        def transcribe(self, _audio, chunk_duration, overlap_duration):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        ParakeetMlxCliTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: DummyModel()),
    )
    transcriber = ParakeetMlxCliTranscriber(None)
    assert transcriber.transcribe(tmp_path / "chunk.wav") is None


def test_parakeet_mlx_batch_transcribe_many_splits_batches(monkeypatch, tmp_path):
    monkeypatch.setenv("PARAKEET_MLX_BATCH_SIZE", "2")
    monkeypatch.setattr(
        ParakeetMlxGenerateBatchTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: object()),
    )

    calls = []

    def fake_transcribe_batch(self, audio_paths, *, batch_id):
        calls.append(list(audio_paths))
        return [f"text-{path.stem}" for path in audio_paths]

    monkeypatch.setattr(
        ParakeetMlxGenerateBatchTranscriber,
        "_transcribe_batch",
        fake_transcribe_batch,
    )

    transcriber = ParakeetMlxGenerateBatchTranscriber(None)
    audio_paths = [tmp_path / f"chunk_{i}.wav" for i in range(5)]
    text = transcriber.transcribe_many(audio_paths)

    assert calls == [
        audio_paths[0:2],
        audio_paths[2:4],
        audio_paths[4:5],
    ]
    assert text == [f"text-chunk_{i}" for i in range(5)]


def test_parakeet_mlx_batch_transcribe_many_handles_batch_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("PARAKEET_MLX_BATCH_SIZE", "2")
    monkeypatch.setattr(
        ParakeetMlxGenerateBatchTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: object()),
    )

    calls = []

    def fake_transcribe_batch(self, audio_paths, *, batch_id):
        calls.append(list(audio_paths))
        if len(calls) == 2:
            raise RuntimeError("boom")
        return [f"text-{path.stem}" for path in audio_paths]

    monkeypatch.setattr(
        ParakeetMlxGenerateBatchTranscriber,
        "_transcribe_batch",
        fake_transcribe_batch,
    )

    transcriber = ParakeetMlxGenerateBatchTranscriber(None)
    audio_paths = [tmp_path / f"chunk_{i}.wav" for i in range(4)]
    with pytest.raises(RuntimeError, match="batch_id=1"):
        transcriber.transcribe_many(audio_paths)
