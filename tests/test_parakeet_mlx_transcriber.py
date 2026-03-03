from types import SimpleNamespace

import pytest

from mkv_episode_matcher import transcribers
from mkv_episode_matcher.transcribers import ParakeetMlxTranscriber


def test_is_parakeet_mlx_supported_non_macos(monkeypatch):
    monkeypatch.setattr(transcribers.sys, "platform", "linux")

    supported, reason = transcribers.is_parakeet_mlx_supported()

    assert supported is False
    assert "macOS" in reason


def test_is_parakeet_mlx_supported_missing_dependencies(monkeypatch):
    monkeypatch.setattr(transcribers.sys, "platform", "darwin")

    def fake_find_spec(name):
        if name == "mlx":
            return None
        return SimpleNamespace()

    monkeypatch.setattr(transcribers.importlib.util, "find_spec", fake_find_spec)

    supported, reason = transcribers.is_parakeet_mlx_supported()

    assert supported is False
    assert "optional dependencies" in reason


def test_parakeet_mlx_raises_clear_error_when_unsupported(monkeypatch):
    monkeypatch.setattr(
        transcribers,
        "is_parakeet_mlx_supported",
        lambda: (False, transcribers.PARAKEET_MLX_UNSUPPORTED_MESSAGE),
    )

    with pytest.raises(RuntimeError, match="only supported on macOS"):
        ParakeetMlxTranscriber(None)


def test_parakeet_mlx_uses_default_model_for_whisper_name(monkeypatch):
    monkeypatch.setattr(transcribers, "is_parakeet_mlx_supported", lambda: (True, None))
    monkeypatch.setattr(
        ParakeetMlxTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: object()),
    )
    transcriber = ParakeetMlxTranscriber("small.en")
    assert transcriber.model_name == "mlx-community/parakeet-tdt-0.6b-v3"


def test_parakeet_mlx_uses_explicit_model_name(monkeypatch):
    monkeypatch.setattr(transcribers, "is_parakeet_mlx_supported", lambda: (True, None))
    monkeypatch.setattr(
        ParakeetMlxTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: object()),
    )
    transcriber = ParakeetMlxTranscriber("mlx-community/parakeet-tdt-0.6b-v3")
    assert transcriber.model_name == "mlx-community/parakeet-tdt-0.6b-v3"


def test_parakeet_mlx_transcribe_many_single_item(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribers, "is_parakeet_mlx_supported", lambda: (True, None))
    monkeypatch.setattr(
        ParakeetMlxTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: object()),
    )

    monkeypatch.setattr(
        ParakeetMlxTranscriber,
        "transcribe_many",
        lambda _self, _paths: ["hello world"],
    )

    transcriber = ParakeetMlxTranscriber(None)
    text = transcriber.transcribe_many([tmp_path / "chunk.wav"])
    assert text == ["hello world"]


def test_parakeet_mlx_batch_transcribe_many_splits_batches(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribers, "is_parakeet_mlx_supported", lambda: (True, None))
    monkeypatch.setenv("PARAKEET_MLX_BATCH_SIZE", "2")
    monkeypatch.setattr(
        ParakeetMlxTranscriber,
        "_load_model",
        staticmethod(lambda *_args, **_kwargs: object()),
    )

    calls = []

    def fake_transcribe_batch(self, audio_paths, *, batch_id):
        calls.append(list(audio_paths))
        return [f"text-{path.stem}" for path in audio_paths]

    monkeypatch.setattr(
        ParakeetMlxTranscriber,
        "_transcribe_batch",
        fake_transcribe_batch,
    )

    transcriber = ParakeetMlxTranscriber(None)
    audio_paths = [tmp_path / f"chunk_{i}.wav" for i in range(5)]
    text = transcriber.transcribe_many(audio_paths)

    assert calls == [
        audio_paths[0:2],
        audio_paths[2:4],
        audio_paths[4:5],
    ]
    assert text == [f"text-chunk_{i}" for i in range(5)]


def test_parakeet_mlx_batch_transcribe_many_handles_batch_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribers, "is_parakeet_mlx_supported", lambda: (True, None))
    monkeypatch.setenv("PARAKEET_MLX_BATCH_SIZE", "2")
    monkeypatch.setattr(
        ParakeetMlxTranscriber,
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
        ParakeetMlxTranscriber,
        "_transcribe_batch",
        fake_transcribe_batch,
    )

    transcriber = ParakeetMlxTranscriber(None)
    audio_paths = [tmp_path / f"chunk_{i}.wav" for i in range(4)]
    with pytest.raises(RuntimeError, match="batch_id=1"):
        transcriber.transcribe_many(audio_paths)
