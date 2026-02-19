from types import SimpleNamespace

from mkv_episode_matcher.transcribers import ParakeetMlxCliTranscriber


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
