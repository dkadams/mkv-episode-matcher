import json
from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.segment_transcriber import SegmentTranscriber
from mkv_episode_matcher.series import Series


def _config() -> Configuration:
    return Configuration(args=Namespace(), stored=ConfigParser())


def _series(tmp_path: Path) -> Series:
    return Series(
        dir=tmp_path / "series",
        detail={"name": "series"},
        name="series",
        segment_duration=30,
        random_seed=12345,
    )


def test_execute_uses_transcribe_many_when_available(monkeypatch, tmp_path):
    class DummyAudioChunkExtractor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def extract(self, _file_path, offset, _duration):
            return tmp_path / f"chunk_{int(offset)}.wav"

    class DummyBatchTranscriber:
        def __init__(self, _model_name):
            self.calls = []

        def transcribe_many(self, audio_paths):
            self.calls.append(list(audio_paths))
            return [f"text-{Path(path).stem}" for path in audio_paths]

    monkeypatch.setattr(
        "mkv_episode_matcher.segment_transcriber.AudioChunkExtractor",
        DummyAudioChunkExtractor,
    )

    series = _series(tmp_path)
    segment_transcriber = SegmentTranscriber(
        _config(),
        series,
        "unused",
        DummyBatchTranscriber,
    )

    video = tmp_path / "episode.mkv"
    output = series.transcription_file(video)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"1": "cached"}), encoding="utf-8")

    result = segment_transcriber.execute([(video, [0, 2])])
    assert result == {video: output}

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["0"] == "text-chunk_0"
    assert payload["1"] == "cached"
    assert payload["2"] == "text-chunk_60"


def test_execute_falls_back_to_transcribe(monkeypatch, tmp_path):
    calls = []

    class DummyAudioChunkExtractor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def extract(self, _file_path, offset, _duration):
            return tmp_path / f"chunk_{int(offset)}.wav"

    class SingleTranscriber:
        def __init__(self, _model_name):
            pass

        def transcribe(self, audio_path):
            calls.append(Path(audio_path))
            return {"text": Path(audio_path).stem}

    monkeypatch.setattr(
        "mkv_episode_matcher.segment_transcriber.AudioChunkExtractor",
        DummyAudioChunkExtractor,
    )

    series = _series(tmp_path)
    segment_transcriber = SegmentTranscriber(
        _config(),
        series,
        "unused",
        SingleTranscriber,
    )

    video = tmp_path / "episode.mkv"
    segment_transcriber.execute([(video, [0, 1])])
    assert calls == [tmp_path / "chunk_0.wav", tmp_path / "chunk_30.wav"]
