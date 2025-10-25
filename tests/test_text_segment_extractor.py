from pathlib import Path

from mkv_episode_matcher import text_segment_extractor as tse


class DummyTranscriber:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path):
        path = Path(audio_path)
        self.calls.append(path)
        return {"text": f"text-for-{path.stem}"}


def test_get_random_segments_uses_transcriber(monkeypatch):
    extract_calls = []

    class DummyAudioChunkExtractor:
        @staticmethod
        def get_video_duration(_):
            return 400

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def extract(self, file_path, offset, duration):
            extract_calls.append((offset, duration))
            return Path(f"/fake/chunk_{offset}.wav")

    monkeypatch.setattr(tse, "AudioChunkExtractor", DummyAudioChunkExtractor)
    monkeypatch.setattr(tse.random, "sample", lambda population, k: [1, 3][:k])

    transcriber = DummyTranscriber()
    extractor = tse.TextSegmentExtractor("tiny", transcriber=transcriber)

    segments = extractor.get_random_segments(Path("video.mkv"), duration=50, count=2)

    assert [index for index, _segment in segments] == [1, 3]
    assert extract_calls == [(50, 50), (150, 50)]
    assert transcriber.calls == [
        Path("/fake/chunk_50.wav"),
        Path("/fake/chunk_150.wav"),
    ]


def test_get_text_segments_returns_indexed_text(monkeypatch):
    class DummyAudioChunkExtractor:
        @staticmethod
        def get_video_duration(_):
            return 80

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def extract(self, file_path, offset, duration):
            return Path(f"/fake/chunk_{offset}.wav")

    monkeypatch.setattr(tse, "AudioChunkExtractor", DummyAudioChunkExtractor)
    monkeypatch.setattr(tse.random, "sample", lambda population, k: list(population)[:k])

    class TextReturningTranscriber:
        def transcribe(self, audio_path):
            return {"text": Path(audio_path).stem}

    extractor = tse.TextSegmentExtractor("tiny", transcriber=TextReturningTranscriber())

    segments = extractor.get_text_segments(Path("video.mkv"), duration=30, count=5)

    assert segments == [
        (0, "chunk_0"),
        (1, "chunk_30"),
        (2, "chunk_60"),
    ]


def test_whisperkit_extract_text_prefers_segments():
    payload = {
        "files": [
            {
                "path": "audio.wav",
                "text": "short text",
                "segments": [
                    {"text": "Hello"},
                    {"text": "from"},
                    {"text": "WhisperKit"},
                ],
            }
        ]
    }

    text = tse.WhisperKitCliTranscriber._extract_text(payload)

    assert text == "Hello from WhisperKit"
