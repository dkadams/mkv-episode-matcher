import json
from pathlib import Path

from mkv_episode_matcher.misalignment import MisalignmentPolicy
from mkv_episode_matcher.segment_transcriber import SegmentTranscriber


class DummySeries:
    segment_duration = 30

    def __init__(self, base: Path):
        self.transcriptions_text_dir = base / "aligned"

    @staticmethod
    def transcription_file_name(input_path: Path) -> str:
        return f"{input_path.stem}.json"


class DummyTranscriber:
    def transcribe_many(self, chunk_paths: list[Path]):
        return ["hello" for _ in chunk_paths]


def test_transcriber_applies_misaligned_offsets(monkeypatch, tmp_path):
    calls = []

    class FakeAudioChunkExtractor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def extract(self, path, start_time, duration):
            calls.append((path, start_time, duration))
            return tmp_path / "chunk.wav"

    monkeypatch.setattr(
        "mkv_episode_matcher.segment_transcriber.AudioChunkExtractor",
        FakeAudioChunkExtractor,
    )

    policy = MisalignmentPolicy("right", 1.0, 1.0, 99, per_segment=False)
    transcriber = SegmentTranscriber(
        config=None,  # unused by SegmentTranscriber
        series=DummySeries(tmp_path),
        model_name="ignored",
        transcriber=DummyTranscriber(),
        misalignment_policy=policy,
        variant_id="misaligned:right",
        output_dir=tmp_path / "out",
    )

    output = transcriber.transcribe(Path("episode.mkv"), [0, 1])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert list(payload.keys()) == ["0", "1"]
    assert calls[0][1] == 1.0
    assert calls[1][1] == 31.0
