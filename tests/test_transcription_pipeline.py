from pathlib import Path

from mkv_episode_matcher.transcription_pipeline import (
    SegmentSpec,
    TranscriptionPipeline,
)


def test_pipeline_extracts_in_effective_offset_order_and_maps_by_segment_index(tmp_path):
    extract_calls: list[float] = []

    class DummyExtractor:
        def extract(self, _video_path, start_time, _duration):
            extract_calls.append(start_time)
            return tmp_path / f"chunk-{int(start_time)}.wav"

    class DummyBatchTranscriber:
        def transcribe_many(self, paths):
            return [f"text-{Path(path).stem}" for path in paths]

    failures = []
    pipeline = TranscriptionPipeline(
        audio_extractor=DummyExtractor(),
        transcriber=DummyBatchTranscriber(),
        normalize_transcript=lambda value: str(value),
        write_failure=failures.append,
        io_workers=1,
        transcribe_workers=1,
        batch_size=8,
    )

    video = tmp_path / "episode.mkv"
    specs_by_path = {
        video: [
            SegmentSpec(video, segment_index=9, base_offset_seconds=270.0, effective_offset_seconds=270.0, duration_seconds=30),
            SegmentSpec(video, segment_index=1, base_offset_seconds=30.0, effective_offset_seconds=30.0, duration_seconds=30),
        ]
    }

    transcribed_by_path, stats_by_path = pipeline.run(specs_by_path)

    assert extract_calls == [30.0, 270.0]
    assert transcribed_by_path[video][1] == "text-chunk-30"
    assert transcribed_by_path[video][9] == "text-chunk-270"
    assert stats_by_path[video].segments_attempted == 2
    assert stats_by_path[video].segments_transcribed == 2
    assert stats_by_path[video].segments_failed == 0
    assert failures == []


def test_pipeline_counts_failures_for_extract_and_empty_transcript(tmp_path):
    class DummyExtractor:
        def extract(self, _video_path, start_time, _duration):
            if start_time < 5:
                raise RuntimeError("boom")
            return tmp_path / "ok.wav"

    class DummySingleTranscriber:
        def transcribe(self, _path):
            return ""

    failures = []
    pipeline = TranscriptionPipeline(
        audio_extractor=DummyExtractor(),
        transcriber=DummySingleTranscriber(),
        normalize_transcript=lambda value: value or None,
        write_failure=failures.append,
        io_workers=1,
        transcribe_workers=1,
    )

    video = tmp_path / "episode.mkv"
    specs_by_path = {
        video: [
            SegmentSpec(video, segment_index=0, base_offset_seconds=0.0, effective_offset_seconds=0.0, duration_seconds=30),
            SegmentSpec(video, segment_index=1, base_offset_seconds=30.0, effective_offset_seconds=30.0, duration_seconds=30),
        ]
    }

    transcribed_by_path, stats_by_path = pipeline.run(specs_by_path)

    assert transcribed_by_path[video] == {}
    assert stats_by_path[video].segments_attempted == 2
    assert stats_by_path[video].segments_transcribed == 0
    assert stats_by_path[video].segments_failed == 2
    assert len(failures) == 2
