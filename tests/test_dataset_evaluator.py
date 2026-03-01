import numpy as np
from rich.progress import Progress

from mkv_episode_matcher.dataset_evaluator import (
    DatasetPaths,
    IntervalSubtitleIndex,
    SubtitleWindowMetadata,
    SegmentRecord,
    _filter_records_by_profiles,
    _first_expected_rank,
    _load_segments_from_records,
    _parse_episode_key,
    _resolve_dataset_paths,
    _segments_by_profile,
    _score_segments,
    _write_errors_output,
)
from mkv_episode_matcher.episode import EpisodeKey


class DummyModel:
    def encode_query(self, _text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeAnnIndex:
    def __init__(self, ids, distances):
        self._ids = np.array([ids], dtype=np.int32)
        self._distances = np.array([distances], dtype=np.float32)

    def get_current_count(self):
        return self._ids.shape[1]

    def knn_query(self, _query, k, num_threads=1, filter=None):
        return self._ids[:, :k], self._distances[:, :k]


def test_parse_episode_key():
    assert _parse_episode_key("S02E11") == EpisodeKey(2, 11)


def test_first_expected_rank_handles_multi_episode_expected():
    ranked = [EpisodeKey(1, 3), EpisodeKey(1, 4), EpisodeKey(1, 5)]
    expected = {EpisodeKey(1, 4), EpisodeKey(1, 7)}
    assert _first_expected_rank(ranked, expected) == 2


def test_score_segments_counts_multi_episode_hit_in_top_k():
    segments = [
        SegmentRecord(
            segment_index=0,
            transcript_text="sample",
            expected={EpisodeKey(1, 2), EpisodeKey(1, 3)},
            video_path="video.mkv",
        )
    ]
    subtitle_indexes = {
        0: IntervalSubtitleIndex(
            episodes=[EpisodeKey(1, 1), EpisodeKey(1, 3), EpisodeKey(1, 2)],
            index=FakeAnnIndex(ids=[0, 1, 2], distances=[0.2, 0.1, 0.3]),
        )
    }

    report = _score_segments(
        segments=segments,
        subtitle_indexes=subtitle_indexes,
        model=DummyModel(),
        top_ks=[1, 3],
        max_failures=5,
        low_info_filter=False,
    )

    assert report["segments_total"] == 1
    assert report["segments_evaluated"] == 1
    assert report["accuracy"]["top_1"] == 1.0
    assert report["accuracy"]["top_3"] == 1.0
    assert report["mrr"] == 1.0


def test_resolve_dataset_paths_reads_meta_paths(tmp_path):
    (tmp_path / "meta.json").write_text(
        (
            "{"
            "\"paths\":{"
            "\"manifest\":\"manifest.jsonl\","
            "\"subtitles_dir\":\"subtitles/srt\","
            "\"transcriptions_dir\":\"transcriptions/text\""
            "}"
            "}"
        ),
        encoding="utf-8",
    )

    resolved = _resolve_dataset_paths(tmp_path)

    assert resolved == DatasetPaths(
        manifest=tmp_path / "manifest.jsonl",
        subtitles_dir=tmp_path / "subtitles" / "srt",
        transcriptions_dir=tmp_path / "transcriptions" / "text",
    )


def test_load_segments_from_manifest_records(tmp_path):
    transcriptions_dir = tmp_path / "transcriptions" / "text"
    transcriptions_dir.mkdir(parents=True)
    transcript_file = transcriptions_dir / "video_a.json"
    transcript_file.write_text(
        "{\"0\":\"hello there\",\"2\":\"general kenobi\"}",
        encoding="utf-8",
    )
    records = [{
        "video_path": "/shows/video_a.mkv",
        "episodes": ["S01E02", "S01E03"],
        "transcription_path": "transcriptions/text/video_a.json",
        "variant_profile": "left",
    }]

    with Progress() as progress:
        task_id = progress.add_task("load", total=len(records))
        segments = _load_segments_from_records(records, tmp_path, task_id, progress)

    assert len(segments) == 2
    assert segments[0].expected == {EpisodeKey(1, 2), EpisodeKey(1, 3)}
    assert segments[0].segment_index == 0
    assert segments[1].segment_index == 2
    assert segments[0].variant_profile == "left"


def test_filter_records_by_profiles_defaults_to_aligned():
    records = [
        {"video_path": "a", "variant_profile": "left"},
        {"video_path": "b"},
    ]
    filtered = _filter_records_by_profiles(records, ["aligned"])
    assert filtered == [{"video_path": "b"}]


def test_segments_grouped_by_profile():
    segments = [
        SegmentRecord(0, "a", {EpisodeKey(1, 1)}, "v1", "aligned"),
        SegmentRecord(1, "b", {EpisodeKey(1, 2)}, "v2", "left"),
        SegmentRecord(2, "c", {EpisodeKey(1, 3)}, "v3", "left"),
    ]
    grouped = _segments_by_profile(segments)
    assert set(grouped.keys()) == {"aligned", "left"}
    assert len(grouped["left"]) == 2


def test_score_segments_uses_neighbor_windows():
    segments = [
        SegmentRecord(
            segment_index=1,
            transcript_text="neighbor lookup",
            expected={EpisodeKey(2, 1)},
            video_path="video.mkv",
        )
    ]
    subtitle_indexes = {
        2: IntervalSubtitleIndex(
            episodes=[EpisodeKey(2, 1)],
            index=FakeAnnIndex(ids=[0], distances=[0.1]),
        )
    }

    report = _score_segments(
        segments=segments,
        subtitle_indexes=subtitle_indexes,
        model=DummyModel(),
        top_ks=[1, 3],
        max_failures=5,
        segment_duration_seconds=30,
        subtitle_overlap_seconds=5,
        low_info_filter=False,
    )

    assert report["segments_evaluated"] == 1
    assert report["accuracy"]["top_1"] == 1.0
    assert report["segments_missing_interval"] == 0


def test_score_segments_dedupes_repeated_episode_candidates():
    segments = [
        SegmentRecord(
            segment_index=0,
            transcript_text="dedupe",
            expected={EpisodeKey(1, 2)},
            video_path="video.mkv",
        )
    ]
    subtitle_indexes = {
        0: IntervalSubtitleIndex(
            episodes=[EpisodeKey(1, 1), EpisodeKey(1, 1), EpisodeKey(1, 2)],
            index=FakeAnnIndex(ids=[0, 1, 2], distances=[0.1, 0.2, 0.3]),
        )
    }

    report = _score_segments(
        segments=segments,
        subtitle_indexes=subtitle_indexes,
        model=DummyModel(),
        top_ks=[1, 2],
        max_failures=5,
        segment_duration_seconds=30,
        subtitle_overlap_seconds=5,
        low_info_filter=False,
    )

    # After per-episode dedupe, rank is 2 (not 3).
    assert report["accuracy"]["top_1"] == 0.0
    assert report["accuracy"]["top_2"] == 1.0
    assert report["mrr"] == 0.5


def test_score_segments_include_match_text_in_mismatch_output():
    segments = [
        SegmentRecord(
            segment_index=0,
            transcript_text="who is this",
            expected={EpisodeKey(1, 1)},
            video_path="video.mkv",
        )
    ]
    subtitle_indexes = {
        0: IntervalSubtitleIndex(
            episodes=[EpisodeKey(1, 2), EpisodeKey(1, 1)],
            index=FakeAnnIndex(ids=[0, 1], distances=[0.1, 0.2]),
            metadata=[
                SubtitleWindowMetadata(
                    episode=EpisodeKey(1, 2),
                    start_ms=0,
                    end_ms=30000,
                    text="wrong-episode subtitle line",
                    subtitle_path="/tmp/S01E02.srt",
                ),
                SubtitleWindowMetadata(
                    episode=EpisodeKey(1, 1),
                    start_ms=0,
                    end_ms=30000,
                    text="right-episode subtitle line",
                    subtitle_path="/tmp/S01E01.srt",
                ),
            ],
        )
    }

    report = _score_segments(
        segments=segments,
        subtitle_indexes=subtitle_indexes,
        model=DummyModel(),
        top_ks=[1, 2],
        max_failures=5,
        include_match_text=True,
        low_info_filter=False,
    )

    mismatch = report["top1_mismatches"][0]
    assert mismatch["top_predictions"][0] == "S01E02"
    assert mismatch["segment_start_seconds"] == 0
    assert mismatch["segment_end_seconds"] == 30
    assert mismatch["segment_start_timestamp"] == "00:00:00"
    assert mismatch["segment_end_timestamp"] == "00:00:30"
    assert "top_prediction_matches" in mismatch
    top_match = mismatch["top_prediction_matches"][0]
    assert top_match["episode"] == "S01E02"
    assert top_match["match_windows"][0]["subtitle_text"] == "wrong-episode subtitle line"
    assert top_match["match_windows"][0]["subtitle_path"] == "/tmp/S01E02.srt"


def test_write_errors_output_csv_includes_match_text_column(tmp_path):
    output_path = tmp_path / "errors.csv"
    _write_errors_output(
        output_path,
        [
            {
                "video_path": "video.mkv",
                "segment_index": 1,
                "variant_profile": "aligned",
                "expected": ["S01E01"],
                "top_predictions": ["S01E02"],
                "top_prediction_matches": [{"episode": "S01E02", "match_windows": []}],
                "rank": 2,
                "transcript_text": "line",
            }
        ],
    )
    csv_text = output_path.read_text(encoding="utf-8")
    assert "top_prediction_matches" in csv_text.splitlines()[0]
    assert "segment_start_timestamp" in csv_text.splitlines()[0]
    assert "S01E02" in csv_text


def test_write_errors_output_html_includes_transcript_and_match_windows(tmp_path):
    output_path = tmp_path / "errors.html"
    _write_errors_output(
        output_path,
        [
            {
                "video_path": "video.mkv",
                "segment_index": 7,
                "segment_start_seconds": 210,
                "segment_end_seconds": 240,
                "segment_start_timestamp": "00:03:30",
                "segment_end_timestamp": "00:04:00",
                "variant_profile": "aligned",
                "expected": ["S01E01"],
                "top_predictions": ["S01E02", "S01E01"],
                "rank": 2,
                "transcript_text": "hello there",
                "top_prediction_matches": [
                    {
                        "episode": "S01E02",
                        "score": 0.42,
                        "support_windows": 2,
                        "nearest_window_offset": 0,
                        "match_windows": [
                            {
                                "window_start_seconds": 10.0,
                                "window_end_seconds": 40.0,
                                "adjusted_distance": 0.42,
                                "subtitle_text": "subtitle text sample",
                                "subtitle_path": "/tmp/S01E02.srt",
                            }
                        ],
                    }
                ],
            }
        ],
    )
    html_text = output_path.read_text(encoding="utf-8")
    assert "<html>" in html_text
    assert "Episode Matcher Mismatch Report" in html_text
    assert "00:03:30 - 00:04:00" in html_text
    assert "hello there" in html_text
    assert "subtitle text sample" in html_text
    assert "/tmp/S01E02.srt" in html_text
