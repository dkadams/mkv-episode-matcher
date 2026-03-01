import numpy as np
from rich.progress import Progress

from mkv_episode_matcher.dataset_evaluator import (
    DatasetPaths,
    SegmentRecord,
    _filter_records_by_profiles,
    _first_expected_rank,
    _load_segments_from_records,
    _parse_episode_key,
    _resolve_dataset_paths,
    _segments_by_profile,
    _score_segments,
)
from mkv_episode_matcher.episode import EpisodeKey


class DummyModel:
    def encode_query(self, _text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


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
    subtitle_vectors = {
        0: (
            [EpisodeKey(1, 1), EpisodeKey(1, 3), EpisodeKey(1, 2)],
            np.array(
                [
                    [0.8, 0.0],
                    [0.9, 0.0],
                    [0.7, 0.0],
                ],
                dtype=np.float32,
            ),
        )
    }

    report = _score_segments(
        segments=segments,
        subtitle_vectors=subtitle_vectors,
        model=DummyModel(),
        top_ks=[1, 3],
        max_failures=5,
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
    subtitle_vectors = {
        2: (
            [EpisodeKey(2, 1)],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
    }

    report = _score_segments(
        segments=segments,
        subtitle_vectors=subtitle_vectors,
        model=DummyModel(),
        top_ks=[1, 3],
        max_failures=5,
        segment_duration_seconds=30,
        subtitle_overlap_seconds=5,
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
    subtitle_vectors = {
        0: (
            [EpisodeKey(1, 1), EpisodeKey(1, 1), EpisodeKey(1, 2)],
            np.array(
                [
                    [0.9, 0.0],
                    [0.8, 0.0],
                    [0.7, 0.0],
                ],
                dtype=np.float32,
            ),
        )
    }

    report = _score_segments(
        segments=segments,
        subtitle_vectors=subtitle_vectors,
        model=DummyModel(),
        top_ks=[1, 2],
        max_failures=5,
        segment_duration_seconds=30,
        subtitle_overlap_seconds=5,
    )

    # After per-episode dedupe, rank is 2 (not 3).
    assert report["accuracy"]["top_1"] == 0.0
    assert report["accuracy"]["top_2"] == 1.0
    assert report["mrr"] == 0.5
