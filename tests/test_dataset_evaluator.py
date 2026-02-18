import numpy as np

from mkv_episode_matcher.dataset_evaluator import (
    SegmentRecord,
    _first_expected_rank,
    _parse_episode_key,
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
