import pytest

from argparse import Namespace
from types import SimpleNamespace

from mkv_episode_matcher.windowing import (
    distance_with_window_penalty,
    make_window_config,
    map_segment_index_to_window_index,
    merge_episode_window_hit,
    neighbor_window_indexes,
    resolve_low_info_filter,
    resolve_max_results_per_query,
    resolve_window_expansion_mode,
    support_aware_score,
    resolve_subtitle_overlap_seconds,
    should_expand_to_neighbor_windows,
)


def test_make_window_config_builds_stride_and_profile():
    config = make_window_config(window_seconds=30, overlap_seconds=5)

    assert config.stride_seconds == 25
    assert config.profile_key == "w30_o5"


def test_make_window_config_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        make_window_config(window_seconds=30, overlap_seconds=30)

    with pytest.raises(ValueError):
        make_window_config(window_seconds=30, overlap_seconds=-1)


def test_resolve_subtitle_overlap_prefers_cli_then_series_then_default():
    assert resolve_subtitle_overlap_seconds(
        Namespace(subtitle_overlap_seconds=7),
        SimpleNamespace(subtitle_overlap_seconds=4),
    ) == 7

    assert resolve_subtitle_overlap_seconds(
        Namespace(),
        SimpleNamespace(subtitle_overlap_seconds=4),
    ) == 4

    assert resolve_subtitle_overlap_seconds(Namespace(), SimpleNamespace()) == 5


def test_segment_index_mapping_and_neighbors():
    mapped = map_segment_index_to_window_index(
        segment_index=3,
        segment_duration_seconds=30,
        stride_seconds=25,
    )
    assert mapped == 3
    assert neighbor_window_indexes(mapped) == [2, 3, 4]
    assert neighbor_window_indexes(mapped, radius=0) == [3]


def test_should_expand_to_neighbor_windows_uses_confidence_gate():
    assert should_expand_to_neighbor_windows([]) is True
    assert should_expand_to_neighbor_windows([0.12, 0.40], expansion_mode="two-stage") is False
    assert should_expand_to_neighbor_windows([0.30, 0.50], expansion_mode="two-stage") is False
    assert should_expand_to_neighbor_windows([0.30, 0.33], expansion_mode="two-stage") is True


def test_distance_with_window_penalty():
    assert distance_with_window_penalty(0.2, 3, 3) == 0.2
    assert distance_with_window_penalty(0.2, 4, 3) == pytest.approx(0.21)


def test_should_expand_to_neighbor_windows_default_is_always():
    assert should_expand_to_neighbor_windows([0.01, 0.8]) is True


def test_support_aware_score_rewards_support_windows():
    support_by_episode = {}
    merge_episode_window_hit(support_by_episode, "S01E01", 10, 0.3)
    merge_episode_window_hit(support_by_episode, "S01E01", 11, 0.31)
    score, nearest_offset = support_aware_score(
        support_by_episode["S01E01"],
        mapped_window_index=10,
    )
    assert nearest_offset == 0
    assert score < 0.3


def test_resolve_window_expansion_mode_and_low_info_defaults():
    args = Namespace()
    series = SimpleNamespace(
        window_expansion_mode="two-stage",
        low_info_filter=False,
        max_results_per_query=12,
    )
    assert resolve_window_expansion_mode(args, series) == "two-stage"
    assert resolve_low_info_filter(args, series) is False
    assert resolve_max_results_per_query(args, series) == 12
