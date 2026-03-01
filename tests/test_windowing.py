import pytest

from argparse import Namespace
from types import SimpleNamespace

from mkv_episode_matcher.windowing import (
    distance_with_window_penalty,
    make_window_config,
    map_segment_index_to_window_index,
    neighbor_window_indexes,
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


def test_should_expand_to_neighbor_windows_uses_confidence_gate():
    assert should_expand_to_neighbor_windows([]) is True
    assert should_expand_to_neighbor_windows([0.12, 0.40]) is False
    assert should_expand_to_neighbor_windows([0.30, 0.50]) is False
    assert should_expand_to_neighbor_windows([0.30, 0.33]) is True


def test_distance_with_window_penalty():
    assert distance_with_window_penalty(0.2, 3, 3) == 0.2
    assert distance_with_window_penalty(0.2, 4, 3) == pytest.approx(0.21)
