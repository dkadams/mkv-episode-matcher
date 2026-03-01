from argparse import Namespace
from types import SimpleNamespace

import pytest

from mkv_episode_matcher.windowing import (
    make_window_config,
    map_segment_index_to_window_index,
    neighbor_window_indexes,
    resolve_subtitle_overlap_seconds,
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
