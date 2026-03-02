import numpy as np

from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.multi_episode_assignment import (
    MultiEpisodeDetection,
    MultiEpisodeSettings,
    RuntimeProfile,
    build_runtime_profile,
    detect_multi_episode_candidate,
    resolve_multi_episode_assignment,
)


def test_build_runtime_profile_regular_prefers_dominant_mode():
    tmdb = {
        EpisodeKey(1, 1): 22.0,
        EpisodeKey(1, 2): 23.0,
        EpisodeKey(1, 3): 22.0,
        EpisodeKey(1, 4): 22.0,
    }
    srt = {
        EpisodeKey(1, 1): 21.5,
        EpisodeKey(1, 2): 22.5,
        EpisodeKey(1, 3): 21.8,
        EpisodeKey(1, 4): 22.1,
    }
    profile = build_runtime_profile(tmdb, srt)
    assert profile.profile_type == "regular"
    assert profile.expected_single_minutes == 22.0


def test_build_runtime_profile_bimodal_detects_2x_pattern():
    tmdb = {
        EpisodeKey(1, 1): 22.0,
        EpisodeKey(1, 2): 22.0,
        EpisodeKey(1, 3): 44.0,
        EpisodeKey(1, 4): 44.0,
    }
    profile = build_runtime_profile(tmdb, {})
    assert profile.profile_type == "bimodal"
    assert profile.expected_single_minutes == 22.0
    assert profile.long_mode_minutes == 44.0


def test_detect_multi_episode_candidate_requires_both_signals():
    settings = MultiEpisodeSettings()
    profile = RuntimeProfile(profile_type="regular", expected_single_minutes=22.0)
    detected = detect_multi_episode_candidate(
        settings=settings,
        profile=profile,
        video_minutes=44.0,
        observed_segments=30,
        segments_per_minute=0.5,
    )
    assert detected.is_multi_candidate is True

    not_detected = detect_multi_episode_candidate(
        settings=settings,
        profile=profile,
        video_minutes=44.0,
        observed_segments=12,
        segments_per_minute=0.5,
    )
    assert not_detected.is_multi_candidate is False


def test_resolve_multi_episode_assignment_selects_consecutive_pair():
    settings = MultiEpisodeSettings(mode="force-2", pair_margin=0.0)
    detection = MultiEpisodeDetection(
        is_multi_candidate=True,
        reasons=("mode_force_2",),
        confidence=1.0,
        runtime_profile_type="regular",
        expected_single_minutes=22.0,
    )
    rows = [(idx, np.array([1.0, 0.0], dtype=np.float32)) for idx in range(88)]

    def query_segment(embedding, mapped_window, neighbor_radius, max_results_per_query):
        # Multi resolver uses a wider radius for second-half chunks.
        if neighbor_radius > 1:
            return [
                (EpisodeKey(2, 13), 0.05),
                (EpisodeKey(3, 1), 0.20),
            ]
        return [
            (EpisodeKey(2, 12), 0.05),
            (EpisodeKey(5, 6), 0.20),
        ]

    assignment = resolve_multi_episode_assignment(
        settings=settings,
        detection=detection,
        segment_rows=rows,
        video_minutes=44.0,
        segment_duration_seconds=30,
        stride_seconds=25,
        window_seconds=30,
        base_neighbor_radius=1,
        max_results_per_query=10,
        query_segment=query_segment,
        fallback_single_episode=EpisodeKey(2, 12),
    )

    assert assignment.assignment_mode == "multi_2"
    assert assignment.assigned_episodes == (EpisodeKey(2, 12), EpisodeKey(2, 13))


def test_resolve_multi_episode_assignment_ambiguous_margin_captures_diagnostics():
    settings = MultiEpisodeSettings(mode="force-2", pair_margin=0.05)
    detection = MultiEpisodeDetection(
        is_multi_candidate=True,
        reasons=("mode_force_2",),
        confidence=1.0,
        runtime_profile_type="regular",
        expected_single_minutes=22.0,
    )
    rows = [(idx, np.array([1.0, 0.0], dtype=np.float32)) for idx in range(88)]

    def query_segment(embedding, mapped_window, neighbor_radius, max_results_per_query):
        if neighbor_radius > 1:
            return [
                (EpisodeKey(1, 2), 0.20),
                (EpisodeKey(1, 3), 0.25),
            ]
        return [
            (EpisodeKey(1, 1), 0.20),
            (EpisodeKey(1, 2), 0.25),
        ]

    assignment = resolve_multi_episode_assignment(
        settings=settings,
        detection=detection,
        segment_rows=rows,
        video_minutes=44.0,
        segment_duration_seconds=30,
        stride_seconds=25,
        window_seconds=30,
        base_neighbor_radius=1,
        max_results_per_query=10,
        query_segment=query_segment,
        fallback_single_episode=EpisodeKey(1, 1),
    )

    assert assignment.assignment_mode == "single"
    assert assignment.assigned_episodes == (EpisodeKey(1, 1),)
    diagnostics = assignment.diagnostics
    assert diagnostics["reason"] == "ambiguous_pair_margin"
    assert diagnostics["pair_margin_threshold"] == 0.05
    assert diagnostics["margin"] < 0.05
    assert diagnostics["split_evaluated"]
    assert diagnostics["split_candidates"]
    assert diagnostics["chunk1_top"]
    assert diagnostics["chunk2_top"]
