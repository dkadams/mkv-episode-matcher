from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pysubs2

from mkv_episode_matcher.episode import EpisodeKey


DEFAULT_MULTI_EPISODE_MODE = "auto"
DEFAULT_MULTI_EPISODE_DURATION_RATIO_THRESHOLD = 1.70
DEFAULT_MULTI_EPISODE_SEGMENTS_RATIO_THRESHOLD = 1.70
DEFAULT_MULTI_EPISODE_MIN_EXTRA_MINUTES = 10.0
DEFAULT_MULTI_EPISODE_MIN_EXTRA_SEGMENTS = 6
DEFAULT_MULTI_EPISODE_SPLIT_SEARCH_WINDOW_SECONDS = 180
DEFAULT_MULTI_EPISODE_MIN_SIDE_SEGMENTS = 4
DEFAULT_MULTI_EPISODE_CANDIDATE_K = 10
DEFAULT_MULTI_EPISODE_CANDIDATE_K_RETRY = 20
DEFAULT_MULTI_EPISODE_SECOND_HALF_HORIZON_MULTIPLIER = 1.5
DEFAULT_MULTI_EPISODE_PAIR_MARGIN = 0.05
DEFAULT_MULTI_EPISODE_MISS_PENALTY = 1.20

MULTI_EPISODE_MODES = {"auto", "off", "force-2"}


@dataclass(frozen=True)
class MultiEpisodeSettings:
    mode: str = DEFAULT_MULTI_EPISODE_MODE
    duration_ratio_threshold: float = DEFAULT_MULTI_EPISODE_DURATION_RATIO_THRESHOLD
    segments_ratio_threshold: float = DEFAULT_MULTI_EPISODE_SEGMENTS_RATIO_THRESHOLD
    min_extra_minutes: float = DEFAULT_MULTI_EPISODE_MIN_EXTRA_MINUTES
    min_extra_segments: int = DEFAULT_MULTI_EPISODE_MIN_EXTRA_SEGMENTS
    split_search_window_seconds: int = DEFAULT_MULTI_EPISODE_SPLIT_SEARCH_WINDOW_SECONDS
    min_side_segments: int = DEFAULT_MULTI_EPISODE_MIN_SIDE_SEGMENTS
    candidate_k: int = DEFAULT_MULTI_EPISODE_CANDIDATE_K
    candidate_k_retry: int = DEFAULT_MULTI_EPISODE_CANDIDATE_K_RETRY
    second_half_horizon_multiplier: float = (
        DEFAULT_MULTI_EPISODE_SECOND_HALF_HORIZON_MULTIPLIER
    )
    pair_margin: float = DEFAULT_MULTI_EPISODE_PAIR_MARGIN
    miss_penalty: float = DEFAULT_MULTI_EPISODE_MISS_PENALTY


@dataclass(frozen=True)
class RuntimeProfile:
    profile_type: str
    expected_single_minutes: float
    long_mode_minutes: float | None = None
    sample_count: int = 0
    dominant_coverage: float = 0.0


@dataclass(frozen=True)
class MultiEpisodeDetection:
    is_multi_candidate: bool
    reasons: tuple[str, ...]
    confidence: float
    runtime_profile_type: str
    expected_single_minutes: float


@dataclass(frozen=True)
class EpisodeLoss:
    episode: EpisodeKey
    mean_loss: float
    hit_ratio: float


@dataclass(frozen=True)
class MultiEpisodeAssignment:
    assignment_mode: str
    assigned_episodes: tuple[EpisodeKey, ...]
    split_seconds: int | None
    assignment_confidence: float
    detector_reasons: tuple[str, ...]
    runtime_profile_type: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def resolve_multi_episode_settings(
    args: Any,
    series: Any,
    default: MultiEpisodeSettings = MultiEpisodeSettings(),
) -> MultiEpisodeSettings:
    mode = _resolve_mode(
        getattr(args, "multi_episode_mode", None),
        getattr(series, "multi_episode_mode", None),
        default.mode,
    )
    return MultiEpisodeSettings(
        mode=mode,
        duration_ratio_threshold=_resolve_float(
            getattr(args, "multi_episode_duration_ratio_threshold", None),
            getattr(series, "multi_episode_duration_ratio_threshold", None),
            default.duration_ratio_threshold,
            min_value=1.0,
        ),
        segments_ratio_threshold=_resolve_float(
            getattr(args, "multi_episode_segments_ratio_threshold", None),
            getattr(series, "multi_episode_segments_ratio_threshold", None),
            default.segments_ratio_threshold,
            min_value=1.0,
        ),
        min_extra_minutes=_resolve_float(
            getattr(args, "multi_episode_min_extra_minutes", None),
            getattr(series, "multi_episode_min_extra_minutes", None),
            default.min_extra_minutes,
            min_value=0.0,
        ),
        min_extra_segments=_resolve_int(
            getattr(args, "multi_episode_min_extra_segments", None),
            getattr(series, "multi_episode_min_extra_segments", None),
            default.min_extra_segments,
            min_value=0,
        ),
        split_search_window_seconds=_resolve_int(
            getattr(args, "multi_episode_split_search_window_seconds", None),
            getattr(series, "multi_episode_split_search_window_seconds", None),
            default.split_search_window_seconds,
            min_value=0,
        ),
        min_side_segments=_resolve_int(
            getattr(args, "multi_episode_min_side_segments", None),
            getattr(series, "multi_episode_min_side_segments", None),
            default.min_side_segments,
            min_value=1,
        ),
        candidate_k=_resolve_int(
            getattr(args, "multi_episode_candidate_k", None),
            getattr(series, "multi_episode_candidate_k", None),
            default.candidate_k,
            min_value=1,
        ),
        candidate_k_retry=_resolve_int(
            getattr(args, "multi_episode_candidate_k_retry", None),
            getattr(series, "multi_episode_candidate_k_retry", None),
            default.candidate_k_retry,
            min_value=1,
        ),
        second_half_horizon_multiplier=_resolve_float(
            getattr(args, "multi_episode_second_half_horizon_multiplier", None),
            getattr(series, "multi_episode_second_half_horizon_multiplier", None),
            default.second_half_horizon_multiplier,
            min_value=1.0,
        ),
        pair_margin=_resolve_float(
            getattr(args, "multi_episode_pair_margin", None),
            getattr(series, "multi_episode_pair_margin", None),
            default.pair_margin,
            min_value=0.0,
        ),
        miss_penalty=_resolve_float(
            getattr(args, "multi_episode_miss_penalty", None),
            getattr(series, "multi_episode_miss_penalty", None),
            default.miss_penalty,
            min_value=0.0,
        ),
    )


def collect_tmdb_episode_runtimes_minutes(series) -> dict[EpisodeKey, float]:
    runtimes: dict[EpisodeKey, float] = {}
    for key, season_detail in series.detail.items():
        if not str(key).startswith("season/"):
            continue
        season_number = season_detail.get("season_number")
        if season_number is None:
            continue
        for episode in season_detail.get("episodes", []):
            episode_number = episode.get("episode_number")
            runtime = episode.get("runtime")
            if episode_number is None or runtime in (None, "", 0):
                continue
            try:
                runtime_value = float(runtime)
            except (TypeError, ValueError):
                continue
            if runtime_value <= 0:
                continue
            runtimes[EpisodeKey(int(season_number), int(episode_number))] = runtime_value
    return runtimes


def collect_srt_episode_runtimes_minutes(subtitles_dir: Path) -> dict[EpisodeKey, float]:
    durations: dict[EpisodeKey, float] = {}
    if not subtitles_dir.exists():
        return durations
    for srt_path in subtitles_dir.rglob("*.srt"):
        episode = EpisodeKey.from_srt_path(srt_path)
        if not episode:
            continue
        try:
            subs = pysubs2.load(str(srt_path), format_="srt")
        except Exception:
            continue
        if not subs:
            continue
        max_end_ms = max(int(sub.end) for sub in subs)
        if max_end_ms <= 0:
            continue
        durations[episode] = max_end_ms / 60000.0
    return durations


def build_runtime_profile(
    tmdb_minutes_by_episode: dict[EpisodeKey, float],
    srt_minutes_by_episode: dict[EpisodeKey, float],
) -> RuntimeProfile:
    keys = set(tmdb_minutes_by_episode) | set(srt_minutes_by_episode)
    canonical_values: list[float] = []
    for episode in keys:
        tmdb_value = tmdb_minutes_by_episode.get(episode)
        srt_value = srt_minutes_by_episode.get(episode)
        if tmdb_value is not None and srt_value is not None:
            canonical_values.append(min(float(tmdb_value), float(srt_value)))
        elif tmdb_value is not None:
            canonical_values.append(float(tmdb_value))
        elif srt_value is not None:
            canonical_values.append(float(srt_value))

    if not canonical_values:
        return RuntimeProfile(
            profile_type="irregular",
            expected_single_minutes=22.0,
            sample_count=0,
        )

    canonical_values.sort()
    sample_count = len(canonical_values)
    rounded = [int(round(value)) for value in canonical_values]
    mode_counter = Counter(rounded)
    (mode_a, count_a), *rest = mode_counter.most_common(2)
    dominant_coverage = count_a / sample_count
    p10 = _percentile(canonical_values, 10.0)
    p90 = _percentile(canonical_values, 90.0)
    spread = p90 - p10

    if dominant_coverage >= 0.75 and spread <= 6.0:
        return RuntimeProfile(
            profile_type="regular",
            expected_single_minutes=float(mode_a),
            sample_count=sample_count,
            dominant_coverage=dominant_coverage,
        )

    if rest:
        mode_b, count_b = rest[0]
        small = float(min(mode_a, mode_b))
        large = float(max(mode_a, mode_b))
        ratio = large / small if small > 0 else 0.0
        coverage_small = mode_counter[int(small)] / sample_count
        coverage_large = mode_counter[int(large)] / sample_count
        if 1.8 <= ratio <= 2.2 and coverage_small >= 0.15 and coverage_large >= 0.15:
            return RuntimeProfile(
                profile_type="bimodal",
                expected_single_minutes=small,
                long_mode_minutes=large,
                sample_count=sample_count,
                dominant_coverage=dominant_coverage,
            )

    return RuntimeProfile(
        profile_type="irregular",
        expected_single_minutes=float(_percentile(canonical_values, 50.0)),
        sample_count=sample_count,
        dominant_coverage=dominant_coverage,
    )


def detect_multi_episode_candidate(
    *,
    settings: MultiEpisodeSettings,
    profile: RuntimeProfile,
    video_minutes: float,
    observed_segments: int,
    segments_per_minute: float,
) -> MultiEpisodeDetection:
    if settings.mode == "force-2":
        return MultiEpisodeDetection(
            is_multi_candidate=True,
            reasons=("mode_force_2",),
            confidence=1.0,
            runtime_profile_type=profile.profile_type,
            expected_single_minutes=float(profile.expected_single_minutes),
        )

    if settings.mode == "off":
        return MultiEpisodeDetection(
            is_multi_candidate=False,
            reasons=("mode_off",),
            confidence=1.0,
            runtime_profile_type=profile.profile_type,
            expected_single_minutes=float(profile.expected_single_minutes),
        )

    reasons: list[str] = []
    expected_minutes = max(1.0, float(profile.expected_single_minutes))
    duration_ratio = float(video_minutes) / expected_minutes
    expected_segments = max(1, int(math.ceil(expected_minutes * float(segments_per_minute))))
    segments_ratio = float(observed_segments) / float(expected_segments)
    extra_minutes = float(video_minutes) - expected_minutes
    extra_segments = int(observed_segments) - int(expected_segments)

    if duration_ratio >= settings.duration_ratio_threshold:
        reasons.append("duration_ratio")
    if segments_ratio >= settings.segments_ratio_threshold:
        reasons.append("segments_ratio")
    if extra_minutes >= settings.min_extra_minutes:
        reasons.append("extra_minutes")
    if extra_segments >= settings.min_extra_segments:
        reasons.append("extra_segments")

    candidate = len(reasons) == 4

    if candidate and profile.profile_type == "bimodal" and profile.long_mode_minutes:
        # In bimodal shows, long episodes are often native and not merged doubles.
        close_threshold = max(4.0, 0.10 * float(profile.long_mode_minutes))
        if abs(float(video_minutes) - float(profile.long_mode_minutes)) <= close_threshold:
            candidate = False
            reasons.append("bimodal_native_long_runtime")

    if candidate and profile.profile_type == "irregular":
        # Fail closed unless the signal is stronger in irregular series.
        stronger_duration = duration_ratio >= (settings.duration_ratio_threshold + 0.20)
        stronger_segments = segments_ratio >= (settings.segments_ratio_threshold + 0.20)
        if not (stronger_duration and stronger_segments):
            candidate = False
            reasons.append("irregular_profile_guard")

    confidence = 0.0
    if candidate:
        confidence = min(
            1.0,
            0.5
            + max(0.0, duration_ratio - settings.duration_ratio_threshold)
            + max(0.0, segments_ratio - settings.segments_ratio_threshold),
        )

    return MultiEpisodeDetection(
        is_multi_candidate=candidate,
        reasons=tuple(reasons) if reasons else ("insufficient_signal",),
        confidence=confidence,
        runtime_profile_type=profile.profile_type,
        expected_single_minutes=expected_minutes,
    )


def candidate_split_seconds(
    *,
    video_minutes: float,
    expected_single_minutes: float,
    segment_duration_seconds: int,
    search_window_seconds: int,
    min_side_segments: int,
) -> list[int]:
    total_seconds = max(1, int(round(float(video_minutes) * 60.0)))
    expected_split = int(round(float(expected_single_minutes) * 60.0))
    expected_split = _snap_to_segment(expected_split, segment_duration_seconds)
    midpoint = _snap_to_segment(total_seconds // 2, segment_duration_seconds)

    candidates = {
        expected_split,
        midpoint,
    }
    step = max(1, int(segment_duration_seconds))
    for delta in range(step, int(search_window_seconds) + 1, step):
        candidates.add(_snap_to_segment(expected_split - delta, segment_duration_seconds))
        candidates.add(_snap_to_segment(expected_split + delta, segment_duration_seconds))

    min_side_seconds = int(min_side_segments) * int(segment_duration_seconds)
    bounded = [
        split
        for split in sorted(candidates)
        if min_side_seconds <= split <= (total_seconds - min_side_seconds)
    ]
    return bounded


def second_half_radius(
    *,
    base_radius: int,
    window_seconds: int,
    stride_seconds: int,
    horizon_multiplier: float,
) -> int:
    expanded_horizon = float(window_seconds) * float(horizon_multiplier)
    hops = int(math.ceil(expanded_horizon / float(stride_seconds)))
    return max(int(base_radius), hops)


def resolve_multi_episode_assignment(
    *,
    settings: MultiEpisodeSettings,
    detection: MultiEpisodeDetection,
    segment_rows: list[tuple[int, np.ndarray]],
    video_minutes: float | None,
    segment_duration_seconds: int,
    stride_seconds: int,
    window_seconds: int,
    base_neighbor_radius: int,
    max_results_per_query: int,
    query_segment: Callable[[np.ndarray, int, int, int], list[tuple[EpisodeKey, float]]],
    fallback_single_episode: EpisodeKey | None,
) -> MultiEpisodeAssignment:
    if not segment_rows:
        episodes = (fallback_single_episode,) if fallback_single_episode else tuple()
        return MultiEpisodeAssignment(
            assignment_mode="single",
            assigned_episodes=episodes,
            split_seconds=None,
            assignment_confidence=0.0,
            detector_reasons=detection.reasons,
            runtime_profile_type=detection.runtime_profile_type,
            diagnostics={"reason": "no_segments"},
        )

    should_try_multi = settings.mode == "force-2" or detection.is_multi_candidate
    if not should_try_multi:
        episodes = (fallback_single_episode,) if fallback_single_episode else tuple()
        return MultiEpisodeAssignment(
            assignment_mode="single",
            assigned_episodes=episodes,
            split_seconds=None,
            assignment_confidence=max(0.0, min(1.0, 1.0 - detection.confidence)),
            detector_reasons=detection.reasons,
            runtime_profile_type=detection.runtime_profile_type,
            diagnostics={"reason": "detector_single_path"},
        )

    splits = candidate_split_seconds(
        video_minutes=(
            float(video_minutes)
            if video_minutes is not None
            else (len(segment_rows) * segment_duration_seconds) / 60.0
        ),
        expected_single_minutes=detection.expected_single_minutes,
        segment_duration_seconds=segment_duration_seconds,
        search_window_seconds=settings.split_search_window_seconds,
        min_side_segments=settings.min_side_segments,
    )
    if not splits:
        episodes = (fallback_single_episode,) if fallback_single_episode else tuple()
        return MultiEpisodeAssignment(
            assignment_mode="single",
            assigned_episodes=episodes,
            split_seconds=None,
            assignment_confidence=0.0,
            detector_reasons=detection.reasons,
            runtime_profile_type=detection.runtime_profile_type,
            diagnostics={"reason": "no_valid_split_candidates"},
        )

    second_radius = second_half_radius(
        base_radius=base_neighbor_radius,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        horizon_multiplier=settings.second_half_horizon_multiplier,
    )

    split_candidates: list[dict[str, Any]] = []
    for split in splits:
        split_result = _score_split_candidate(
            split_seconds=int(split),
            segment_rows=segment_rows,
            segment_duration_seconds=segment_duration_seconds,
            stride_seconds=stride_seconds,
            chunk1_radius=int(base_neighbor_radius),
            chunk2_radius=int(second_radius),
            settings=settings,
            max_results_per_query=max_results_per_query,
            query_segment=query_segment,
        )
        if split_result:
            split_candidates.append(split_result)

    if not split_candidates:
        episodes = (fallback_single_episode,) if fallback_single_episode else tuple()
        return MultiEpisodeAssignment(
            assignment_mode="single",
            assigned_episodes=episodes,
            split_seconds=None,
            assignment_confidence=0.0,
            detector_reasons=detection.reasons,
            runtime_profile_type=detection.runtime_profile_type,
            diagnostics={"reason": "no_viable_consecutive_pair"},
        )

    split_candidates.sort(key=lambda row: row["pair_score"])
    best = split_candidates[0]
    second = split_candidates[1] if len(split_candidates) > 1 else None
    margin = (
        float(second["pair_score"] - best["pair_score"])
        if second is not None
        else float("inf")
    )
    if margin < float(settings.pair_margin):
        episodes = (fallback_single_episode,) if fallback_single_episode else tuple()
        return MultiEpisodeAssignment(
            assignment_mode="single",
            assigned_episodes=episodes,
            split_seconds=None,
            assignment_confidence=0.0,
            detector_reasons=detection.reasons,
            runtime_profile_type=detection.runtime_profile_type,
            diagnostics={
                "reason": "ambiguous_pair_margin",
                "best_pair_score": best["pair_score"],
                "second_pair_score": second["pair_score"] if second else None,
                "margin": margin,
                "pair_margin_threshold": float(settings.pair_margin),
                "best_split_seconds": int(best["split_seconds"]),
                "best_pair": [str(best["pair"][0]), str(best["pair"][1])],
                "second_split_seconds": int(second["split_seconds"]) if second else None,
                "second_pair": (
                    [str(second["pair"][0]), str(second["pair"][1])]
                    if second is not None
                    else None
                ),
                "chunk1_top": _episode_loss_rows(best["chunk1_ranked"], limit=8),
                "chunk2_top": _episode_loss_rows(best["chunk2_ranked"], limit=8),
                "chunk1_segment_count": int(best["chunk1_segment_count"]),
                "chunk2_segment_count": int(best["chunk2_segment_count"]),
                "split_evaluated": [int(row["split_seconds"]) for row in split_candidates],
                "split_candidates": _split_candidate_rows(split_candidates, limit=8),
            },
        )

    assignment_confidence = min(1.0, 0.5 + margin)
    return MultiEpisodeAssignment(
        assignment_mode="multi_2",
        assigned_episodes=(best["pair"][0], best["pair"][1]),
        split_seconds=int(best["split_seconds"]),
        assignment_confidence=assignment_confidence,
        detector_reasons=detection.reasons,
        runtime_profile_type=detection.runtime_profile_type,
        diagnostics={
            "best_pair_score": best["pair_score"],
            "second_pair_score": second["pair_score"] if second else None,
            "margin": margin,
            "pair_margin_threshold": float(settings.pair_margin),
            "chunk1_top": [
                (str(loss.episode), loss.mean_loss, loss.hit_ratio)
                for loss in best["chunk1_ranked"][:5]
            ],
            "chunk2_top": [
                (str(loss.episode), loss.mean_loss, loss.hit_ratio)
                for loss in best["chunk2_ranked"][:5]
            ],
            "chunk1_segment_count": int(best["chunk1_segment_count"]),
            "chunk2_segment_count": int(best["chunk2_segment_count"]),
            "split_evaluated": [row["split_seconds"] for row in split_candidates],
            "split_candidates": _split_candidate_rows(split_candidates, limit=8),
        },
    )


def is_consecutive_episode_pair(first: EpisodeKey, second: EpisodeKey) -> bool:
    return (
        first.season_number == second.season_number
        and second.episode_number == (first.episode_number + 1)
    )


def _score_split_candidate(
    *,
    split_seconds: int,
    segment_rows: list[tuple[int, np.ndarray]],
    segment_duration_seconds: int,
    stride_seconds: int,
    chunk1_radius: int,
    chunk2_radius: int,
    settings: MultiEpisodeSettings,
    max_results_per_query: int,
    query_segment: Callable[[np.ndarray, int, int, int], list[tuple[EpisodeKey, float]]],
) -> dict[str, Any] | None:
    chunk_rows = {
        1: [(segment_index, embedding) for segment_index, embedding in segment_rows
            if (int(segment_index) * int(segment_duration_seconds)) < int(split_seconds)],
        2: [(segment_index, embedding) for segment_index, embedding in segment_rows
            if (int(segment_index) * int(segment_duration_seconds)) >= int(split_seconds)],
    }
    if len(chunk_rows[1]) < settings.min_side_segments or len(chunk_rows[2]) < settings.min_side_segments:
        return None

    chunk_losses = {
        1: _chunk_episode_losses(
            rows=chunk_rows[1],
            segment_duration_seconds=segment_duration_seconds,
            offset_seconds=0,
            stride_seconds=stride_seconds,
            neighbor_radius=chunk1_radius,
            miss_penalty=settings.miss_penalty,
            max_results_per_query=max_results_per_query,
            query_segment=query_segment,
        ),
        2: _chunk_episode_losses(
            rows=chunk_rows[2],
            segment_duration_seconds=segment_duration_seconds,
            offset_seconds=split_seconds,
            stride_seconds=stride_seconds,
            neighbor_radius=chunk2_radius,
            miss_penalty=settings.miss_penalty,
            max_results_per_query=max_results_per_query,
            query_segment=query_segment,
        ),
    }

    chunk1_ranked = sorted(chunk_losses[1].values(), key=lambda row: (row.mean_loss, -row.hit_ratio, row.episode))
    chunk2_ranked = sorted(chunk_losses[2].values(), key=lambda row: (row.mean_loss, -row.hit_ratio, row.episode))
    if not chunk1_ranked or not chunk2_ranked:
        return None

    pair = _select_best_consecutive_pair(
        chunk1_ranked=chunk1_ranked,
        chunk2_ranked=chunk2_ranked,
        first_k=settings.candidate_k,
    )
    if not pair:
        pair = _select_best_consecutive_pair(
            chunk1_ranked=chunk1_ranked,
            chunk2_ranked=chunk2_ranked,
            first_k=settings.candidate_k_retry,
        )
    if not pair:
        return None

    first_loss, second_loss = pair
    pair_score = float(first_loss.mean_loss + second_loss.mean_loss)
    return {
        "split_seconds": int(split_seconds),
        "pair": (first_loss.episode, second_loss.episode),
        "pair_score": pair_score,
        "chunk1_ranked": chunk1_ranked,
        "chunk2_ranked": chunk2_ranked,
        "chunk1_segment_count": int(len(chunk_rows[1])),
        "chunk2_segment_count": int(len(chunk_rows[2])),
    }


def _chunk_episode_losses(
    *,
    rows: list[tuple[int, np.ndarray]],
    segment_duration_seconds: int,
    offset_seconds: int,
    stride_seconds: int,
    neighbor_radius: int,
    miss_penalty: float,
    max_results_per_query: int,
    query_segment: Callable[[np.ndarray, int, int, int], list[tuple[EpisodeKey, float]]],
) -> dict[EpisodeKey, EpisodeLoss]:
    segment_count = len(rows)
    if segment_count == 0:
        return {}

    total_loss_by_episode: dict[EpisodeKey, float] = {}
    hit_count_by_episode: dict[EpisodeKey, int] = {}

    for segment_index, embedding in rows:
        segment_start = int(segment_index) * int(segment_duration_seconds)
        effective_seconds = max(0, segment_start - int(offset_seconds))
        mapped_window = int(effective_seconds // int(stride_seconds))
        hits = query_segment(
            embedding,
            mapped_window,
            int(neighbor_radius),
            int(max_results_per_query),
        )
        for episode, score in hits:
            if episode not in total_loss_by_episode:
                total_loss_by_episode[episode] = float(miss_penalty) * float(segment_count)
                hit_count_by_episode[episode] = 0
            total_loss_by_episode[episode] -= float(miss_penalty)
            total_loss_by_episode[episode] += float(score)
            hit_count_by_episode[episode] += 1

    losses = {
        episode: EpisodeLoss(
            episode=episode,
            mean_loss=float(total_loss / float(segment_count)),
            hit_ratio=float(hit_count_by_episode[episode]) / float(segment_count),
        )
        for episode, total_loss in total_loss_by_episode.items()
    }
    return losses


def _select_best_consecutive_pair(
    *,
    chunk1_ranked: list[EpisodeLoss],
    chunk2_ranked: list[EpisodeLoss],
    first_k: int,
) -> tuple[EpisodeLoss, EpisodeLoss] | None:
    first = chunk1_ranked[: max(1, int(first_k))]
    second = chunk2_ranked[: max(1, int(first_k))]
    pairs: list[tuple[EpisodeLoss, EpisodeLoss]] = []
    for first_loss in first:
        for second_loss in second:
            if is_consecutive_episode_pair(first_loss.episode, second_loss.episode):
                pairs.append((first_loss, second_loss))
    if not pairs:
        return None
    pairs.sort(
        key=lambda pair: (
            pair[0].mean_loss + pair[1].mean_loss,
            -(pair[0].hit_ratio + pair[1].hit_ratio),
            pair[0].episode,
            pair[1].episode,
        )
    )
    return pairs[0]


def _episode_loss_rows(losses: list[EpisodeLoss], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loss in losses[: max(1, int(limit))]:
        rows.append(
            {
                "episode": str(loss.episode),
                "mean_loss": float(loss.mean_loss),
                "hit_ratio": float(loss.hit_ratio),
            }
        )
    return rows


def _split_candidate_rows(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates[: max(1, int(limit))]:
        rows.append(
            {
                "split_seconds": int(candidate["split_seconds"]),
                "pair": [str(candidate["pair"][0]), str(candidate["pair"][1])],
                "pair_score": float(candidate["pair_score"]),
                "chunk1_segment_count": int(candidate["chunk1_segment_count"]),
                "chunk2_segment_count": int(candidate["chunk2_segment_count"]),
                "chunk1_top": _episode_loss_rows(candidate["chunk1_ranked"], limit=3),
                "chunk2_top": _episode_loss_rows(candidate["chunk2_ranked"], limit=3),
            }
        )
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    idx = (len(values) - 1) * (float(pct) / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _snap_to_segment(seconds: int, segment_duration_seconds: int) -> int:
    if segment_duration_seconds <= 0:
        return int(seconds)
    step = int(segment_duration_seconds)
    return max(0, int(round(seconds / step) * step))


def _resolve_mode(cli_value: Any, series_value: Any, default: str) -> str:
    value = str(cli_value if cli_value is not None else (series_value if series_value is not None else default))
    if value not in MULTI_EPISODE_MODES:
        choices = ", ".join(sorted(MULTI_EPISODE_MODES))
        raise ValueError(f"multi episode mode must be one of: {choices}")
    return value


def _resolve_int(cli_value: Any, series_value: Any, default: int, *, min_value: int) -> int:
    value = int(cli_value if cli_value is not None else (series_value if series_value is not None else default))
    if value < min_value:
        raise ValueError(f"value must be >= {min_value}")
    return value


def _resolve_float(cli_value: Any, series_value: Any, default: float, *, min_value: float) -> float:
    value = float(cli_value if cli_value is not None else (series_value if series_value is not None else default))
    if value < min_value:
        raise ValueError(f"value must be >= {min_value}")
    return value
