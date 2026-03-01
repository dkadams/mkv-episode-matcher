from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_SUBTITLE_OVERLAP_SECONDS = 5
DEFAULT_WINDOW_NEIGHBOR_RADIUS = 1
DEFAULT_WINDOW_EXPANSION_MODE = "always"
DEFAULT_WINDOW_OFFSET_DISTANCE_PENALTY = 0.01
DEFAULT_CONFIDENT_DISTANCE_THRESHOLD = 0.10
DEFAULT_CONFIDENT_MARGIN_THRESHOLD = 0.20
DEFAULT_LOW_INFO_FILTER = True
DEFAULT_LOW_INFO_MIN_WORDS = 8
DEFAULT_LOW_INFO_CUE_RATIO = 0.25
DEFAULT_MAX_RESULTS_PER_QUERY = 10
DEFAULT_SUPPORT_WINDOW_BONUS = 0.012
DEFAULT_SUPPORT_OFFSET_PENALTY = 0.003

WINDOW_EXPANSION_MODES = {"always", "two-stage"}


@dataclass(frozen=True)
class WindowConfig:
    window_seconds: int
    overlap_seconds: int
    stride_seconds: int
    profile_key: str


@dataclass
class EpisodeSupport:
    window_best_distances: dict[int, float]
    best_distance: float
    best_window_index: int


def resolve_subtitle_overlap_seconds(args: Any, series: Any, default: int = DEFAULT_SUBTITLE_OVERLAP_SECONDS) -> int:
    cli_overlap = getattr(args, "subtitle_overlap_seconds", None)
    if cli_overlap is not None:
        return int(cli_overlap)

    series_overlap = getattr(series, "subtitle_overlap_seconds", None)
    if series_overlap is not None:
        return int(series_overlap)

    return int(default)


def resolve_window_expansion_mode(
    args: Any,
    series: Any,
    default: str = DEFAULT_WINDOW_EXPANSION_MODE,
) -> str:
    cli_mode = getattr(args, "window_expansion_mode", None)
    if cli_mode is not None:
        mode = str(cli_mode)
    else:
        series_mode = getattr(series, "window_expansion_mode", None)
        mode = str(series_mode) if series_mode is not None else str(default)

    if mode not in WINDOW_EXPANSION_MODES:
        choices = ", ".join(sorted(WINDOW_EXPANSION_MODES))
        raise ValueError(f"window expansion mode must be one of: {choices}")
    return mode


def resolve_window_neighbor_radius(
    args: Any,
    series: Any,
    default: int = DEFAULT_WINDOW_NEIGHBOR_RADIUS,
) -> int:
    cli_radius = getattr(args, "window_neighbor_radius", None)
    if cli_radius is not None:
        radius = int(cli_radius)
    else:
        series_radius = getattr(series, "window_neighbor_radius", None)
        radius = int(series_radius) if series_radius is not None else int(default)
    if radius < 0:
        raise ValueError("window neighbor radius must be >= 0")
    return radius


def resolve_low_info_filter(
    args: Any,
    series: Any,
    default: bool = DEFAULT_LOW_INFO_FILTER,
) -> bool:
    cli_value = getattr(args, "low_info_filter", None)
    if cli_value is not None:
        return bool(cli_value)
    series_value = getattr(series, "low_info_filter", None)
    if series_value is not None:
        return bool(series_value)
    return bool(default)


def resolve_low_info_min_words(
    args: Any,
    series: Any,
    default: int = DEFAULT_LOW_INFO_MIN_WORDS,
) -> int:
    cli_value = getattr(args, "low_info_min_words", None)
    if cli_value is not None:
        value = int(cli_value)
    else:
        series_value = getattr(series, "low_info_min_words", None)
        value = int(series_value) if series_value is not None else int(default)
    if value < 1:
        raise ValueError("low-info min words must be >= 1")
    return value


def resolve_low_info_cue_ratio(
    args: Any,
    series: Any,
    default: float = DEFAULT_LOW_INFO_CUE_RATIO,
) -> float:
    cli_value = getattr(args, "low_info_cue_ratio", None)
    if cli_value is not None:
        value = float(cli_value)
    else:
        series_value = getattr(series, "low_info_cue_ratio", None)
        value = float(series_value) if series_value is not None else float(default)
    if value < 0.0 or value > 1.0:
        raise ValueError("low-info cue ratio must be within [0.0, 1.0]")
    return value


def resolve_max_results_per_query(
    args: Any,
    series: Any,
    default: int = DEFAULT_MAX_RESULTS_PER_QUERY,
) -> int:
    cli_value = getattr(args, "max_results_per_query", None)
    if cli_value is not None:
        value = int(cli_value)
    else:
        series_value = getattr(series, "max_results_per_query", None)
        value = int(series_value) if series_value is not None else int(default)
    if value < 1:
        raise ValueError("max results per query must be >= 1")
    return value


def validate_window_settings(window_seconds: int, overlap_seconds: int):
    if window_seconds <= 0:
        raise ValueError("segment/window duration must be > 0 seconds")
    if overlap_seconds < 0:
        raise ValueError("subtitle overlap must be >= 0 seconds")
    if overlap_seconds >= window_seconds:
        raise ValueError("subtitle overlap must be less than segment/window duration")


def make_window_config(window_seconds: int, overlap_seconds: int) -> WindowConfig:
    validate_window_settings(window_seconds, overlap_seconds)
    stride_seconds = window_seconds - overlap_seconds
    if stride_seconds < 1:
        raise ValueError("subtitle window stride must be at least 1 second")
    profile_key = f"w{window_seconds}_o{overlap_seconds}"
    return WindowConfig(
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        stride_seconds=stride_seconds,
        profile_key=profile_key,
    )


def map_segment_index_to_window_index(
    segment_index: int,
    segment_duration_seconds: int,
    stride_seconds: int,
) -> int:
    segment_start_seconds = int(segment_index) * int(segment_duration_seconds)
    return segment_start_seconds // int(stride_seconds)


def neighbor_window_indexes(mapped_window_index: int, radius: int = DEFAULT_WINDOW_NEIGHBOR_RADIUS) -> list[int]:
    if radius < 0:
        raise ValueError("window neighbor radius must be >= 0")
    return [idx for idx in range(mapped_window_index - radius, mapped_window_index + radius + 1) if idx >= 0]


def distance_with_window_penalty(
    raw_distance: float,
    window_index: int,
    mapped_window_index: int,
    penalty_per_window: float = DEFAULT_WINDOW_OFFSET_DISTANCE_PENALTY,
) -> float:
    return float(raw_distance) + abs(int(window_index) - int(mapped_window_index)) * float(penalty_per_window)


def should_expand_to_neighbor_windows(
    mapped_window_distances: list[float],
    expansion_mode: str = DEFAULT_WINDOW_EXPANSION_MODE,
    confident_distance_threshold: float = DEFAULT_CONFIDENT_DISTANCE_THRESHOLD,
    confident_margin_threshold: float = DEFAULT_CONFIDENT_MARGIN_THRESHOLD,
) -> bool:
    if expansion_mode == "always":
        return True
    if expansion_mode != "two-stage":
        choices = ", ".join(sorted(WINDOW_EXPANSION_MODES))
        raise ValueError(f"window expansion mode must be one of: {choices}")

    if not mapped_window_distances:
        return True

    ordered = sorted(float(distance) for distance in mapped_window_distances)
    best = ordered[0]
    if best <= confident_distance_threshold:
        return False

    if len(ordered) >= 2:
        margin = ordered[1] - best
        if margin >= confident_margin_threshold:
            return False

    return True


def merge_episode_window_hit(
    support_by_episode: dict[Any, EpisodeSupport],
    episode: Any,
    window_index: int,
    distance: float,
):
    existing = support_by_episode.get(episode)
    if existing is None:
        support_by_episode[episode] = EpisodeSupport(
            window_best_distances={int(window_index): float(distance)},
            best_distance=float(distance),
            best_window_index=int(window_index),
        )
        return

    per_window_best = existing.window_best_distances.get(int(window_index))
    if per_window_best is None or float(distance) < per_window_best:
        existing.window_best_distances[int(window_index)] = float(distance)

    if float(distance) < existing.best_distance or (
        float(distance) == existing.best_distance and int(window_index) < existing.best_window_index
    ):
        existing.best_distance = float(distance)
        existing.best_window_index = int(window_index)


def support_aware_score(
    support: EpisodeSupport,
    mapped_window_index: int,
    support_window_bonus: float = DEFAULT_SUPPORT_WINDOW_BONUS,
    support_offset_penalty: float = DEFAULT_SUPPORT_OFFSET_PENALTY,
) -> tuple[float, int]:
    support_windows = len(support.window_best_distances)
    nearest_window_offset = min(
        abs(int(window_idx) - int(mapped_window_index))
        for window_idx in support.window_best_distances.keys()
    )
    best_distance = min(support.window_best_distances.values())
    score = (
        float(best_distance)
        - float(support_window_bonus) * max(0, support_windows - 1)
        + float(support_offset_penalty) * nearest_window_offset
    )
    return score, nearest_window_offset
