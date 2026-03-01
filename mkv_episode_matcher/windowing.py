from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_SUBTITLE_OVERLAP_SECONDS = 5
DEFAULT_WINDOW_NEIGHBOR_RADIUS = 1
DEFAULT_WINDOW_OFFSET_DISTANCE_PENALTY = 0.01
DEFAULT_CONFIDENT_DISTANCE_THRESHOLD = 0.10
DEFAULT_CONFIDENT_MARGIN_THRESHOLD = 0.20


@dataclass(frozen=True)
class WindowConfig:
    window_seconds: int
    overlap_seconds: int
    stride_seconds: int
    profile_key: str


def resolve_subtitle_overlap_seconds(args: Any, series: Any, default: int = DEFAULT_SUBTITLE_OVERLAP_SECONDS) -> int:
    cli_overlap = getattr(args, "subtitle_overlap_seconds", None)
    if cli_overlap is not None:
        return int(cli_overlap)

    series_overlap = getattr(series, "subtitle_overlap_seconds", None)
    if series_overlap is not None:
        return int(series_overlap)

    return int(default)


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
    confident_distance_threshold: float = DEFAULT_CONFIDENT_DISTANCE_THRESHOLD,
    confident_margin_threshold: float = DEFAULT_CONFIDENT_MARGIN_THRESHOLD,
) -> bool:
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
