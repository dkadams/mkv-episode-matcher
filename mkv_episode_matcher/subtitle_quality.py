from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pysubs2
from loguru import logger

from mkv_episode_matcher.episode import EpisodeKey

DEFAULT_SUBTITLE_QUALITY_ENABLED = True
DEFAULT_SUBTITLE_QUALITY_MAX_CANDIDATES = 3
DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MAX = 1.45
DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MIN = 0.55
DEFAULT_SUBTITLE_QUALITY_OVERLAP_CONTAINMENT_THRESHOLD = 0.35


@dataclass(frozen=True)
class SubtitleQualitySettings:
    enabled: bool = DEFAULT_SUBTITLE_QUALITY_ENABLED
    max_candidates: int = DEFAULT_SUBTITLE_QUALITY_MAX_CANDIDATES
    runtime_ratio_max: float = DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MAX
    runtime_ratio_min: float = DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MIN
    overlap_containment_threshold: float = DEFAULT_SUBTITLE_QUALITY_OVERLAP_CONTAINMENT_THRESHOLD


@dataclass(frozen=True)
class SubtitleSignalMetrics:
    runtime_minutes: float
    runtime_ratio: float | None
    runtime_hard_fail: bool
    line_count: int
    unique_line_count: int
    arrow_count: int
    arrow_ratio: float


def resolve_subtitle_quality_settings(args: Any, series: Any) -> SubtitleQualitySettings:
    enabled = _resolve_bool(
        getattr(args, "subtitle_quality", None),
        getattr(series, "subtitle_quality_enabled", None),
        DEFAULT_SUBTITLE_QUALITY_ENABLED,
    )
    max_candidates = _resolve_int(
        getattr(args, "subtitle_quality_max_candidates", None),
        getattr(series, "subtitle_quality_max_candidates", None),
        DEFAULT_SUBTITLE_QUALITY_MAX_CANDIDATES,
        minimum=1,
    )
    runtime_ratio_max = _resolve_float(
        getattr(args, "subtitle_quality_runtime_ratio_max", None),
        getattr(series, "subtitle_quality_runtime_ratio_max", None),
        DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MAX,
        minimum=1.0,
    )
    runtime_ratio_min = _resolve_float(
        getattr(args, "subtitle_quality_runtime_ratio_min", None),
        getattr(series, "subtitle_quality_runtime_ratio_min", None),
        DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MIN,
        minimum=0.0,
    )
    overlap_containment_threshold = _resolve_float(
        getattr(args, "subtitle_quality_overlap_containment_threshold", None),
        getattr(series, "subtitle_quality_overlap_containment_threshold", None),
        DEFAULT_SUBTITLE_QUALITY_OVERLAP_CONTAINMENT_THRESHOLD,
        minimum=0.0,
    )
    if runtime_ratio_min >= runtime_ratio_max:
        raise ValueError("subtitle quality runtime ratio min must be less than max")
    if overlap_containment_threshold > 1.0:
        raise ValueError("subtitle quality overlap containment threshold must be <= 1.0")
    return SubtitleQualitySettings(
        enabled=enabled,
        max_candidates=max_candidates,
        runtime_ratio_max=runtime_ratio_max,
        runtime_ratio_min=runtime_ratio_min,
        overlap_containment_threshold=overlap_containment_threshold,
    )


def quality_dir_for_subtitles(subtitles_dir: Path) -> Path:
    return subtitles_dir / "quality"


def quality_file_for_episode(subtitles_dir: Path, episode_key: EpisodeKey) -> Path:
    return quality_dir_for_subtitles(subtitles_dir) / f"{episode_key}.quality.json"


def append_quarantine_event(subtitles_dir: Path, event: dict[str, Any]):
    quality_dir = quality_dir_for_subtitles(subtitles_dir)
    quality_dir.mkdir(parents=True, exist_ok=True)
    quarantine_file = quality_dir / "quarantine.jsonl"
    with quarantine_file.open("a", encoding="utf-8") as out:
        out.write(json.dumps(event, ensure_ascii=False))
        out.write("\n")


def save_episode_quality(subtitles_dir: Path, episode_key: EpisodeKey, payload: dict[str, Any]):
    quality_file = quality_file_for_episode(subtitles_dir, episode_key)
    quality_file.parent.mkdir(parents=True, exist_ok=True)
    with quality_file.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)


def load_episode_quality(subtitles_dir: Path, episode_key: EpisodeKey) -> dict[str, Any] | None:
    quality_file = quality_file_for_episode(subtitles_dir, episode_key)
    if not quality_file.exists():
        return None
    try:
        with quality_file.open("r", encoding="utf-8") as in_:
            return json.load(in_)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load subtitle quality file: {}", quality_file)
        return None


def load_quarantined_episodes(subtitles_dir: Path) -> set[EpisodeKey]:
    quality_dir = quality_dir_for_subtitles(subtitles_dir)
    if not quality_dir.exists():
        return set()
    quarantined: set[EpisodeKey] = set()
    for quality_file in sorted(quality_dir.glob("S*E*.quality.json")):
        try:
            payload = json.loads(quality_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("verdict")) != "quarantined":
            continue
        episode_value = payload.get("episode")
        if not episode_value:
            continue
        episode_key = _parse_episode_key_string(str(episode_value))
        if episode_key:
            quarantined.add(episode_key)
    return quarantined


def subtitle_paths_by_episode(
    subtitles_dir: Path,
    *,
    include_quarantined: bool,
) -> dict[EpisodeKey, Path]:
    quarantined = load_quarantined_episodes(subtitles_dir)
    mapping: dict[EpisodeKey, Path] = {}
    for subtitle_path in sorted(subtitles_dir.rglob("*.srt")):
        try:
            episode_key = EpisodeKey.from_srt_path(subtitle_path)
        except ValueError:
            logger.warning("Skipping ambiguous subtitle filename: {}", subtitle_path)
            continue
        if not episode_key:
            continue
        if not include_quarantined and episode_key in quarantined:
            continue
        mapping[episode_key] = subtitle_path
    return mapping


def subtitle_file_entries(
    subtitles_dir: Path,
    *,
    include_quarantined: bool,
) -> list[tuple[EpisodeKey, Path]]:
    mapping = subtitle_paths_by_episode(
        subtitles_dir,
        include_quarantined=include_quarantined,
    )
    return sorted(mapping.items(), key=lambda row: row[0])


def evaluate_subtitle_signals(
    subtitle_path: Path,
    *,
    expected_runtime_minutes: float | None,
    settings: SubtitleQualitySettings,
) -> tuple[SubtitleSignalMetrics, set[str]]:
    subs = pysubs2.load(str(subtitle_path), format_="srt")
    line_texts = [str(sub.text or "") for sub in subs]
    normalized_lines = {
        _normalize_line_text(line)
        for line in line_texts
        if _normalize_line_text(line)
    }
    runtime_minutes = (
        float(max((int(sub.end) for sub in subs), default=0)) / 60000.0
    )
    runtime_ratio = None
    runtime_hard_fail = False
    if expected_runtime_minutes and expected_runtime_minutes > 0:
        runtime_ratio = runtime_minutes / float(expected_runtime_minutes)
        runtime_hard_fail = (
            runtime_ratio > float(settings.runtime_ratio_max)
            or runtime_ratio < float(settings.runtime_ratio_min)
        )
    arrow_count = sum(text.count(">>") for text in line_texts)
    line_count = len(subs)
    arrow_ratio = float(arrow_count) / float(max(1, line_count))
    metrics = SubtitleSignalMetrics(
        runtime_minutes=float(runtime_minutes),
        runtime_ratio=(float(runtime_ratio) if runtime_ratio is not None else None),
        runtime_hard_fail=bool(runtime_hard_fail),
        line_count=line_count,
        unique_line_count=len(normalized_lines),
        arrow_count=int(arrow_count),
        arrow_ratio=float(arrow_ratio),
    )
    return metrics, normalized_lines


def candidate_quality_diagnostics(
    *,
    episode_key: EpisodeKey,
    candidate_metrics: SubtitleSignalMetrics,
    candidate_lines: set[str],
    neighbor_metrics_by_episode: dict[EpisodeKey, SubtitleSignalMetrics],
    neighbor_lines_by_episode: dict[EpisodeKey, set[str]],
    settings: SubtitleQualitySettings,
    metadata_episode_mismatch: bool,
) -> dict[str, Any]:
    hard_fail_reasons: list[str] = []
    soft_penalties: list[dict[str, Any]] = []

    if metadata_episode_mismatch:
        hard_fail_reasons.append("metadata_episode_mismatch")

    runtime_ratio = candidate_metrics.runtime_ratio
    if runtime_ratio is not None:
        if candidate_metrics.runtime_hard_fail:
            hard_fail_reasons.append("runtime_ratio_out_of_bounds")
        elif abs(float(runtime_ratio) - 1.0) > 0.15:
            soft_penalties.append(
                {
                    "reason": "runtime_ratio_soft_deviation",
                    "value": abs(float(runtime_ratio) - 1.0),
                }
            )

    overlap_rows: list[dict[str, Any]] = []
    threshold = float(settings.overlap_containment_threshold)
    for neighbor_key in sorted(neighbor_metrics_by_episode.keys()):
        neighbor_lines = neighbor_lines_by_episode.get(neighbor_key, set())
        if not neighbor_lines or not candidate_lines:
            continue
        containment_candidate_in_neighbor = containment_ratio(candidate_lines, neighbor_lines)
        containment_neighbor_in_candidate = containment_ratio(neighbor_lines, candidate_lines)
        suspicious = (
            containment_candidate_in_neighbor >= threshold
            or containment_neighbor_in_candidate >= threshold
        )
        overlap_rows.append(
            {
                "neighbor_episode": str(neighbor_key),
                "containment_candidate_in_neighbor": containment_candidate_in_neighbor,
                "containment_neighbor_in_candidate": containment_neighbor_in_candidate,
                "suspicious": suspicious,
                "neighbor_runtime_ratio": neighbor_metrics_by_episode[neighbor_key].runtime_ratio,
                "neighbor_runtime_hard_fail": neighbor_metrics_by_episode[neighbor_key].runtime_hard_fail,
            }
        )
        if not suspicious:
            continue

        neighbor_runtime_hard_fail = bool(neighbor_metrics_by_episode[neighbor_key].runtime_hard_fail)
        candidate_runtime_hard_fail = bool(candidate_metrics.runtime_hard_fail)
        if candidate_runtime_hard_fail and not neighbor_runtime_hard_fail:
            hard_fail_reasons.append("overlap_with_neighbor_runtime_pass")
        elif candidate_runtime_hard_fail and neighbor_runtime_hard_fail:
            hard_fail_reasons.append("overlap_both_runtime_hard_fail")
        elif not candidate_runtime_hard_fail and not neighbor_runtime_hard_fail:
            soft_penalties.append(
                {
                    "reason": "overlap_with_neighbor_runtime_pass",
                    "value": max(
                        containment_candidate_in_neighbor,
                        containment_neighbor_in_candidate,
                    ),
                    "neighbor_episode": str(neighbor_key),
                }
            )

    soft_penalty_total = float(sum(float(row.get("value", 0.0)) for row in soft_penalties))
    hard_fail_reasons = sorted(set(hard_fail_reasons))
    return {
        "episode": str(episode_key),
        "hard_fail_reasons": hard_fail_reasons,
        "hard_fail": bool(hard_fail_reasons),
        "soft_penalties": soft_penalties,
        "soft_penalty_total": soft_penalty_total,
        "metrics": {
            "runtime_minutes": candidate_metrics.runtime_minutes,
            "runtime_ratio": candidate_metrics.runtime_ratio,
            "runtime_hard_fail": candidate_metrics.runtime_hard_fail,
            "line_count": candidate_metrics.line_count,
            "unique_line_count": candidate_metrics.unique_line_count,
            "arrow_count": candidate_metrics.arrow_count,
            "arrow_ratio": candidate_metrics.arrow_ratio,
            "neighbor_overlap": overlap_rows,
        },
    }


def metadata_penalty(
    *,
    candidate: dict[str, Any],
    max_downloads: float,
    max_votes: float,
    max_ratings: float,
) -> float:
    downloads = _as_float(candidate.get("download_count")) + _as_float(candidate.get("new_download_count"))
    votes = _as_float(candidate.get("votes"))
    ratings = _as_float(candidate.get("ratings"))
    trusted = bool(candidate.get("from_trusted", False))

    downloads_norm = downloads / max_downloads if max_downloads > 0 else 0.0
    votes_norm = votes / max_votes if max_votes > 0 else 0.0
    ratings_norm = ratings / max_ratings if max_ratings > 0 else 0.0
    trusted_norm = 1.0 if trusted else 0.0

    strength = (
        0.40 * downloads_norm
        + 0.25 * votes_norm
        + 0.25 * ratings_norm
        + 0.10 * trusted_norm
    )
    return float(max(0.0, 1.0 - strength))


def choose_best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not attempts:
        return None

    max_downloads = max(
        (
            _as_float(attempt.get("subtitle", {}).get("download_count"))
            + _as_float(attempt.get("subtitle", {}).get("new_download_count"))
            for attempt in attempts
        ),
        default=0.0,
    )
    max_votes = max(
        (_as_float(attempt.get("subtitle", {}).get("votes")) for attempt in attempts),
        default=0.0,
    )
    max_ratings = max(
        (_as_float(attempt.get("subtitle", {}).get("ratings")) for attempt in attempts),
        default=0.0,
    )

    for attempt in attempts:
        diagnostics = attempt.get("quality", {})
        subtitle_info = attempt.get("subtitle", {})
        attempt["metadata_penalty"] = metadata_penalty(
            candidate=subtitle_info,
            max_downloads=max_downloads,
            max_votes=max_votes,
            max_ratings=max_ratings,
        )
        rank_penalty = 0.01 * float(max(0, int(attempt.get("rank", 0)) - 1))
        attempt["candidate_score"] = (
            float(diagnostics.get("soft_penalty_total", 0.0))
            + float(attempt["metadata_penalty"])
            + rank_penalty
        )

    passing = [row for row in attempts if not bool(row.get("quality", {}).get("hard_fail", False))]
    rows = passing if passing else attempts
    rows.sort(key=lambda row: (float(row.get("candidate_score", math.inf)), int(row.get("rank", 0))))
    selected = rows[0]
    selected["selection_verdict"] = "pass" if passing else "quarantined"
    return selected


def containment_ratio(source_lines: set[str], target_lines: set[str]) -> float:
    if not source_lines:
        return 0.0
    return float(len(source_lines & target_lines)) / float(len(source_lines))


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_episode_key_string(value: str) -> EpisodeKey | None:
    match = re.match(r"^S(\d+)E(\d+)$", str(value))
    if not match:
        return None
    return EpisodeKey(int(match.group(1)), int(match.group(2)))


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_bool(cli_value: Any, series_value: Any, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if series_value is not None:
        return bool(series_value)
    return bool(default)


def _resolve_int(cli_value: Any, series_value: Any, default: int, *, minimum: int) -> int:
    value = int(cli_value if cli_value is not None else (series_value if series_value is not None else default))
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _resolve_float(cli_value: Any, series_value: Any, default: float, *, minimum: float) -> float:
    value = float(cli_value if cli_value is not None else (series_value if series_value is not None else default))
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _normalize_line_text(text: str) -> str:
    line = text.replace("\\N", " ")
    line = re.sub(r"<[^>]+>", " ", line)
    line = line.lower().strip()
    line = re.sub(r"\s+", " ", line)
    return line
