import csv
import html
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import hnswlib
import numpy as np
import pysubs2
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.embedding_model import SentenceTransformerModel
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.multi_episode_assignment import (
    build_runtime_profile,
    collect_srt_episode_runtimes_minutes,
    collect_tmdb_episode_runtimes_minutes,
    detect_multi_episode_candidate,
    resolve_multi_episode_assignment,
    resolve_multi_episode_settings,
)
from mkv_episode_matcher.segment_quality import analyze_segment_quality
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.subtitle_quality import subtitle_file_entries
from mkv_episode_matcher.subtitle_quality import load_quarantined_episodes
from mkv_episode_matcher.windowing import (
    DEFAULT_SUPPORT_OFFSET_PENALTY,
    DEFAULT_SUPPORT_WINDOW_BONUS,
    distance_with_window_penalty,
    make_window_config,
    merge_episode_window_hit,
    map_segment_index_to_window_index,
    neighbor_window_indexes,
    resolve_low_info_cue_ratio,
    resolve_low_info_filter,
    resolve_low_info_min_words,
    resolve_max_results_per_query,
    resolve_window_expansion_mode,
    resolve_window_neighbor_radius,
    should_expand_to_neighbor_windows,
    support_aware_score,
)

console = Console()

EPISODE_PATTERN = re.compile(r"^S(?P<season>\d+)E(?P<episode>\d+)$")


@dataclass(frozen=True)
class SegmentRecord:
    segment_index: int
    transcript_text: str
    expected: set[EpisodeKey]
    video_path: str
    variant_profile: str = "aligned"

@dataclass(frozen=True)
class DatasetPaths:
    manifest: Path
    subtitles_dir: Path
    transcriptions_dir: Path


@dataclass(frozen=True)
class IntervalSubtitleIndex:
    episodes: list[EpisodeKey]
    index: Any
    metadata: list["SubtitleWindowMetadata"] | None = None


@dataclass(frozen=True)
class SubtitleWindowMetadata:
    episode: EpisodeKey
    start_ms: int
    end_ms: int
    text: str
    subtitle_path: str


@dataclass(frozen=True)
class IntervalQueryHit:
    episode: EpisodeKey
    distance: float
    interval_index: int
    metadata: SubtitleWindowMetadata | None = None


def evaluate_dataset(config: Configuration):
    dataset_dir = Path(config.args.dataset_dir).expanduser().resolve()
    dataset_meta = _load_dataset_meta(dataset_dir)
    dataset_paths = _resolve_dataset_paths(dataset_dir)
    if not dataset_paths.manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_paths.manifest}")
    if not dataset_paths.subtitles_dir.exists():
        raise FileNotFoundError(f"Subtitles directory not found: {dataset_paths.subtitles_dir}")
    if not dataset_paths.transcriptions_dir.exists():
        raise FileNotFoundError(
            f"Transcriptions directory not found: {dataset_paths.transcriptions_dir}"
        )

    interval_seconds = _resolve_segment_duration(
        dataset_dir,
        config.args.segment_duration,
    )
    subtitle_overlap_seconds = _resolve_subtitle_overlap_seconds(
        dataset_dir,
        config.args.subtitle_overlap_seconds,
    )
    top_ks = sorted({k for k in config.args.top_k if k > 0})
    if not top_ks:
        raise ValueError("At least one positive --top-k value is required")
    if config.args.support_window_bonus < 0:
        raise ValueError("--support-window-bonus must be >= 0")
    if config.args.support_offset_penalty < 0:
        raise ValueError("--support-offset-penalty must be >= 0")
    settings = _resolve_matching_settings(dataset_meta, config.args)

    with Progress() as progress:
        records = list(_load_manifest(dataset_paths.manifest, limit=config.args.limit))
        records = _filter_records_by_profiles(records, config.args.profiles)
        load_task = progress.add_task("Loading transcription segments", total=len(records))
        segments = _load_segments_from_records(
            records,
            dataset_dir=dataset_dir,
            task_id=load_task,
            progress=progress,
        )

    if not segments:
        console.print("[orange1]No segments to evaluate.")
        return

    model = SentenceTransformerModel()
    with Progress() as progress:
        subtitle_indexes = _build_subtitle_indexes(
            dataset_paths.subtitles_dir,
            interval_seconds,
            subtitle_overlap_seconds,
            model,
            include_quarantined_subs=bool(config.args.include_quarantined_subs),
            progress=progress,
        )
        overall_report = _score_segments(
            segments=segments,
            subtitle_indexes=subtitle_indexes,
            model=model,
            top_ks=top_ks,
            max_failures=config.args.show_failures,
            segment_duration_seconds=interval_seconds,
            subtitle_overlap_seconds=subtitle_overlap_seconds,
            window_expansion_mode=settings["window_expansion_mode"],
            window_neighbor_radius=settings["window_neighbor_radius"],
            low_info_filter=settings["low_info_filter"],
            low_info_min_words=settings["low_info_min_words"],
            low_info_cue_ratio=settings["low_info_cue_ratio"],
            max_results_per_query=settings["max_results_per_query"],
            support_window_bonus=float(config.args.support_window_bonus),
            support_offset_penalty=float(config.args.support_offset_penalty),
            include_match_text=bool(config.args.include_match_text),
            progress=progress,
        )
        video_report = _score_video_assignments(
            records=records,
            dataset_dir=dataset_dir,
            subtitle_indexes=subtitle_indexes,
            model=model,
            segment_duration_seconds=interval_seconds,
            subtitle_overlap_seconds=subtitle_overlap_seconds,
            settings=settings,
            support_window_bonus=float(config.args.support_window_bonus),
            support_offset_penalty=float(config.args.support_offset_penalty),
            include_quarantined_subs=bool(config.args.include_quarantined_subs),
        )

    report = {
        "overall": overall_report,
        "video_level": video_report,
    }
    if config.args.report_by_profile:
        by_profile = {}
        for profile, profile_segments in _segments_by_profile(segments).items():
            by_profile[profile] = _score_segments(
                segments=profile_segments,
                subtitle_indexes=subtitle_indexes,
                model=model,
                top_ks=top_ks,
                max_failures=config.args.show_failures,
                segment_duration_seconds=interval_seconds,
                subtitle_overlap_seconds=subtitle_overlap_seconds,
                window_expansion_mode=settings["window_expansion_mode"],
                window_neighbor_radius=settings["window_neighbor_radius"],
                low_info_filter=settings["low_info_filter"],
                low_info_min_words=settings["low_info_min_words"],
                low_info_cue_ratio=settings["low_info_cue_ratio"],
                max_results_per_query=settings["max_results_per_query"],
                support_window_bonus=float(config.args.support_window_bonus),
                support_offset_penalty=float(config.args.support_offset_penalty),
                collect_mismatches=False,
                include_match_text=False,
            )
        report["by_profile"] = by_profile

    report.update(
        {
            "dataset_dir": str(dataset_dir),
            "segment_duration": interval_seconds,
            "subtitle_overlap_seconds": subtitle_overlap_seconds,
            "window_expansion_mode": settings["window_expansion_mode"],
            "window_neighbor_radius": settings["window_neighbor_radius"],
            "low_info_filter": settings["low_info_filter"],
            "low_info_min_words": settings["low_info_min_words"],
            "low_info_cue_ratio": settings["low_info_cue_ratio"],
            "max_results_per_query": settings["max_results_per_query"],
            "include_quarantined_subs": bool(config.args.include_quarantined_subs),
            "multi_episode_mode": settings["multi_episode_settings"].mode,
            "multi_episode_duration_ratio_threshold": settings["multi_episode_settings"].duration_ratio_threshold,
            "multi_episode_segments_ratio_threshold": settings["multi_episode_settings"].segments_ratio_threshold,
            "multi_episode_min_extra_minutes": settings["multi_episode_settings"].min_extra_minutes,
            "multi_episode_min_extra_segments": settings["multi_episode_settings"].min_extra_segments,
            "multi_episode_split_search_window_seconds": settings["multi_episode_settings"].split_search_window_seconds,
            "multi_episode_min_side_segments": settings["multi_episode_settings"].min_side_segments,
            "multi_episode_candidate_k": settings["multi_episode_settings"].candidate_k,
            "multi_episode_candidate_k_retry": settings["multi_episode_settings"].candidate_k_retry,
            "multi_episode_second_half_horizon_multiplier": (
                settings["multi_episode_settings"].second_half_horizon_multiplier
            ),
            "multi_episode_pair_margin": settings["multi_episode_settings"].pair_margin,
            "multi_episode_miss_penalty": settings["multi_episode_settings"].miss_penalty,
            "support_window_bonus": float(config.args.support_window_bonus),
            "support_offset_penalty": float(config.args.support_offset_penalty),
            "retrieval_mode": "indexed_hnswlib_support_aware",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    _print_report(report, top_ks)

    if config.args.errors_output:
        mismatches = list(overall_report.get("top1_mismatches", []))
        mismatches.extend(video_report.get("video_mismatches", []))
        _write_errors_output(
            Path(config.args.errors_output).expanduser().resolve(),
            mismatches,
        )
    if config.args.multi_failures_output:
        multi_failures = _collect_multi_episode_failures(video_report)
        _write_multi_episode_failures_output(
            Path(config.args.multi_failures_output).expanduser().resolve(),
            multi_failures,
            pair_margin=float(settings["multi_episode_settings"].pair_margin),
        )

    if config.args.output:
        output = Path(config.args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
        console.print(f"[green]Wrote report to {output}[/green]")


def _resolve_segment_duration(dataset_dir: Path, override: int | None) -> int:
    if override:
        return override
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return 30
    with meta_path.open("r", encoding="utf-8") as meta_in:
        meta = json.load(meta_in)
    return int(meta.get("segment_duration", 30))


def _resolve_subtitle_overlap_seconds(dataset_dir: Path, override: int | None) -> int:
    if override is not None:
        return int(override)
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return 5
    with meta_path.open("r", encoding="utf-8") as meta_in:
        meta = json.load(meta_in)
    return int(meta.get("subtitle_overlap_seconds", 5))


def _load_dataset_meta(dataset_dir: Path) -> dict:
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as meta_in:
        return json.load(meta_in)


def _resolve_matching_settings(meta: dict, args) -> dict[str, Any]:
    series_like = SimpleNamespace(
        window_expansion_mode=meta.get("window_expansion_mode"),
        window_neighbor_radius=meta.get("window_neighbor_radius"),
        low_info_filter=meta.get("low_info_filter"),
        low_info_min_words=meta.get("low_info_min_words"),
        low_info_cue_ratio=meta.get("low_info_cue_ratio"),
        max_results_per_query=meta.get("max_results_per_query"),
        multi_episode_mode=meta.get("multi_episode_mode"),
        multi_episode_duration_ratio_threshold=meta.get("multi_episode_duration_ratio_threshold"),
        multi_episode_segments_ratio_threshold=meta.get("multi_episode_segments_ratio_threshold"),
        multi_episode_min_extra_minutes=meta.get("multi_episode_min_extra_minutes"),
        multi_episode_min_extra_segments=meta.get("multi_episode_min_extra_segments"),
        multi_episode_split_search_window_seconds=meta.get("multi_episode_split_search_window_seconds"),
        multi_episode_min_side_segments=meta.get("multi_episode_min_side_segments"),
        multi_episode_candidate_k=meta.get("multi_episode_candidate_k"),
        multi_episode_candidate_k_retry=meta.get("multi_episode_candidate_k_retry"),
        multi_episode_second_half_horizon_multiplier=meta.get("multi_episode_second_half_horizon_multiplier"),
        multi_episode_pair_margin=meta.get("multi_episode_pair_margin"),
        multi_episode_miss_penalty=meta.get("multi_episode_miss_penalty"),
    )
    return {
        "window_expansion_mode": resolve_window_expansion_mode(args, series_like),
        "window_neighbor_radius": resolve_window_neighbor_radius(args, series_like),
        "low_info_filter": resolve_low_info_filter(args, series_like),
        "low_info_min_words": resolve_low_info_min_words(args, series_like),
        "low_info_cue_ratio": resolve_low_info_cue_ratio(args, series_like),
        "max_results_per_query": resolve_max_results_per_query(args, series_like),
        "multi_episode_settings": resolve_multi_episode_settings(args, series_like),
    }


def _resolve_dataset_paths(dataset_dir: Path) -> DatasetPaths:
    meta_path = dataset_dir / "meta.json"
    defaults = {
        "manifest": "manifest.jsonl",
        "subtitles_dir": "subtitles/srt",
        "transcriptions_dir": "transcriptions/text",
    }
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as meta_in:
            meta = json.load(meta_in)
        path_config = {**defaults, **meta.get("paths", {})}
    else:
        path_config = defaults

    return DatasetPaths(
        manifest=dataset_dir / path_config["manifest"],
        subtitles_dir=dataset_dir / path_config["subtitles_dir"],
        transcriptions_dir=dataset_dir / path_config["transcriptions_dir"],
    )


def _load_manifest(manifest_path: Path, limit: int | None) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as manifest_in:
        for idx, line in enumerate(manifest_in):
            if limit is not None and idx >= limit:
                break
            yield json.loads(line)


def _filter_records_by_profiles(records: list[dict], profiles: list[str] | None) -> list[dict]:
    if not profiles:
        return records
    profile_set = set(profiles)
    filtered = []
    for record in records:
        profile = str(record.get("variant_profile", "aligned"))
        if profile in profile_set:
            filtered.append(record)
    return filtered


def _load_segments_from_records(records: list[dict], dataset_dir: Path, task_id: int,
    progress: Progress) -> list[SegmentRecord]:
    segments = []
    for payload in records:
        transcription_rel = payload.get("transcription_path")
        if not transcription_rel:
            progress.update(task_id, advance=1)
            continue
        transcription_path = dataset_dir / transcription_rel
        if not transcription_path.exists():
            progress.update(task_id, advance=1)
            continue

        expected_ids = payload.get("episodes") or [payload.get("episode")]
        expected = {
            _parse_episode_key(episode_id)
            for episode_id in expected_ids
            if episode_id
        }
        if not expected:
            progress.update(task_id, advance=1)
            continue

        with transcription_path.open("r", encoding="utf-8") as transcript_in:
            transcript_map = json.load(transcript_in)
        variant_profile = str(payload.get("variant_profile", "aligned"))
        for segment_index, transcript_text in transcript_map.items():
            text = str(transcript_text).strip()
            if not text:
                continue
            try:
                interval_index = int(segment_index)
            except (TypeError, ValueError):
                continue
            segments.append(SegmentRecord(
                segment_index=interval_index,
                transcript_text=text,
                expected=expected,
                video_path=str(payload.get("video_path", "")),
                variant_profile=variant_profile,
            ))
        progress.update(task_id, advance=1)
    return segments


def _build_subtitle_indexes(subtitles_dir: Path, segment_duration_seconds: int,
    subtitle_overlap_seconds: int,
    model: SentenceTransformerModel,
    include_quarantined_subs: bool = False,
    progress: Progress | None = None) -> dict[int, IntervalSubtitleIndex]:
    window_config = make_window_config(segment_duration_seconds, subtitle_overlap_seconds)
    vectors_by_interval: dict[int, list[tuple[EpisodeKey, np.ndarray, SubtitleWindowMetadata]]] = {}
    srt_paths = [
        path
        for _, path in subtitle_file_entries(
            subtitles_dir,
            include_quarantined=include_quarantined_subs,
        )
    ]
    build_task = None
    if progress:
        build_task = progress.add_task("Embedding subtitle intervals", total=len(srt_paths))

    for srt_path in srt_paths:
        episode = EpisodeKey.from_srt_path(srt_path)
        if not episode:
            if progress and build_task is not None:
                progress.update(build_task, advance=1)
            continue
        for interval_index, interval_start_ms, interval_end_ms, interval_text in _window_texts(
            srt_path,
            window_config.window_seconds,
            window_config.overlap_seconds,
        ):
            embedding = model.encode_document(interval_text)
            metadata = SubtitleWindowMetadata(
                episode=episode,
                start_ms=interval_start_ms,
                end_ms=interval_end_ms,
                text=interval_text,
                subtitle_path=str(srt_path),
            )
            vectors_by_interval.setdefault(interval_index, []).append(
                (episode, embedding, metadata)
            )
        if progress and build_task is not None:
            progress.update(build_task, advance=1)

    index_task = None
    if progress:
        index_task = progress.add_task("Indexing subtitle intervals", total=len(vectors_by_interval))

    reduced: dict[int, IntervalSubtitleIndex] = {}
    for interval_index, episode_vectors in vectors_by_interval.items():
        episodes = [episode for episode, _, _ in episode_vectors]
        matrix = np.vstack([vector for _, vector, _ in episode_vectors]).astype(np.float32)
        metadata = [row_metadata for _, _, row_metadata in episode_vectors]
        ann_index = _build_interval_index(matrix)
        reduced[interval_index] = IntervalSubtitleIndex(
            episodes=episodes,
            index=ann_index,
            metadata=metadata,
        )
        if progress and index_task is not None:
            progress.update(index_task, advance=1)
    return reduced


def _build_interval_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
    index.add_items(vectors, np.arange(len(vectors), dtype=np.int32))
    index.set_ef(200)
    return index


def _window_texts(srt_path: Path, window_seconds: int, overlap_seconds: int) -> Iterable[tuple[int, int, int, str]]:
    subs = pysubs2.load(str(srt_path), format_="srt")
    if not subs:
        return
    max_ts = max(int(sub.end) for sub in subs)
    window_config = make_window_config(window_seconds, overlap_seconds)
    stride_ms = window_config.stride_seconds * 1000
    window_ms = window_config.window_seconds * 1000
    interval_count = math.ceil(max_ts / stride_ms)
    for interval_index in range(interval_count):
        start = interval_index * stride_ms
        end = start + window_ms
        interval_text = " ".join(
            sub.plaintext for sub in subs
            if sub.start < end and sub.end > start
        )
        yield interval_index, start, end, interval_text


def _score_segments(segments: list[SegmentRecord],
    subtitle_indexes: dict[int, IntervalSubtitleIndex],
    model: SentenceTransformerModel, top_ks: list[int], max_failures: int,
    segment_duration_seconds: int = 30,
    subtitle_overlap_seconds: int = 5,
    window_expansion_mode: str = "always",
    window_neighbor_radius: int = 1,
    low_info_filter: bool = True,
    low_info_min_words: int = 8,
    low_info_cue_ratio: float = 0.25,
    max_results_per_query: int = 10,
    support_window_bonus: float = DEFAULT_SUPPORT_WINDOW_BONUS,
    support_offset_penalty: float = DEFAULT_SUPPORT_OFFSET_PENALTY,
    collect_mismatches: bool = True,
    include_match_text: bool = False,
    progress: Progress | None = None) -> dict:
    window_config = make_window_config(segment_duration_seconds, subtitle_overlap_seconds)
    top_hits = {k: 0 for k in top_ks}
    reciprocal_rank_sum = 0.0
    evaluated = 0
    missing_interval = 0
    segments_filtered_low_info = 0
    low_info_reason_counts = Counter()
    top1_confusions = Counter()
    failures = []
    top1_mismatches = []

    score_task = None
    if progress:
        score_task = progress.add_task("Scoring transcript segments", total=len(segments))

    for segment in segments:
        segment_start_seconds = int(segment.segment_index) * int(segment_duration_seconds)
        segment_end_seconds = segment_start_seconds + int(segment_duration_seconds)
        segment_start_timestamp = _format_seconds_timestamp(segment_start_seconds)
        segment_end_timestamp = _format_seconds_timestamp(segment_end_seconds)

        if low_info_filter:
            quality = analyze_segment_quality(
                segment.transcript_text,
                min_words=low_info_min_words,
                cue_ratio_threshold=low_info_cue_ratio,
            )
            if quality.is_low_info:
                segments_filtered_low_info += 1
                low_info_reason_counts.update(quality.reasons)
                if progress and score_task is not None:
                    progress.update(score_task, advance=1)
                continue

        mapped_interval = map_segment_index_to_window_index(
            segment.segment_index,
            segment_duration_seconds,
            window_config.stride_seconds,
        )
        query_embedding = model.encode_query(segment.transcript_text)
        per_episode_support = {}
        per_episode_match_windows: dict[EpisodeKey, dict[int, dict[str, Any]]] = {}
        mapped_window_distances: list[float] = []

        mapped_data = subtitle_indexes.get(mapped_interval)
        if mapped_data:
            for hit in _query_interval_index(mapped_data, query_embedding, mapped_interval,
                max_results_per_query):
                mapped_window_distances.append(hit.distance)
                adjusted_distance = distance_with_window_penalty(
                    hit.distance,
                    hit.interval_index,
                    mapped_interval,
                )
                merge_episode_window_hit(
                    per_episode_support,
                    hit.episode,
                    hit.interval_index,
                    adjusted_distance,
                )
                _merge_episode_match_window(
                    per_episode_match_windows,
                    hit,
                    adjusted_distance,
                    include_match_text=include_match_text,
                )

        if should_expand_to_neighbor_windows(
            mapped_window_distances,
            expansion_mode=window_expansion_mode,
        ):
            for interval_index in neighbor_window_indexes(
                mapped_interval,
                radius=window_neighbor_radius,
            ):
                if interval_index == mapped_interval:
                    continue
                interval_data = subtitle_indexes.get(interval_index)
                if not interval_data:
                    continue
                for hit in _query_interval_index(interval_data, query_embedding, interval_index,
                    max_results_per_query):
                    adjusted_distance = distance_with_window_penalty(
                        hit.distance,
                        hit.interval_index,
                        mapped_interval,
                    )
                    merge_episode_window_hit(
                        per_episode_support,
                        hit.episode,
                        hit.interval_index,
                        adjusted_distance,
                    )
                    _merge_episode_match_window(
                        per_episode_match_windows,
                        hit,
                        adjusted_distance,
                        include_match_text=include_match_text,
                    )

        if not per_episode_support:
            missing_interval += 1
            if progress and score_task is not None:
                progress.update(score_task, advance=1)
            continue

        ranked_scored: list[tuple[EpisodeKey, float, int]] = []
        for episode, support in per_episode_support.items():
            score, nearest_offset = support_aware_score(
                support,
                mapped_interval,
                support_window_bonus=support_window_bonus,
                support_offset_penalty=support_offset_penalty,
            )
            ranked_scored.append((episode, score, nearest_offset))
        ranked_scored.sort(key=lambda item: (item[1], item[2], item[0]))
        ranked = [episode for episode, _, _ in ranked_scored]

        rank = _first_expected_rank(ranked, segment.expected)
        evaluated += 1
        if rank:
            reciprocal_rank_sum += 1.0 / rank
            for k in top_ks:
                if rank <= k:
                    top_hits[k] += 1
        top_prediction_matches = _top_prediction_matches(
            ranked_scored,
            per_episode_support,
            per_episode_match_windows,
            max_results=max(top_ks),
            max_windows_per_episode=3,
        ) if include_match_text else []
        top_predictions = [str(ep) for ep in ranked[:max(top_ks)]]
        expected_ids = [str(ep) for ep in sorted(segment.expected)]
        if rank != 1:
            expected_label = "|".join(expected_ids)
            predicted_label = top_predictions[0] if top_predictions else "<none>"
            top1_confusions[(expected_label, predicted_label)] += 1
            if collect_mismatches:
                mismatch = {
                    "video_path": segment.video_path,
                    "segment_index": segment.segment_index,
                    "segment_start_seconds": segment_start_seconds,
                    "segment_end_seconds": segment_end_seconds,
                    "segment_start_timestamp": segment_start_timestamp,
                    "segment_end_timestamp": segment_end_timestamp,
                    "variant_profile": segment.variant_profile,
                    "expected": expected_ids,
                    "top_predictions": top_predictions,
                    "rank": rank,
                    "transcript_text": segment.transcript_text,
                }
                if include_match_text:
                    mismatch["top_prediction_matches"] = top_prediction_matches
                top1_mismatches.append(mismatch)
        if (rank is None or rank > max(top_ks)) and len(failures) < max_failures:
            failure = {
                "video_path": segment.video_path,
                "segment_index": segment.segment_index,
                "segment_start_seconds": segment_start_seconds,
                "segment_end_seconds": segment_end_seconds,
                "segment_start_timestamp": segment_start_timestamp,
                "segment_end_timestamp": segment_end_timestamp,
                "variant_profile": segment.variant_profile,
                "expected": expected_ids,
                "top_predictions": top_predictions,
                "rank": rank,
                "transcript_text": segment.transcript_text,
            }
            if include_match_text:
                failure["top_prediction_matches"] = top_prediction_matches
            failures.append(failure)
        if progress and score_task is not None:
            progress.update(score_task, advance=1)

    top_accuracy = {
        f"top_{k}": (top_hits[k] / evaluated if evaluated else 0.0)
        for k in top_ks
    }
    confusion_rows = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in top1_confusions.most_common()
    ]
    return {
        "segments_total": len(segments),
        "segments_evaluated": evaluated,
        "segments_missing_interval": missing_interval,
        "segments_filtered_low_info": segments_filtered_low_info,
        "filtered_low_info_ratio": (
            segments_filtered_low_info / len(segments) if segments else 0.0
        ),
        "low_info_reason_counts": dict(low_info_reason_counts),
        "mrr": reciprocal_rank_sum / evaluated if evaluated else 0.0,
        "accuracy": top_accuracy,
        "top_hits": {f"top_{k}": top_hits[k] for k in top_ks},
        "top1_confusions": confusion_rows,
        "top1_mismatches": top1_mismatches,
        "failure_examples": failures,
    }


def _score_video_assignments(
    *,
    records: list[dict],
    dataset_dir: Path,
    subtitle_indexes: dict[int, IntervalSubtitleIndex],
    model: SentenceTransformerModel,
    segment_duration_seconds: int,
    subtitle_overlap_seconds: int,
    settings: dict[str, Any],
    support_window_bonus: float,
    support_offset_penalty: float,
    include_quarantined_subs: bool,
) -> dict[str, Any]:
    runtime_profile = _build_dataset_runtime_profile(
        dataset_dir,
        include_quarantined_subs=include_quarantined_subs,
    )
    multi_settings = settings["multi_episode_settings"]
    window_config = make_window_config(segment_duration_seconds, subtitle_overlap_seconds)

    total = 0
    exact_order_hits = 0
    set_hits = 0
    single_hits = 0
    single_total = 0
    detected_multi = 0
    fallback_to_single = 0
    mismatches: list[dict[str, Any]] = []

    for payload in records:
        transcription_rel = payload.get("transcription_path")
        if not transcription_rel:
            continue
        transcription_path = dataset_dir / str(transcription_rel)
        if not transcription_path.exists():
            continue
        with transcription_path.open("r", encoding="utf-8") as transcript_in:
            transcript_map = json.load(transcript_in)

        rows: list[tuple[int, np.ndarray]] = []
        for segment_index, transcript_text in transcript_map.items():
            text = str(transcript_text).strip()
            if not text:
                continue
            try:
                idx = int(segment_index)
            except (TypeError, ValueError):
                continue
            if settings["low_info_filter"]:
                quality = analyze_segment_quality(
                    text,
                    min_words=settings["low_info_min_words"],
                    cue_ratio_threshold=settings["low_info_cue_ratio"],
                )
                if quality.is_low_info:
                    continue
            rows.append((idx, model.encode_query(text)))

        if not rows:
            continue
        rows.sort(key=lambda row: row[0])

        expected_ids = payload.get("episodes") or [payload.get("episode")]
        expected = [_parse_episode_key(value) for value in expected_ids if value]
        if not expected:
            continue

        fallback_single = _rank_video_single_episode(
            rows=rows,
            subtitle_indexes=subtitle_indexes,
            segment_duration_seconds=segment_duration_seconds,
            window_stride_seconds=window_config.stride_seconds,
            window_expansion_mode=settings["window_expansion_mode"],
            base_neighbor_radius=settings["window_neighbor_radius"],
            max_results_per_query=settings["max_results_per_query"],
            support_window_bonus=support_window_bonus,
            support_offset_penalty=support_offset_penalty,
        )
        duration_minutes = float(
            payload.get(
                "duration_minutes",
                max(1.0, (max(segment for segment, _ in rows) + 1) * segment_duration_seconds / 60.0),
            )
        )
        detection = detect_multi_episode_candidate(
            settings=multi_settings,
            profile=runtime_profile,
            video_minutes=duration_minutes,
            observed_segments=len(rows),
            segments_per_minute=float(payload.get("segments_per_minute", 0.5)),
        )
        if detection.is_multi_candidate:
            detected_multi += 1

        def query_segment(
            embedding: np.ndarray,
            mapped_window: int,
            neighbor_radius: int,
            max_results_per_query: int,
        ) -> list[tuple[EpisodeKey, float]]:
            ranked = _query_segment_from_subtitle_indexes(
                subtitle_indexes=subtitle_indexes,
                embedding=embedding,
                mapped_interval=mapped_window,
                window_expansion_mode=settings["window_expansion_mode"],
                neighbor_radius=neighbor_radius,
                max_results_per_query=max_results_per_query,
                support_window_bonus=support_window_bonus,
                support_offset_penalty=support_offset_penalty,
            )
            return [(episode, score) for episode, score, _ in ranked]

        assignment = resolve_multi_episode_assignment(
            settings=multi_settings,
            detection=detection,
            segment_rows=rows,
            video_minutes=duration_minutes,
            segment_duration_seconds=segment_duration_seconds,
            stride_seconds=window_config.stride_seconds,
            window_seconds=window_config.window_seconds,
            base_neighbor_radius=settings["window_neighbor_radius"],
            max_results_per_query=settings["max_results_per_query"],
            query_segment=query_segment,
            fallback_single_episode=fallback_single,
        )
        should_try_multi = (multi_settings.mode == "force-2") or detection.is_multi_candidate
        if should_try_multi and assignment.assignment_mode == "single":
            fallback_to_single += 1

        predicted = list(assignment.assigned_episodes)
        total += 1
        exact_order = predicted == expected
        set_match = len(predicted) == len(expected) and set(predicted) == set(expected)
        if exact_order:
            exact_order_hits += 1
        if set_match:
            set_hits += 1
        if len(expected) == 1:
            single_total += 1
            if predicted and predicted[0] == expected[0]:
                single_hits += 1

        if not exact_order:
            mismatches.append(
                {
                    "record_type": "video_assignment_mismatch",
                    "video_path": payload.get("video_path", ""),
                    "transcription_path": str(transcription_rel),
                    "variant_profile": payload.get("variant_profile", "aligned"),
                    "expected": [str(ep) for ep in expected],
                    "predicted": [str(ep) for ep in predicted],
                    "assignment_mode": assignment.assignment_mode,
                    "split_seconds": assignment.split_seconds,
                    "assignment_confidence": assignment.assignment_confidence,
                    "runtime_profile_type": assignment.runtime_profile_type,
                    "detector_reasons": list(assignment.detector_reasons),
                    "diagnostics": assignment.diagnostics,
                }
            )

    return {
        "videos_total": total,
        "videos_multi_detected": detected_multi,
        "fallback_to_single_count": fallback_to_single,
        "video_exact_order_accuracy": (exact_order_hits / total) if total else 0.0,
        "video_episode_set_accuracy": (set_hits / total) if total else 0.0,
        "video_single_top1_accuracy": (single_hits / single_total) if single_total else 0.0,
        "video_mismatches": mismatches,
    }


def _query_interval_index(
    interval_data: IntervalSubtitleIndex,
    query_embedding: np.ndarray,
    interval_index: int,
    max_results_per_query: int,
) -> list[IntervalQueryHit]:
    count = interval_data.index.get_current_count()
    if count <= 0:
        return []
    k = min(max_results_per_query, count)
    ids_by_q, dists_by_q = interval_data.index.knn_query(
        query_embedding, k=k, num_threads=1, filter=None
    )
    ids, dists = ids_by_q[0], dists_by_q[0]
    hits: list[IntervalQueryHit] = []
    for id, distance in zip(ids, dists):
        idx = int(id)
        metadata = None
        if interval_data.metadata and idx < len(interval_data.metadata):
            metadata = interval_data.metadata[idx]
        hits.append(
            IntervalQueryHit(
                episode=interval_data.episodes[idx],
                distance=float(distance),
                interval_index=interval_index,
                metadata=metadata,
            )
        )
    return hits


def _query_segment_from_subtitle_indexes(
    *,
    subtitle_indexes: dict[int, IntervalSubtitleIndex],
    embedding: np.ndarray,
    mapped_interval: int,
    window_expansion_mode: str,
    neighbor_radius: int,
    max_results_per_query: int,
    support_window_bonus: float,
    support_offset_penalty: float,
) -> list[tuple[EpisodeKey, float, int]]:
    per_episode_support = {}
    mapped_window_distances: list[float] = []

    mapped_data = subtitle_indexes.get(int(mapped_interval))
    if mapped_data:
        for hit in _query_interval_index(
            mapped_data,
            embedding,
            int(mapped_interval),
            int(max_results_per_query),
        ):
            mapped_window_distances.append(hit.distance)
            adjusted_distance = distance_with_window_penalty(
                hit.distance,
                hit.interval_index,
                int(mapped_interval),
            )
            merge_episode_window_hit(
                per_episode_support,
                hit.episode,
                hit.interval_index,
                adjusted_distance,
            )

    if should_expand_to_neighbor_windows(
        mapped_window_distances,
        expansion_mode=window_expansion_mode,
    ):
        for interval_index in neighbor_window_indexes(
            int(mapped_interval),
            radius=int(neighbor_radius),
        ):
            if interval_index == int(mapped_interval):
                continue
            interval_data = subtitle_indexes.get(interval_index)
            if not interval_data:
                continue
            for hit in _query_interval_index(
                interval_data,
                embedding,
                interval_index,
                int(max_results_per_query),
            ):
                adjusted_distance = distance_with_window_penalty(
                    hit.distance,
                    hit.interval_index,
                    int(mapped_interval),
                )
                merge_episode_window_hit(
                    per_episode_support,
                    hit.episode,
                    hit.interval_index,
                    adjusted_distance,
                )

    ranked = []
    for episode, support in per_episode_support.items():
        score, nearest_offset = support_aware_score(
            support,
            int(mapped_interval),
            support_window_bonus=support_window_bonus,
            support_offset_penalty=support_offset_penalty,
        )
        ranked.append((episode, float(score), int(support.best_window_index), int(nearest_offset)))
    ranked.sort(key=lambda item: (item[1], item[3], item[0]))
    return [(episode, score, best_window_index) for episode, score, best_window_index, _ in ranked]


def _rank_video_single_episode(
    *,
    rows: list[tuple[int, np.ndarray]],
    subtitle_indexes: dict[int, IntervalSubtitleIndex],
    segment_duration_seconds: int,
    window_stride_seconds: int,
    window_expansion_mode: str,
    base_neighbor_radius: int,
    max_results_per_query: int,
    support_window_bonus: float,
    support_offset_penalty: float,
) -> EpisodeKey | None:
    stats: dict[EpisodeKey, tuple[int, float]] = {}
    for segment_index, embedding in rows:
        mapped = map_segment_index_to_window_index(
            int(segment_index),
            int(segment_duration_seconds),
            int(window_stride_seconds),
        )
        ranked = _query_segment_from_subtitle_indexes(
            subtitle_indexes=subtitle_indexes,
            embedding=embedding,
            mapped_interval=int(mapped),
            window_expansion_mode=window_expansion_mode,
            neighbor_radius=int(base_neighbor_radius),
            max_results_per_query=int(max_results_per_query),
            support_window_bonus=support_window_bonus,
            support_offset_penalty=support_offset_penalty,
        )
        for episode, score, _ in ranked:
            count, min_score = stats.get(episode, (0, float("inf")))
            stats[episode] = (count + 1, min(float(min_score), float(score)))
    if not stats:
        return None
    ordered = sorted(stats.items(), key=lambda row: (-row[1][0], row[1][1], row[0]))
    return ordered[0][0]


def _build_dataset_runtime_profile(dataset_dir: Path, *, include_quarantined_subs: bool):
    meta = _load_dataset_meta(dataset_dir)
    tmdb_minutes = {}
    source_series_dir = meta.get("source_series_dir")
    if source_series_dir:
        series = Series.from_dir(Path(source_series_dir).expanduser().resolve())
        if series:
            tmdb_minutes = collect_tmdb_episode_runtimes_minutes(series)
    dataset_paths = _resolve_dataset_paths(dataset_dir)
    srt_minutes = collect_srt_episode_runtimes_minutes(dataset_paths.subtitles_dir)
    if not include_quarantined_subs:
        for episode_key in load_quarantined_episodes(dataset_paths.subtitles_dir):
            srt_minutes.pop(episode_key, None)
    return build_runtime_profile(tmdb_minutes, srt_minutes)


def _merge_episode_match_window(
    per_episode_match_windows: dict[EpisodeKey, dict[int, dict[str, Any]]],
    hit: IntervalQueryHit,
    adjusted_distance: float,
    include_match_text: bool,
):
    if not include_match_text or not hit.metadata:
        return
    per_window = per_episode_match_windows.setdefault(hit.episode, {})
    existing = per_window.get(hit.interval_index)
    if existing is not None and existing["adjusted_distance"] <= float(adjusted_distance):
        return
    metadata = hit.metadata
    per_window[hit.interval_index] = {
        "window_index": int(hit.interval_index),
        "window_start_ms": int(metadata.start_ms),
        "window_end_ms": int(metadata.end_ms),
        "window_start_seconds": float(metadata.start_ms) / 1000.0,
        "window_end_seconds": float(metadata.end_ms) / 1000.0,
        "distance": float(hit.distance),
        "adjusted_distance": float(adjusted_distance),
        "subtitle_text": metadata.text,
        "subtitle_path": metadata.subtitle_path,
    }


def _top_prediction_matches(
    ranked_scored: list[tuple[EpisodeKey, float, int]],
    per_episode_support: dict[EpisodeKey, Any],
    per_episode_match_windows: dict[EpisodeKey, dict[int, dict[str, Any]]],
    max_results: int,
    max_windows_per_episode: int = 3,
) -> list[dict]:
    rows = []
    for episode, score, nearest_window_offset in ranked_scored[:max_results]:
        support = per_episode_support[episode]
        window_rows = sorted(
            per_episode_match_windows.get(episode, {}).values(),
            key=lambda row: (row["adjusted_distance"], row["window_index"]),
        )[:max_windows_per_episode]
        rows.append(
            {
                "episode": str(episode),
                "score": float(score),
                "nearest_window_offset": int(nearest_window_offset),
                "support_windows": len(support.window_best_distances),
                "match_windows": window_rows,
            }
        )
    return rows


def _segments_by_profile(segments: list[SegmentRecord]) -> dict[str, list[SegmentRecord]]:
    grouped = {}
    for segment in segments:
        grouped.setdefault(segment.variant_profile, []).append(segment)
    return grouped


def _first_expected_rank(ranked: list[EpisodeKey], expected: set[EpisodeKey]) -> int | None:
    for idx, episode in enumerate(ranked, start=1):
        if episode in expected:
            return idx
    return None


def _format_seconds_timestamp(total_seconds: float) -> str:
    value = float(total_seconds)
    whole_seconds = int(value)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60
    fractional = value - whole_seconds
    if fractional > 0:
        milliseconds = int(round(fractional * 1000))
        if milliseconds >= 1000:
            milliseconds = 0
            seconds += 1
        if seconds >= 60:
            seconds = 0
            minutes += 1
        if minutes >= 60:
            minutes = 0
            hours += 1
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_episode_key(value: str) -> EpisodeKey:
    match = EPISODE_PATTERN.match(value)
    if not match:
        raise ValueError(f"Invalid episode format: {value}")
    return EpisodeKey(int(match.group("season")), int(match.group("episode")))


def _print_report(report: dict, top_ks: list[int]):
    table = Table(title="Dataset Evaluation")
    table.add_column("Metric")
    table.add_column("Value")
    overall = report["overall"]
    table.add_row("Segments total", str(overall["segments_total"]))
    table.add_row("Segments evaluated", str(overall["segments_evaluated"]))
    table.add_row("Segments missing interval", str(overall["segments_missing_interval"]))
    table.add_row("Segments filtered low-info", str(overall["segments_filtered_low_info"]))
    table.add_row("Filtered low-info ratio", f"{overall['filtered_low_info_ratio']:.4f}")
    for k in top_ks:
        acc = overall["accuracy"][f"top_{k}"]
        hits = overall["top_hits"][f"top_{k}"]
        table.add_row(f"Top-{k} accuracy", f"{acc:.4f} ({hits})")
    table.add_row("MRR", f"{overall['mrr']:.4f}")
    console.print(table)

    by_profile = report.get("by_profile", {})
    if by_profile:
        profile_table = Table(title="By Profile")
        profile_table.add_column("Profile")
        profile_table.add_column("Segments")
        for k in top_ks:
            profile_table.add_column(f"Top-{k}")
        profile_table.add_column("MRR")
        for profile, metrics in sorted(by_profile.items()):
            row = [profile, str(metrics["segments_evaluated"])]
            for k in top_ks:
                row.append(f"{metrics['accuracy'][f'top_{k}']:.4f}")
            row.append(f"{metrics['mrr']:.4f}")
            profile_table.add_row(*row)
        console.print(profile_table)

    confusions = overall.get("top1_confusions", [])
    if confusions:
        confusion_table = Table(title="Top-1 Confusions")
        confusion_table.add_column("Expected")
        confusion_table.add_column("Predicted")
        confusion_table.add_column("Count")
        for row in confusions[:10]:
            confusion_table.add_row(row["expected"], row["predicted"], str(row["count"]))
        console.print(confusion_table)

    video_level = report.get("video_level")
    if video_level:
        video_table = Table(title="Video-Level Assignment")
        video_table.add_column("Metric")
        video_table.add_column("Value")
        video_table.add_row("Videos total", str(video_level.get("videos_total", 0)))
        video_table.add_row(
            "Videos multi detected",
            str(video_level.get("videos_multi_detected", 0)),
        )
        video_table.add_row(
            "Fallback-to-single count",
            str(video_level.get("fallback_to_single_count", 0)),
        )
        video_table.add_row(
            "Exact ordered pair accuracy",
            f"{video_level.get('video_exact_order_accuracy', 0.0):.4f}",
        )
        video_table.add_row(
            "Episode set accuracy",
            f"{video_level.get('video_episode_set_accuracy', 0.0):.4f}",
        )
        video_table.add_row(
            "Single top-1 accuracy",
            f"{video_level.get('video_single_top1_accuracy', 0.0):.4f}",
        )
        console.print(video_table)


def _write_errors_output(path: Path, mismatches: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        _write_errors_output_html(path, mismatches)
    elif suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(
                out,
                fieldnames=[
                    "record_type",
                    "video_path",
                    "transcription_path",
                    "segment_index",
                    "segment_start_seconds",
                    "segment_end_seconds",
                    "segment_start_timestamp",
                    "segment_end_timestamp",
                    "variant_profile",
                    "expected",
                    "top_predictions",
                    "predicted",
                    "top_prediction_matches",
                    "rank",
                    "assignment_mode",
                    "split_seconds",
                    "assignment_confidence",
                    "runtime_profile_type",
                    "detector_reasons",
                    "diagnostics",
                    "transcript_text",
                ],
            )
            writer.writeheader()
            for mismatch in mismatches:
                writer.writerow(
                    {
                        "video_path": mismatch.get("video_path", ""),
                        "record_type": mismatch.get("record_type", "segment_mismatch"),
                        "transcription_path": mismatch.get("transcription_path", ""),
                        "segment_index": mismatch.get("segment_index", ""),
                        "segment_start_seconds": mismatch.get("segment_start_seconds", ""),
                        "segment_end_seconds": mismatch.get("segment_end_seconds", ""),
                        "segment_start_timestamp": mismatch.get("segment_start_timestamp", ""),
                        "segment_end_timestamp": mismatch.get("segment_end_timestamp", ""),
                        "variant_profile": mismatch.get("variant_profile", ""),
                        "expected": "|".join(mismatch.get("expected", [])),
                        "top_predictions": "|".join(mismatch.get("top_predictions", [])),
                        "predicted": "|".join(mismatch.get("predicted", [])),
                        "top_prediction_matches": json.dumps(
                            mismatch.get("top_prediction_matches", []),
                            ensure_ascii=False,
                        ),
                        "rank": mismatch.get("rank", ""),
                        "assignment_mode": mismatch.get("assignment_mode", ""),
                        "split_seconds": mismatch.get("split_seconds", ""),
                        "assignment_confidence": mismatch.get("assignment_confidence", ""),
                        "runtime_profile_type": mismatch.get("runtime_profile_type", ""),
                        "detector_reasons": "|".join(mismatch.get("detector_reasons", [])),
                        "diagnostics": json.dumps(mismatch.get("diagnostics", {}), ensure_ascii=False),
                        "transcript_text": mismatch.get("transcript_text", ""),
                    }
                )
    else:
        with path.open("w", encoding="utf-8") as out:
            for mismatch in mismatches:
                out.write(json.dumps(mismatch, ensure_ascii=False))
                out.write("\n")
    console.print(f"[green]Wrote errors output to {path}[/green]")


def _collect_multi_episode_failures(video_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mismatch in video_report.get("video_mismatches", []):
        expected = mismatch.get("expected") or []
        if len(expected) < 2:
            continue
        if mismatch.get("assignment_mode") == "multi_2":
            continue
        rows.append(mismatch)
    return rows


def _write_multi_episode_failures_output(path: Path, failures: list[dict], *, pair_margin: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_multi_episode_failures_html(path, failures, pair_margin=pair_margin)
    console.print(f"[green]Wrote multi-episode failure report to {path}[/green]")


def _write_multi_episode_failures_html(path: Path, failures: list[dict], *, pair_margin: float):
    def _fmt_episode_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<div class='empty'>No ranked episode candidates captured</div>"
        parts = []
        for row in rows:
            episode = html.escape(str(row.get("episode", "")))
            mean_loss = row.get("mean_loss", "")
            hit_ratio = row.get("hit_ratio", "")
            parts.append(
                "<tr>"
                f"<td><code>{episode}</code></td>"
                f"<td>{mean_loss}</td>"
                f"<td>{hit_ratio}</td>"
                "</tr>"
            )
        return (
            "<table><thead><tr><th>Episode</th><th>Mean loss</th><th>Hit ratio</th></tr></thead>"
            f"<tbody>{''.join(parts)}</tbody></table>"
        )

    def _fmt_split_candidates(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<div class='empty'>No split candidate details captured</div>"
        cards = []
        for row in rows:
            split_seconds = row.get("split_seconds", "")
            pair = row.get("pair", [])
            pair_score = row.get("pair_score", "")
            c1_count = row.get("chunk1_segment_count", "")
            c2_count = row.get("chunk2_segment_count", "")
            pair_label = " -> ".join(html.escape(str(item)) for item in pair)
            cards.append(
                "<details class='split'>"
                f"<summary><strong>Split {split_seconds}s</strong> pair=<code>{pair_label}</code> "
                f"score={pair_score} chunk_sizes=({c1_count},{c2_count})</summary>"
                "<div class='split-grid'>"
                "<section><h4>Chunk 1 top</h4>"
                f"{_fmt_episode_rows(row.get('chunk1_top', []))}"
                "</section>"
                "<section><h4>Chunk 2 top</h4>"
                f"{_fmt_episode_rows(row.get('chunk2_top', []))}"
                "</section>"
                "</div>"
                "</details>"
            )
        return "".join(cards)

    articles = []
    for idx, failure in enumerate(failures, start=1):
        video_path = html.escape(str(failure.get("video_path", "")))
        transcription_path = html.escape(str(failure.get("transcription_path", "")))
        variant = html.escape(str(failure.get("variant_profile", "")))
        expected = html.escape("|".join(failure.get("expected", [])))
        predicted = html.escape("|".join(failure.get("predicted", [])))
        detector_reasons = ", ".join(
            html.escape(str(reason))
            for reason in (failure.get("detector_reasons") or [])
        )
        diagnostics = failure.get("diagnostics") or {}
        reason = html.escape(str(diagnostics.get("reason", "")))
        margin = diagnostics.get("margin", "")
        best_pair_score = diagnostics.get("best_pair_score", "")
        second_pair_score = diagnostics.get("second_pair_score", "")
        threshold = diagnostics.get("pair_margin_threshold", pair_margin)
        best_split_seconds = diagnostics.get("best_split_seconds", "")
        best_pair = " -> ".join(str(v) for v in (diagnostics.get("best_pair") or []))
        split_evaluated = diagnostics.get("split_evaluated") or []
        chunk1_top = diagnostics.get("chunk1_top", [])
        chunk2_top = diagnostics.get("chunk2_top", [])
        split_candidates = diagnostics.get("split_candidates", [])
        articles.append(
            "<article class='failure'>"
            f"<h2>Multi Failure #{idx}</h2>"
            "<div class='meta-grid'>"
            f"<div><strong>Video:</strong> <code>{video_path}</code></div>"
            f"<div><strong>Transcription:</strong> <code>{transcription_path}</code></div>"
            f"<div><strong>Profile:</strong> {variant}</div>"
            f"<div><strong>Expected:</strong> {expected}</div>"
            f"<div><strong>Predicted:</strong> {predicted}</div>"
            f"<div><strong>Assignment mode:</strong> {html.escape(str(failure.get('assignment_mode', '')))}</div>"
            f"<div><strong>Runtime profile:</strong> {html.escape(str(failure.get('runtime_profile_type', '')))}</div>"
            f"<div><strong>Detector reasons:</strong> {detector_reasons}</div>"
            f"<div><strong>Failure reason:</strong> {reason}</div>"
            f"<div><strong>Margin:</strong> {margin}</div>"
            f"<div><strong>Pair margin threshold:</strong> {threshold}</div>"
            f"<div><strong>Best pair score:</strong> {best_pair_score}</div>"
            f"<div><strong>Second pair score:</strong> {second_pair_score}</div>"
            f"<div><strong>Best split:</strong> {best_split_seconds}</div>"
            f"<div><strong>Best pair:</strong> <code>{html.escape(best_pair)}</code></div>"
            f"<div><strong>Splits evaluated:</strong> {html.escape(str(split_evaluated))}</div>"
            "</div>"
            "<details open><summary><strong>Best split chunk rankings</strong></summary>"
            "<div class='split-grid'>"
            "<section><h4>Chunk 1 top</h4>"
            f"{_fmt_episode_rows(chunk1_top)}"
            "</section>"
            "<section><h4>Chunk 2 top</h4>"
            f"{_fmt_episode_rows(chunk2_top)}"
            "</section>"
            "</div>"
            "</details>"
            "<details open><summary><strong>Top split candidates</strong></summary>"
            f"{_fmt_split_candidates(split_candidates)}"
            "</details>"
            "</article>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Episode Matcher Multi-Episode Failures</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#f8fafc;color:#0f172a;margin:0;padding:24px;}"
        "h1{margin:0 0 12px 0;font-size:24px;}"
        ".summary{margin:0 0 20px 0;color:#334155;}"
        ".failure{background:#fff;border:1px solid #cbd5e1;border-radius:10px;"
        "padding:16px;margin:0 0 16px 0;box-shadow:0 1px 2px rgba(15,23,42,.06);}"
        ".failure h2{margin:0 0 12px 0;font-size:18px;}"
        ".meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));"
        "gap:8px 16px;margin-bottom:12px;}"
        ".split-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));"
        "gap:12px;}"
        ".split{margin:10px 0;padding:10px;border:1px solid #e2e8f0;border-radius:8px;}"
        "details{margin:8px 0;}"
        "summary{cursor:pointer;}"
        "table{width:100%;border-collapse:collapse;}"
        "th,td{border:1px solid #e2e8f0;padding:6px 8px;text-align:left;}"
        "th{background:#f1f5f9;}"
        ".empty{color:#64748b;font-style:italic;}"
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}"
        "</style></head><body>"
        "<h1>Multi-Episode Matching Failures</h1>"
        f"<p class='summary'>Failures: {len(failures)} | Pair margin threshold: {pair_margin}</p>"
        f"{''.join(articles) if articles else '<p>No multi-episode matching failures found.</p>'}"
        "</body></html>"
    )
    path.write_text(html_doc, encoding="utf-8")


def _write_errors_output_html(path: Path, mismatches: list[dict]):
    def _fmt_windows(match_windows: list[dict]) -> str:
        if not match_windows:
            return "<div class='empty'>No window details</div>"
        parts = []
        for window in match_windows:
            start_sec = float(window.get("window_start_seconds", 0.0))
            end_sec = float(window.get("window_end_seconds", 0.0))
            score = window.get("adjusted_distance", "")
            text = html.escape(str(window.get("subtitle_text", "")))
            subtitle_path = html.escape(str(window.get("subtitle_path", "")))
            parts.append(
                "<div class='window'>"
                f"<div><strong>Window:</strong> {start_sec:.1f}s - {end_sec:.1f}s</div>"
                f"<div><strong>Adjusted distance:</strong> {score}</div>"
                f"<div><strong>Subtitle file:</strong> <code>{subtitle_path}</code></div>"
                f"<pre>{text}</pre>"
                "</div>"
            )
        return "".join(parts)

    def _fmt_predictions(top_prediction_matches: list[dict]) -> str:
        if not top_prediction_matches:
            return "<div class='empty'>No matched subtitle text details</div>"
        blocks = []
        for prediction in top_prediction_matches:
            episode = html.escape(str(prediction.get("episode", "")))
            score = prediction.get("score", "")
            support_windows = prediction.get("support_windows", "")
            nearest_offset = prediction.get("nearest_window_offset", "")
            windows_html = _fmt_windows(prediction.get("match_windows", []))
            blocks.append(
                "<details class='prediction'>"
                f"<summary><strong>{episode}</strong> score={score} "
                f"support_windows={support_windows} nearest_offset={nearest_offset}</summary>"
                f"{windows_html}</details>"
            )
        return "".join(blocks)

    rows = []
    for idx, mismatch in enumerate(mismatches, start=1):
        record_type = html.escape(str(mismatch.get("record_type", "segment_mismatch")))
        video_path = html.escape(str(mismatch.get("video_path", "")))
        variant = html.escape(str(mismatch.get("variant_profile", "")))
        expected = html.escape("|".join(mismatch.get("expected", [])))
        top_predictions = html.escape("|".join(mismatch.get("top_predictions", []) or mismatch.get("predicted", [])))
        rank = mismatch.get("rank", "")
        segment_index = mismatch.get("segment_index", "")
        segment_start_seconds = mismatch.get("segment_start_seconds", "")
        segment_end_seconds = mismatch.get("segment_end_seconds", "")
        segment_start_timestamp = html.escape(str(mismatch.get("segment_start_timestamp", "")))
        segment_end_timestamp = html.escape(str(mismatch.get("segment_end_timestamp", "")))
        transcript_text = html.escape(str(mismatch.get("transcript_text", "")))
        prediction_details = _fmt_predictions(mismatch.get("top_prediction_matches", []))
        assignment_mode = html.escape(str(mismatch.get("assignment_mode", "")))
        split_seconds = html.escape(str(mismatch.get("split_seconds", "")))
        assignment_confidence = html.escape(str(mismatch.get("assignment_confidence", "")))
        runtime_profile_type = html.escape(str(mismatch.get("runtime_profile_type", "")))
        rows.append(
            "<article class='mismatch'>"
            f"<h2>Mismatch #{idx}</h2>"
            "<div class='meta-grid'>"
            f"<div><strong>Type:</strong> {record_type}</div>"
            f"<div><strong>Video:</strong> <code>{video_path}</code></div>"
            f"<div><strong>Segment:</strong> {segment_index}</div>"
            f"<div><strong>Segment time (s):</strong> {segment_start_seconds} - {segment_end_seconds}</div>"
            f"<div><strong>Segment time:</strong> {segment_start_timestamp} - {segment_end_timestamp}</div>"
            f"<div><strong>Profile:</strong> {variant}</div>"
            f"<div><strong>Expected:</strong> {expected}</div>"
            f"<div><strong>Top predictions:</strong> {top_predictions}</div>"
            f"<div><strong>Rank:</strong> {rank}</div>"
            f"<div><strong>Assignment mode:</strong> {assignment_mode}</div>"
            f"<div><strong>Split seconds:</strong> {split_seconds}</div>"
            f"<div><strong>Assignment confidence:</strong> {assignment_confidence}</div>"
            f"<div><strong>Runtime profile:</strong> {runtime_profile_type}</div>"
            "</div>"
            "<details open><summary><strong>Transcript segment text</strong></summary>"
            f"<pre>{transcript_text}</pre></details>"
            "<details open><summary><strong>Matched subtitle windows</strong></summary>"
            f"{prediction_details}</details>"
            "</article>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Episode Matcher Mismatch Report</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#f8fafc;color:#0f172a;margin:0;padding:24px;}"
        "h1{margin:0 0 16px 0;font-size:24px;}"
        ".summary{margin:0 0 24px 0;color:#334155;}"
        ".mismatch{background:#fff;border:1px solid #cbd5e1;border-radius:10px;"
        "padding:16px;margin:0 0 16px 0;box-shadow:0 1px 2px rgba(15,23,42,.06);}"
        ".mismatch h2{margin:0 0 12px 0;font-size:18px;}"
        ".meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));"
        "gap:8px 16px;margin-bottom:12px;}"
        "details{margin:8px 0;}"
        "summary{cursor:pointer;}"
        "pre{white-space:pre-wrap;word-wrap:break-word;background:#f1f5f9;"
        "padding:10px;border-radius:8px;border:1px solid #e2e8f0;}"
        ".prediction{margin:10px 0;padding:10px;border:1px solid #e2e8f0;border-radius:8px;}"
        ".window{margin:10px 0;padding:10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;}"
        ".empty{color:#64748b;font-style:italic;}"
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}"
        "</style></head><body>"
        "<h1>Episode Matcher Mismatch Report</h1>"
        f"<p class='summary'>Total mismatches: {len(mismatches)}</p>"
        f"{''.join(rows) if rows else '<p>No mismatches found.</p>'}"
        "</body></html>"
    )
    with path.open("w", encoding="utf-8") as out:
        out.write(html_doc)
