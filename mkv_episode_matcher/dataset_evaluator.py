import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
from mkv_episode_matcher.windowing import (
    distance_with_window_penalty,
    make_window_config,
    map_segment_index_to_window_index,
    neighbor_window_indexes,
    should_expand_to_neighbor_windows,
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


def evaluate_dataset(config: Configuration):
    dataset_dir = Path(config.args.dataset_dir).expanduser().resolve()
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
            progress=progress,
        )

    report = {
        "overall": overall_report,
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
            )
        report["by_profile"] = by_profile

    report.update({
        "dataset_dir": str(dataset_dir),
        "segment_duration": interval_seconds,
        "subtitle_overlap_seconds": subtitle_overlap_seconds,
        "retrieval_mode": "indexed_hnswlib_two_stage_penalized",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    _print_report(report, top_ks)

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
    progress: Progress | None = None) -> dict[int, IntervalSubtitleIndex]:
    window_config = make_window_config(segment_duration_seconds, subtitle_overlap_seconds)
    vectors_by_interval: dict[int, list[tuple[EpisodeKey, np.ndarray]]] = {}
    srt_paths = sorted(subtitles_dir.rglob("*.srt"))
    build_task = None
    if progress:
        build_task = progress.add_task("Embedding subtitle intervals", total=len(srt_paths))

    for srt_path in srt_paths:
        episode = EpisodeKey.from_srt_path(srt_path)
        if not episode:
            if progress and build_task is not None:
                progress.update(build_task, advance=1)
            continue
        for interval_index, interval_text in _window_texts(
            srt_path,
            window_config.window_seconds,
            window_config.overlap_seconds,
        ):
            embedding = model.encode_document(interval_text)
            vectors_by_interval.setdefault(interval_index, []).append(
                (episode, embedding)
            )
        if progress and build_task is not None:
            progress.update(build_task, advance=1)

    index_task = None
    if progress:
        index_task = progress.add_task("Indexing subtitle intervals", total=len(vectors_by_interval))

    reduced: dict[int, IntervalSubtitleIndex] = {}
    for interval_index, episode_vectors in vectors_by_interval.items():
        episodes = [episode for episode, _ in episode_vectors]
        matrix = np.vstack([vector for _, vector in episode_vectors]).astype(np.float32)
        ann_index = _build_interval_index(matrix)
        reduced[interval_index] = IntervalSubtitleIndex(episodes=episodes, index=ann_index)
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


def _window_texts(srt_path: Path, window_seconds: int, overlap_seconds: int) -> Iterable[tuple[int, str]]:
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
        yield interval_index, interval_text


def _score_segments(segments: list[SegmentRecord],
    subtitle_indexes: dict[int, IntervalSubtitleIndex],
    model: SentenceTransformerModel, top_ks: list[int], max_failures: int,
    segment_duration_seconds: int = 30,
    subtitle_overlap_seconds: int = 5,
    max_results_per_query: int = 10,
    progress: Progress | None = None) -> dict:
    window_config = make_window_config(segment_duration_seconds, subtitle_overlap_seconds)
    top_hits = {k: 0 for k in top_ks}
    reciprocal_rank_sum = 0.0
    evaluated = 0
    missing_interval = 0
    failures = []

    score_task = None
    if progress:
        score_task = progress.add_task("Scoring transcript segments", total=len(segments))

    for segment in segments:
        mapped_interval = map_segment_index_to_window_index(
            segment.segment_index,
            segment_duration_seconds,
            window_config.stride_seconds,
        )
        query_embedding = model.encode_query(segment.transcript_text)
        per_episode_best: dict[EpisodeKey, float] = {}
        mapped_window_distances: list[float] = []

        mapped_data = subtitle_indexes.get(mapped_interval)
        if mapped_data:
            for episode, raw_distance, interval_index in _query_interval_index(
                mapped_data, query_embedding, mapped_interval, max_results_per_query
            ):
                mapped_window_distances.append(raw_distance)
                adjusted_distance = distance_with_window_penalty(
                    raw_distance,
                    interval_index,
                    mapped_interval,
                )
                current = per_episode_best.get(episode)
                if current is None or adjusted_distance < current:
                    per_episode_best[episode] = adjusted_distance

        if should_expand_to_neighbor_windows(mapped_window_distances):
            for interval_index in neighbor_window_indexes(mapped_interval):
                if interval_index == mapped_interval:
                    continue
                interval_data = subtitle_indexes.get(interval_index)
                if not interval_data:
                    continue
                for episode, raw_distance, candidate_interval in _query_interval_index(
                    interval_data,
                    query_embedding,
                    interval_index,
                    max_results_per_query,
                ):
                    adjusted_distance = distance_with_window_penalty(
                        raw_distance,
                        candidate_interval,
                        mapped_interval,
                    )
                    current = per_episode_best.get(episode)
                    if current is None or adjusted_distance < current:
                        per_episode_best[episode] = adjusted_distance

        if not per_episode_best:
            missing_interval += 1
            if progress and score_task is not None:
                progress.update(score_task, advance=1)
            continue

        ranked = [episode for episode, _ in sorted(
            per_episode_best.items(),
            key=lambda item: (item[1], item[0]),
        )]

        rank = _first_expected_rank(ranked, segment.expected)
        evaluated += 1
        if rank:
            reciprocal_rank_sum += 1.0 / rank
            for k in top_ks:
                if rank <= k:
                    top_hits[k] += 1
        elif len(failures) < max_failures:
            failures.append({
                "video_path": segment.video_path,
                "segment_index": segment.segment_index,
                "expected": [str(ep) for ep in sorted(segment.expected)],
                "top_predictions": [str(ep) for ep in ranked[:max(top_ks)]],
            })
        if progress and score_task is not None:
            progress.update(score_task, advance=1)

    top_accuracy = {
        f"top_{k}": (top_hits[k] / evaluated if evaluated else 0.0)
        for k in top_ks
    }
    return {
        "segments_total": len(segments),
        "segments_evaluated": evaluated,
        "segments_missing_interval": missing_interval,
        "mrr": reciprocal_rank_sum / evaluated if evaluated else 0.0,
        "accuracy": top_accuracy,
        "top_hits": {f"top_{k}": top_hits[k] for k in top_ks},
        "failure_examples": failures,
    }


def _query_interval_index(
    interval_data: IntervalSubtitleIndex,
    query_embedding: np.ndarray,
    interval_index: int,
    max_results_per_query: int,
) -> list[tuple[EpisodeKey, float, int]]:
    count = interval_data.index.get_current_count()
    if count <= 0:
        return []
    k = min(max_results_per_query, count)
    ids_by_q, dists_by_q = interval_data.index.knn_query(
        query_embedding, k=k, num_threads=1, filter=None
    )
    ids, dists = ids_by_q[0], dists_by_q[0]
    return [
        (interval_data.episodes[int(id)], float(distance), interval_index)
        for id, distance in zip(ids, dists)
    ]


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
