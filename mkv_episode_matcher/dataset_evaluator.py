import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pysubs2
from rich.console import Console
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.embedding_model import SentenceTransformerModel
from mkv_episode_matcher.episode import EpisodeKey

console = Console()

EPISODE_PATTERN = re.compile(r"^S(?P<season>\d+)E(?P<episode>\d+)$")


@dataclass(frozen=True)
class SegmentRecord:
    segment_index: int
    transcript_text: str
    expected: set[EpisodeKey]
    video_path: str


def evaluate_dataset(config: Configuration):
    dataset_dir = Path(config.args.dataset_dir).expanduser().resolve()
    segments_path = dataset_dir / "segments.jsonl"
    subtitles_dir = dataset_dir / "subtitles"
    if not segments_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {segments_path}")
    if not subtitles_dir.exists():
        raise FileNotFoundError(f"Subtitles directory not found: {subtitles_dir}")

    interval_seconds = _resolve_segment_duration(
        dataset_dir,
        config.args.segment_duration,
    )
    top_ks = sorted({k for k in config.args.top_k if k > 0})
    if not top_ks:
        raise ValueError("At least one positive --top-k value is required")

    segments = list(_load_segments(segments_path, limit=config.args.limit))
    if not segments:
        console.print("[orange1]No segments to evaluate.")
        return

    model = SentenceTransformerModel()
    subtitle_vectors = _build_subtitle_vectors(subtitles_dir, interval_seconds, model)
    report = _score_segments(
        segments=segments,
        subtitle_vectors=subtitle_vectors,
        model=model,
        top_ks=top_ks,
        max_failures=config.args.show_failures,
    )

    report.update({
        "dataset_dir": str(dataset_dir),
        "segment_duration": interval_seconds,
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


def _load_segments(segments_path: Path, limit: int | None) -> Iterable[SegmentRecord]:
    with segments_path.open("r", encoding="utf-8") as segments_in:
        for idx, line in enumerate(segments_in):
            if limit is not None and idx >= limit:
                break
            payload = json.loads(line)
            transcript_text = str(payload.get("transcript_text", "")).strip()
            if not transcript_text:
                continue
            expected_ids = payload.get("episodes") or [payload.get("episode")]
            expected = {
                _parse_episode_key(episode_id)
                for episode_id in expected_ids
                if episode_id
            }
            if not expected:
                continue
            yield SegmentRecord(
                segment_index=int(payload["segment_index"]),
                transcript_text=transcript_text,
                expected=expected,
                video_path=str(payload.get("video_path", "")),
            )


def _build_subtitle_vectors(subtitles_dir: Path, interval_seconds: int,
    model: SentenceTransformerModel) -> dict[int, tuple[list[EpisodeKey], np.ndarray]]:
    vectors_by_interval: dict[int, list[tuple[EpisodeKey, np.ndarray]]] = {}

    for srt_path in sorted(subtitles_dir.rglob("*.srt")):
        episode = EpisodeKey.from_srt_path(srt_path)
        if not episode:
            continue
        for interval_index, interval_text in _interval_texts(srt_path, interval_seconds):
            embedding = model.encode_document(interval_text)
            vectors_by_interval.setdefault(interval_index, []).append(
                (episode, embedding)
            )

    reduced: dict[int, tuple[list[EpisodeKey], np.ndarray]] = {}
    for interval_index, episode_vectors in vectors_by_interval.items():
        episodes = [episode for episode, _ in episode_vectors]
        matrix = np.vstack([vector for _, vector in episode_vectors]).astype(np.float32)
        reduced[interval_index] = (episodes, matrix)
    return reduced


def _interval_texts(srt_path: Path, interval_seconds: int) -> Iterable[tuple[int, str]]:
    subs = pysubs2.load(str(srt_path), format_="srt")
    if not subs:
        return
    max_ts = max(int(sub.end) for sub in subs)
    interval_ms = interval_seconds * 1000
    interval_count = math.ceil(max_ts / interval_ms)
    for interval_index in range(interval_count):
        start = interval_index * interval_ms
        interval = range(start, start + interval_ms)
        interval_text = " ".join(
            sub.plaintext for sub in subs
            if sub.start in interval or sub.end in interval
        )
        yield interval_index, interval_text


def _score_segments(segments: list[SegmentRecord],
    subtitle_vectors: dict[int, tuple[list[EpisodeKey], np.ndarray]],
    model: SentenceTransformerModel, top_ks: list[int], max_failures: int) -> dict:
    top_hits = {k: 0 for k in top_ks}
    reciprocal_rank_sum = 0.0
    evaluated = 0
    missing_interval = 0
    failures = []

    for segment in segments:
        interval_data = subtitle_vectors.get(segment.segment_index)
        if not interval_data:
            missing_interval += 1
            continue

        episodes, matrix = interval_data
        query = model.encode_query(segment.transcript_text)
        scores = matrix @ query
        ranked_indexes = np.argsort(-scores)
        ranked = [episodes[i] for i in ranked_indexes]

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
    table.add_row("Segments total", str(report["segments_total"]))
    table.add_row("Segments evaluated", str(report["segments_evaluated"]))
    table.add_row("Segments missing interval", str(report["segments_missing_interval"]))
    for k in top_ks:
        acc = report["accuracy"][f"top_{k}"]
        hits = report["top_hits"][f"top_{k}"]
        table.add_row(f"Top-{k} accuracy", f"{acc:.4f} ({hits})")
    table.add_row("MRR", f"{report['mrr']:.4f}")
    console.print(table)
