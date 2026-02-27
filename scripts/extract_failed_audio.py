#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CHUNK_PATH_START_RE = re.compile(r"\.AT(?P<start_ms>\d+)ms\.wav$")


@dataclass(frozen=True)
class FailedSegment:
    video_path: Path
    segment_index: int | None
    profile: str
    failure_type: str
    duration_seconds: float
    start_seconds: float
    source_log: Path
    source_line: int
    payload: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio clips for failed transcription intervals."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory (for example: data/labeled-xscribes/30Rock).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where extracted clips should be written. Defaults to <dataset-dir>/debug/failed-audio.",
    )
    parser.add_argument(
        "--logs-glob",
        default=".work/**/failure-logs/transcription-failures-*.jsonl",
        help="Glob pattern (relative to dataset-dir) used to find failure log files.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Optional profile filter. Example: --profiles aligned left right random",
    )
    parser.add_argument(
        "--failure-types",
        nargs="+",
        default=["empty_transcript", "transcribe_exception", "batch_transcribe_exception"],
        help="Failure types to include.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional limit on number of clips to extract.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output clips if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be extracted without invoking ffmpeg.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg binary name/path.",
    )
    return parser.parse_args()


def _extract_start_seconds(payload: dict) -> float | None:
    effective = payload.get("effective_offset_seconds")
    if isinstance(effective, (int, float)):
        return max(0.0, float(effective))

    chunk_path = payload.get("chunk_path")
    if isinstance(chunk_path, str):
        match = CHUNK_PATH_START_RE.search(chunk_path)
        if match:
            start_ms = int(match.group("start_ms"))
            return start_ms / 1000.0

    segment_index = payload.get("segment_index")
    duration_seconds = payload.get("duration_seconds")
    if isinstance(segment_index, int) and isinstance(duration_seconds, (int, float)):
        return max(0.0, float(segment_index) * float(duration_seconds))
    return None


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "unknown"


def _discover_failures(
    dataset_dir: Path,
    logs_glob: str,
    profiles: set[str] | None,
    failure_types: set[str],
) -> tuple[list[FailedSegment], list[str]]:
    failures: list[FailedSegment] = []
    errors: list[str] = []
    seen_keys: set[tuple[str, int, int, str, str]] = set()

    for log_path in sorted(dataset_dir.glob(logs_glob)):
        if not log_path.is_file():
            continue
        with log_path.open(encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{log_path}:{line_num}: invalid json ({exc})")
                    continue

                profile = str(payload.get("misalignment_profile") or "aligned")
                failure_type = str(payload.get("failure_type") or "unknown")
                if profiles and profile not in profiles:
                    continue
                if failure_type not in failure_types:
                    continue

                video_raw = payload.get("video_path")
                if not isinstance(video_raw, str):
                    errors.append(f"{log_path}:{line_num}: missing video_path")
                    continue
                video_path = Path(video_raw)

                duration = payload.get("duration_seconds")
                if not isinstance(duration, (int, float)) or float(duration) <= 0:
                    errors.append(f"{log_path}:{line_num}: invalid duration_seconds")
                    continue

                start_seconds = _extract_start_seconds(payload)
                if start_seconds is None:
                    errors.append(f"{log_path}:{line_num}: could not determine start_seconds")
                    continue

                segment_index = payload.get("segment_index")
                segment_index_int = segment_index if isinstance(segment_index, int) else -1
                key = (
                    str(video_path),
                    int(round(start_seconds * 1000)),
                    int(round(float(duration) * 1000)),
                    profile,
                    failure_type,
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                failures.append(
                    FailedSegment(
                        video_path=video_path,
                        segment_index=segment_index_int if segment_index_int >= 0 else None,
                        profile=profile,
                        failure_type=failure_type,
                        duration_seconds=float(duration),
                        start_seconds=float(start_seconds),
                        source_log=log_path,
                        source_line=line_num,
                        payload=payload,
                    )
                )

    return failures, errors


def _output_path(output_dir: Path, failure: FailedSegment) -> Path:
    start_ms = int(round(failure.start_seconds * 1000))
    dur_ms = int(round(failure.duration_seconds * 1000))
    segment = f"seg{failure.segment_index:04d}" if failure.segment_index is not None else "segXXXX"
    video_hash = hashlib.sha1(str(failure.video_path).encode("utf-8")).hexdigest()[:8]
    base_name = _sanitize(failure.video_path.stem)
    file_name = (
        f"{failure.profile}__{failure.failure_type}__{video_hash}_{base_name}"
        f"__{segment}__at{start_ms}ms__dur{dur_ms}ms.wav"
    )
    return output_dir / failure.profile / file_name


def _run_ffmpeg(ffmpeg_bin: str, failure: FailedSegment, destination: Path) -> subprocess.CompletedProcess:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{failure.start_seconds:.3f}",
        "-t",
        f"{failure.duration_seconds:.3f}",
        "-i",
        str(failure.video_path),
        "-vn",
        "-sn",
        "-dn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        str(destination),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        print(f"error: dataset dir does not exist: {dataset_dir}", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (dataset_dir / "debug" / "failed-audio").resolve()
    )

    profiles = {value.strip() for value in args.profiles} if args.profiles else None
    failure_types = {value.strip() for value in args.failure_types}
    failures, parse_errors = _discover_failures(
        dataset_dir=dataset_dir,
        logs_glob=args.logs_glob,
        profiles=profiles,
        failure_types=failure_types,
    )

    if args.max_clips is not None and args.max_clips >= 0:
        failures = failures[: args.max_clips]

    print(f"dataset: {dataset_dir}")
    print(f"output:  {output_dir}")
    print(f"logs:    {args.logs_glob}")
    print(f"clips:   {len(failures)}")
    if parse_errors:
        print(f"parse warnings: {len(parse_errors)}")
        for error in parse_errors[:10]:
            print(f"  - {error}")
        if len(parse_errors) > 10:
            print(f"  - ... {len(parse_errors) - 10} more")

    written = 0
    skipped_existing = 0
    failed = 0
    manifest_path = output_dir / "extracted.jsonl"
    manifest_lines: list[str] = []
    for item in failures:
        destination = _output_path(output_dir, item)
        if destination.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        if args.dry_run:
            print(
                f"[dry-run] {item.video_path} :: profile={item.profile} "
                f"segment={item.segment_index} start={item.start_seconds:.3f}s "
                f"dur={item.duration_seconds:.3f}s -> {destination}"
            )
            written += 1
            continue

        result = _run_ffmpeg(args.ffmpeg_bin, item, destination)
        if result.returncode != 0:
            failed += 1
            stderr = (result.stderr or "").strip().replace("\n", " | ")
            print(
                f"[ffmpeg-failed] {item.video_path} seg={item.segment_index} "
                f"start={item.start_seconds:.3f}s dur={item.duration_seconds:.3f}s :: {stderr}",
                file=sys.stderr,
            )
            continue

        written += 1
        manifest_lines.append(
            json.dumps(
                {
                    "output_path": str(destination),
                    "video_path": str(item.video_path),
                    "profile": item.profile,
                    "failure_type": item.failure_type,
                    "segment_index": item.segment_index,
                    "start_seconds": item.start_seconds,
                    "duration_seconds": item.duration_seconds,
                    "source_log": str(item.source_log),
                    "source_line": item.source_line,
                },
                ensure_ascii=False,
            )
        )

    if not args.dry_run and manifest_lines:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as fh:
            for line in manifest_lines:
                fh.write(line)
                fh.write("\n")

    print(
        "done: "
        f"written={written}, skipped_existing={skipped_existing}, "
        f"ffmpeg_failed={failed}"
    )
    if not args.dry_run and manifest_lines:
        print(f"manifest: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
