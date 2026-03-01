import dataclasses
import json
import shutil
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import IndexedEpisodeMatcher
from mkv_episode_matcher.misalignment import MisalignmentPolicy
from mkv_episode_matcher.series import Series, SeriesDirectoryProcessor, get_specified_episodes
from mkv_episode_matcher.windowing import (
    make_window_config,
    resolve_subtitle_overlap_seconds,
)

console = Console()


def collect_dataset(config: Configuration):
    processor = SeriesDirectoryProcessor(config)
    for series_dir in processor.series_dirs:
        series = Series.from_dir(series_dir)
        if not series:
            continue
        _collect_series_dataset(config, series, processor.series_dirs)


def _collect_series_dataset(config: Configuration, series, all_series_dirs):
    if config.args.segment_duration is not None:
        series = dataclasses.replace(series, segment_duration=config.args.segment_duration)
    subtitle_overlap_seconds = resolve_subtitle_overlap_seconds(config.args, series)
    make_window_config(series.segment_duration, subtitle_overlap_seconds)

    misalign_profiles = list(dict.fromkeys(config.args.misalign_profiles or []))
    if config.args.misalign_min_seconds < 0:
        raise ValueError("--misalign-min-seconds must be >= 0")
    if config.args.misalign_max_seconds < config.args.misalign_min_seconds:
        raise ValueError("--misalign-max-seconds must be >= --misalign-min-seconds")
    if not config.args.include_aligned and not misalign_profiles:
        raise ValueError("Enable --include-aligned or provide --misalign-profiles")

    output_root = _resolve_output_root(config, series, all_series_dirs)
    output_root.mkdir(parents=True, exist_ok=True)
    aligned_out = output_root / "transcriptions" / "text"
    aligned_out.mkdir(parents=True, exist_ok=True)
    misaligned_out = output_root / "transcriptions" / "misaligned"
    misaligned_out.mkdir(parents=True, exist_ok=True)
    subtitles_out = output_root / "subtitles" / "srt"
    subtitles_out.mkdir(parents=True, exist_ok=True)

    work_dir = output_root / ".work" / "misaligned"
    work_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_root / "meta.json"
    manifest_path = output_root / "manifest.jsonl"
    exclusions_path = output_root / "exclusions.jsonl"
    exclusions_path.write_text("", encoding="utf-8")

    specified = get_specified_episodes(config, series)
    specified_keys = {episode.key() for episode in specified}

    video_files = list(IndexedEpisodeMatcher._collect_files([series.dir]))
    filtered_videos = []
    for path in video_files:
        episode_keys = EpisodeKey.from_vid_path(path)
        if not episode_keys:
            _write_exclusion(exclusions_path, {
                "video_path": str(path),
                "reason": "unparseable_episode",
            })
            continue
        if specified_keys and not any(key in specified_keys for key in episode_keys):
            _write_exclusion(exclusions_path, {
                "video_path": str(path),
                "episodes": [str(key) for key in episode_keys],
                "reason": "episode_filtered_out",
            })
            continue
        filtered_videos.append(path)

    if not filtered_videos:
        console.print(f"[orange1]No videos found to process for {series.name}.")
        return

    matcher = IndexedEpisodeMatcher(config, series)
    with Progress() as progress:
        info_by_path, _ = matcher.get_transcriptions(progress, filtered_videos)

    aligned_transcriptions: dict[Path, Path] = {}
    if config.args.include_aligned:
        with Progress() as progress:
            _, transcriptions = matcher.get_transcriptions(progress, filtered_videos)
        aligned_transcriptions = _copy_transcription_map(transcriptions, aligned_out)

    misalign_seed = (config.args.misalign_seed
                     if config.args.misalign_seed is not None
                     else series.random_seed)
    misaligned_transcriptions: dict[str, dict[Path, Path]] = {}
    for profile in misalign_profiles:
        profile_work_dir = work_dir / profile
        profile_work_dir.mkdir(parents=True, exist_ok=True)

        per_segment = (profile == "random") or config.args.misalign_per_segment
        policy = MisalignmentPolicy(
            profile=profile,
            min_seconds=config.args.misalign_min_seconds,
            max_seconds=config.args.misalign_max_seconds,
            seed=misalign_seed,
            per_segment=per_segment,
        )
        with Progress() as progress:
            _, transcriptions = matcher.get_transcriptions(
                progress,
                filtered_videos,
                no_transcription_cache=True,
                transcription_output_dir=profile_work_dir,
                misalignment_policy=policy,
                variant_id=f"misaligned:{profile}",
            )
        profile_out_dir = misaligned_out / profile
        profile_out_dir.mkdir(parents=True, exist_ok=True)
        misaligned_transcriptions[profile] = _copy_transcription_map(transcriptions, profile_out_dir)

    rows_written = 0
    aligned_count = 0
    misaligned_count = 0
    misaligned_by_profile = {profile: 0 for profile in misalign_profiles}
    videos_with_any_variant = set()

    with manifest_path.open("w", encoding="utf-8") as manifest_out:
        for path in filtered_videos:
            episode_keys = EpisodeKey.from_vid_path(path)
            if not episode_keys:
                continue

            video_info = info_by_path.get(path)
            if not video_info:
                _write_exclusion(exclusions_path, {
                    "video_path": str(path),
                    "episodes": [str(key) for key in episode_keys],
                    "reason": "missing_video_info",
                })
                continue

            aligned_rel_path = None
            if config.args.include_aligned:
                aligned_file = aligned_transcriptions.get(path)
                if not aligned_file or not aligned_file.exists():
                    _write_exclusion(exclusions_path, {
                        "video_path": str(path),
                        "episodes": [str(key) for key in episode_keys],
                        "variant_profile": "aligned",
                        "reason": "missing_transcript",
                    })
                else:
                    aligned_rel_path = str(aligned_file.relative_to(output_root))
                    payload = {
                        "series_name": series.name,
                        "episodes": [str(key) for key in episode_keys],
                        "video_path": str(path),
                        "transcription_path": aligned_rel_path,
                        "segment_duration": series.segment_duration,
                        "subtitle_overlap_seconds": subtitle_overlap_seconds,
                        "segments_per_minute": config.args.segments_per_minute,
                        "duration_minutes": video_info.minutes,
                        "variant_type": "aligned",
                        "variant_profile": "aligned",
                    }
                    if len(episode_keys) == 1:
                        payload["episode"] = str(episode_keys[0])
                    manifest_out.write(json.dumps(payload, ensure_ascii=False))
                    manifest_out.write("\n")
                    rows_written += 1
                    aligned_count += 1
                    videos_with_any_variant.add(path)

            for profile in misalign_profiles:
                profile_file = misaligned_transcriptions.get(profile, {}).get(path)
                if not profile_file or not profile_file.exists():
                    _write_exclusion(exclusions_path, {
                        "video_path": str(path),
                        "episodes": [str(key) for key in episode_keys],
                        "variant_profile": profile,
                        "reason": "missing_transcript",
                    })
                    continue

                per_segment = (profile == "random") or config.args.misalign_per_segment
                payload = {
                    "series_name": series.name,
                    "episodes": [str(key) for key in episode_keys],
                    "video_path": str(path),
                    "transcription_path": str(profile_file.relative_to(output_root)),
                    "segment_duration": series.segment_duration,
                    "subtitle_overlap_seconds": subtitle_overlap_seconds,
                    "segments_per_minute": config.args.segments_per_minute,
                    "duration_minutes": video_info.minutes,
                    "variant_type": "misaligned",
                    "variant_profile": profile,
                    "source_transcription_path": aligned_rel_path,
                    "misalignment": {
                        "min_seconds": config.args.misalign_min_seconds,
                        "max_seconds": config.args.misalign_max_seconds,
                        "seed": misalign_seed,
                        "mode": ("per_segment" if per_segment else "file_constant"),
                        "direction": profile,
                    },
                }
                if len(episode_keys) == 1:
                    payload["episode"] = str(episode_keys[0])
                manifest_out.write(json.dumps(payload, ensure_ascii=False))
                manifest_out.write("\n")
                rows_written += 1
                misaligned_count += 1
                misaligned_by_profile[profile] += 1
                videos_with_any_variant.add(path)

    copied_subtitles = _copy_subtitles(series, subtitles_out, specified_keys, exclusions_path)

    with meta_path.open("w", encoding="utf-8") as meta_out:
        json.dump({
            "schema_version": 2,
            "dataset_type": "mkv-episode-matcher-eval",
            "series_name": series.name,
            "source_series_dir": str(series.dir),
            "segment_duration": series.segment_duration,
            "subtitle_overlap_seconds": subtitle_overlap_seconds,
            "segments_per_minute": config.args.segments_per_minute,
            "random_seed": series.random_seed,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "misalignment_profiles_enabled": misalign_profiles,
            "misalignment_defaults": {
                "min_seconds": config.args.misalign_min_seconds,
                "max_seconds": config.args.misalign_max_seconds,
                "seed": misalign_seed,
            },
            "paths": {
                "manifest": "manifest.jsonl",
                "transcriptions_dir": "transcriptions/text",
                "subtitles_dir": "subtitles/srt",
                "exclusions": "exclusions.jsonl",
            },
            "counts": {
                "videos_total": len(filtered_videos),
                "videos_included": len(videos_with_any_variant),
                "manifest_rows": rows_written,
                "transcriptions_written": aligned_count + misaligned_count,
                "aligned_transcriptions_written": aligned_count,
                "misaligned_transcriptions_written": misaligned_count,
                "misaligned_by_profile": misaligned_by_profile,
                "subtitles_copied": copied_subtitles,
            },
        }, meta_out, ensure_ascii=False, indent=2)

    console.print(
        f"[bold green]Dataset written to {output_root}[/bold green]\n"
        f"Videos: {len(videos_with_any_variant)}/{len(filtered_videos)}, "
        f"Rows: {rows_written}"
    )


def _copy_transcription_map(transcriptions: dict[Path, Path], destination_dir: Path) -> dict[Path, Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source_video, transcript_path in transcriptions.items():
        if not transcript_path.exists():
            continue
        dest = destination_dir / transcript_path.name
        shutil.copyfile(transcript_path, dest)
        copied[source_video] = dest
    return copied


def _resolve_output_root(config: Configuration, series, all_series_dirs) -> Path:
    base = Path(config.args.output_dir).expanduser().resolve()
    if len(all_series_dirs) == 1:
        return base
    return base / series.dir.name


def _copy_subtitles(series, subtitles_out: Path, specified_keys, exclusions_path: Path) -> int:
    if not series.subtitles_dir.exists():
        _write_exclusion(exclusions_path, {
            "reason": "missing_subtitles_dir",
            "subtitles_dir": str(series.subtitles_dir),
        })
        return 0

    copied = 0
    for srt in series.subtitles_dir.rglob("*.srt"):
        episode_key = EpisodeKey.from_srt_path(srt)
        if not episode_key:
            _write_exclusion(exclusions_path, {
                "subtitle_path": str(srt),
                "reason": "unparseable_subtitle_episode",
            })
            continue
        if specified_keys and episode_key not in specified_keys:
            continue
        dest = subtitles_out / srt.name
        shutil.copyfile(srt, dest)
        copied += 1
    return copied


def _write_exclusion(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False))
        out.write("\n")
