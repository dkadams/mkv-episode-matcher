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
from mkv_episode_matcher.series import Series, SeriesDirectoryProcessor, get_specified_episodes

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

    output_root = _resolve_output_root(config, series, all_series_dirs)
    output_root.mkdir(parents=True, exist_ok=True)
    transcriptions_out = output_root / "transcriptions" / "text"
    transcriptions_out.mkdir(parents=True, exist_ok=True)
    subtitles_out = output_root / "subtitles" / "srt"
    subtitles_out.mkdir(parents=True, exist_ok=True)

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
        info_by_path, transcriptions = matcher.get_transcriptions(progress, filtered_videos)
    total_videos = 0
    kept_videos = 0
    copied_transcriptions = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_out:
        for path in filtered_videos:
            total_videos += 1
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

            transcript_path = transcriptions.get(path)
            if not transcript_path or not transcript_path.exists():
                _write_exclusion(exclusions_path, {
                    "video_path": str(path),
                    "episodes": [str(key) for key in episode_keys],
                    "reason": "missing_transcript",
                })
                continue

            copied_path = transcriptions_out / transcript_path.name
            shutil.copy2(transcript_path, copied_path)
            copied_transcriptions += 1

            payload = {
                "series_name": series.name,
                "episodes": [str(key) for key in episode_keys],
                "video_path": str(path),
                "transcription_path": str(copied_path.relative_to(output_root)),
                "segment_duration": series.segment_duration,
                "segments_per_minute": config.args.segments_per_minute,
                "duration_minutes": video_info.minutes,
            }
            if len(episode_keys) == 1:
                payload["episode"] = str(episode_keys[0])
            manifest_out.write(json.dumps(payload, ensure_ascii=False))
            manifest_out.write("\n")
            kept_videos += 1

    copied_subtitles = _copy_subtitles(series, subtitles_out, specified_keys, exclusions_path)

    with meta_path.open("w", encoding="utf-8") as meta_out:
        json.dump({
            "schema_version": 2,
            "dataset_type": "mkv-episode-matcher-eval",
            "series_name": series.name,
            "source_series_dir": str(series.dir),
            "segment_duration": series.segment_duration,
            "segments_per_minute": config.args.segments_per_minute,
            "random_seed": series.random_seed,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "paths": {
                "manifest": "manifest.jsonl",
                "transcriptions_dir": "transcriptions/text",
                "subtitles_dir": "subtitles/srt",
                "exclusions": "exclusions.jsonl",
            },
            "counts": {
                "videos_total": total_videos,
                "videos_included": kept_videos,
                "transcriptions_written": copied_transcriptions,
                "subtitles_copied": copied_subtitles,
            },
        }, meta_out, ensure_ascii=False, indent=2)

    console.print(
        f"[bold green]Dataset written to {output_root}[/bold green]\n"
        f"Videos: {kept_videos}/{total_videos}"
    )


def _resolve_output_root(config: Configuration, series, all_series_dirs) -> Path:
    base = Path(config.args.output_dir).expanduser().resolve()
    if len(all_series_dirs) == 1:
        return base
    return base / series.dir.name


def _copy_subtitles(series, subtitles_out: Path, specified_keys, exclusions_path: Path):
    if not series.subtitles_dir.exists():
        _write_exclusion(exclusions_path, {
            "reason": "missing_subtitles_dir",
            "subtitles_dir": str(series.subtitles_dir),
        })
        return

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
        shutil.copy2(srt, dest)
        copied += 1
    return copied


def _write_exclusion(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False))
        out.write("\n")
