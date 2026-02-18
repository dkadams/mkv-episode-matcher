import dataclasses
import json
import math
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
    subtitles_out = output_root / "subtitles"
    subtitles_out.mkdir(parents=True, exist_ok=True)

    meta_path = output_root / "meta.json"
    segments_path = output_root / "segments.jsonl"
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

    segment_indexes = matcher.get_segment_selection(info_by_path.values())

    total_segments = 0
    kept_segments = 0
    with segments_path.open("w", encoding="utf-8") as segments_out:
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

            transcript_path = transcriptions.get(path)
            if not transcript_path or not transcript_path.exists():
                _write_exclusion(exclusions_path, {
                    "video_path": str(path),
                    "episodes": [str(key) for key in episode_keys],
                    "reason": "missing_transcript",
                })
                continue

            with transcript_path.open("r", encoding="utf-8") as transcript_in:
                transcript = json.load(transcript_in)

            target_segments = math.ceil(video_info.minutes * config.args.segments_per_minute)
            selected = segment_indexes[:target_segments]
            for segment_index in selected:
                total_segments += 1
                text = transcript.get(str(segment_index)) or transcript.get(segment_index)
                if not text:
                    _write_exclusion(exclusions_path, {
                        "video_path": str(path),
                        "episodes": [str(key) for key in episode_keys],
                        "segment_index": segment_index,
                        "reason": "missing_segment_text",
                    })
                    continue

                start_sec = segment_index * series.segment_duration
                end_sec = start_sec + series.segment_duration
                payload = {
                    "series_name": series.name,
                    "episodes": [str(key) for key in episode_keys],
                    "video_path": str(path),
                    "segment_index": segment_index,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "segment_duration": series.segment_duration,
                    "transcript_text": text,
                }
                if len(episode_keys) == 1:
                    payload["episode"] = str(episode_keys[0])
                segments_out.write(json.dumps(payload, ensure_ascii=False))
                segments_out.write("\n")
                kept_segments += 1

    _copy_subtitles(series, subtitles_out, specified_keys, exclusions_path)

    with meta_path.open("w", encoding="utf-8") as meta_out:
        json.dump({
            "series_name": series.name,
            "series_dir": str(series.dir),
            "segment_duration": series.segment_duration,
            "segments_per_minute": config.args.segments_per_minute,
            "random_seed": series.random_seed,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "segments_total": total_segments,
            "segments_kept": kept_segments,
        }, meta_out, ensure_ascii=False, indent=2)

    console.print(
        f"[bold green]Dataset written to {output_root}[/bold green]\n"
        f"Segments: {kept_segments}/{total_segments}"
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


def _write_exclusion(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False))
        out.write("\n")

