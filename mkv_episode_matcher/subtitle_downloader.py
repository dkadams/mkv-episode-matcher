import shutil
import json
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from opensubtitlescom import OpenSubtitles, opensubtitles, \
    OpenSubtitlesException
from opensubtitlescom.responses import Subtitle, DownloadResponse
from rich.console import Console
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import Episode, EpisodeKey
from mkv_episode_matcher.series import SeriesDirectoryProcessor, Series, \
    get_specified_episodes
from mkv_episode_matcher.subtitle_quality import (
    SubtitleQualitySettings,
    candidate_quality_diagnostics,
    choose_best_attempt,
    current_timestamp,
    evaluate_subtitle_signals,
    resolve_subtitle_quality_settings,
    save_episode_quality,
    subtitle_paths_by_episode,
    append_quarantine_event,
)

console = Console()

def download_subtitles(config):
    def series_downloader(series):
        console.print(f"[bold green]Downloading subtitles for: {series.name}")

        episodes = get_specified_episodes(config, series)

        downloader = OpenSubtitlesDownloader(config, series)
        for episode in episodes:
            downloader.download(episode)
        downloader.write_quality_report()

        console.print(f"[bold green]Subtitles downloaded")

    SeriesDirectoryProcessor(config).process_series(series_downloader)

class OpenSubtitlesDownloader:
    def __init__(self, config: Configuration, series: Series):
        if not config.has_required_settings():
            console.print("[bold red]Error: missing configuration settings. Run mkv-episode-matcher config")
            return

        self.config = config
        self.series = series
        self.quality_settings = resolve_subtitle_quality_settings(config.args, series)
        self.quality_rows: list[dict] = []
        
        api_config = config.stored["api"]
        open_subtitles_api_key = api_config.get("open_subtitles_api_key")
        open_subtitles_user_agent = api_config.get("open_subtitles_user_agent")
        open_subtitles_username = api_config.get("open_subtitles_username")
        open_subtitles_password = api_config.get("open_subtitles_password")

        self.client = OpenSubtitles(open_subtitles_user_agent, open_subtitles_api_key)
        self.client.login(open_subtitles_username, open_subtitles_password)

    def download(self, episode: Episode):
        console.print(f"Preparing to download series: {self.series.name} - {episode.short_str()}...")

        existing_subtitle = self.find_existing_subtitle(episode)
        episode_key = episode.key()
        existing_attempt = None
        if existing_subtitle and self.quality_settings.enabled:
            existing_attempt = self._evaluate_existing_subtitle(episode, existing_subtitle)
            if (
                not self.config.args.refresh
                and not bool(existing_attempt["quality"]["hard_fail"])
            ):
                console.print(f"[bold green]\tsubtitle exists and passed quality checks: {existing_subtitle}. Skipping download.[/bold green]")
                self._persist_quality(
                    episode_key=episode_key,
                    selected_attempt=existing_attempt,
                    attempts=[existing_attempt],
                )
                return
        elif existing_subtitle and not self.config.args.refresh:
            console.print(f"[bold green]\tsubtitle exists: {existing_subtitle}. Skipping download.[/bold green]")
            return

        response = self.client.search(parent_tmdb_id=self.series.detail["id"],
                                      tmdb_id=episode.tmdb_id, languages="en")
        if len(response.data) == 0:
            console.print(f"No subtitles found for {self.series.name} - {episode.short_str()}")
            if existing_attempt:
                selected_existing = dict(existing_attempt)
                selected_existing["selection_verdict"] = (
                    "quarantined"
                    if bool(existing_attempt.get("quality", {}).get("hard_fail", False))
                    else "pass"
                )
                self._persist_quality(
                    episode_key=episode_key,
                    selected_attempt=selected_existing,
                    attempts=[existing_attempt],
                )
            return

        console.print(f"[green]Found {len(response.data)} subtitles for {self.series.name} - {episode.short_str()}")
        subtitles = sorted(response.data,
                           key=lambda subtitle: subtitle.download_count
                                                + subtitle.new_download_count, reverse=True)
        self.print_subtitles_table(episode, subtitles)

        srt_filename = f"{episode.short_str()}.srt"
        srt_filepath = self.series.subtitles_dir / srt_filename
        if not self.series.subtitles_dir.exists():
            self.series.subtitles_dir.mkdir(parents=True, exist_ok=True)

        if not self.quality_settings.enabled:
            selected_subtitle = subtitles[0]
            downloaded = self._download_candidate_subtitle(selected_subtitle)
            if downloaded is None:
                console.print(f"[bold red]\tFailed to download subtitle for {episode.short_str()}[/bold red]")
                return
            shutil.move(downloaded, srt_filepath)
            logger.info("Subtitle saved to {}", srt_filepath)
            opensubs_filepath = srt_filepath.with_suffix(".opensubtitles")
            with opensubs_filepath.open("w", encoding="utf-8") as json_out:
                json_out.write(selected_subtitle.to_json())
            logger.info("Subtitle metadata saved to {}", opensubs_filepath)
            return
        attempts: list[dict] = []
        if existing_attempt:
            attempts.append(existing_attempt)

        evaluated = 0
        for subtitle in subtitles:
            if evaluated >= int(self.quality_settings.max_candidates):
                break
            downloaded = self._download_candidate_subtitle(subtitle)
            if downloaded is None:
                continue
            evaluated += 1
            try:
                attempt = self._evaluate_candidate_subtitle(
                    episode=episode,
                    subtitle=subtitle,
                    subtitle_path=downloaded,
                    rank=evaluated,
                )
                attempts.append(attempt)
            except Exception as exc:
                logger.warning(
                    "Failed to evaluate downloaded subtitle candidate for {}: {}",
                    episode.short_str(),
                    exc,
                )
                downloaded.unlink(missing_ok=True)

        selected_attempt = choose_best_attempt(attempts)
        if not selected_attempt:
            console.print(f"[bold red]\tUnable to evaluate subtitle candidates for {episode.short_str()}[/bold red]")
            return

        selected_path = Path(selected_attempt["subtitle_path"])
        if selected_path.resolve() != srt_filepath.resolve():
            shutil.move(selected_path, srt_filepath)
            logger.info("Subtitle saved to {}", srt_filepath)

        for attempt in attempts:
            attempt_path = Path(attempt["subtitle_path"])
            if attempt_path.resolve() == srt_filepath.resolve():
                continue
            if attempt_path.exists():
                attempt_path.unlink(missing_ok=True)

        subtitle_json = selected_attempt.get("subtitle_json")
        if subtitle_json:
            opensubs_filepath = srt_filepath.with_suffix(".opensubtitles")
            with opensubs_filepath.open("w", encoding="utf-8") as json_out:
                json_out.write(subtitle_json)
            logger.info("Subtitle metadata saved to {}", opensubs_filepath)

        self._persist_quality(
            episode_key=episode_key,
            selected_attempt=selected_attempt,
            attempts=attempts,
        )

    def download_srt_file(self, subtitle: Subtitle, path: Path):
        if self.client.user_downloads_remaining <= 0:
            raise OpenSubtitlesException(
                "Download limit reached. " 
                "Please upgrade your OpenSubtitles account "
                "or wait for your quota to reset (~24hrs)"
            )

        download_body = {
            "file_id": subtitle.file_id,
            "sub_format": "srt"
        }
        raw_api_response = self.client.send_api("download", download_body)
        api_response = DownloadResponse(raw_api_response)
        self.client.user_downloads_remaining = api_response.remaining

        download_response = requests.get(api_response.link)
        download_response.raise_for_status()

        with open(path, "wb") as f:
            f.write(download_response.content)

    def write_quality_report(self):
        report_path_value = getattr(self.config.args, "subtitle_quality_report", None)
        if not report_path_value:
            return
        report_path = Path(report_path_value).expanduser()
        if report_path.suffix.lower() != ".json":
            report_path.mkdir(parents=True, exist_ok=True)
            report_path = report_path / f"{self.series.name}.subtitle-quality-report.json"
        else:
            report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "series_name": self.series.name,
            "generated_at": current_timestamp(),
            "subtitle_quality_settings": {
                "enabled": self.quality_settings.enabled,
                "max_candidates": self.quality_settings.max_candidates,
                "runtime_ratio_max": self.quality_settings.runtime_ratio_max,
                "runtime_ratio_min": self.quality_settings.runtime_ratio_min,
                "overlap_containment_threshold": self.quality_settings.overlap_containment_threshold,
            },
            "episodes": self.quality_rows,
        }
        with report_path.open("w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False, indent=2)
        logger.info("Subtitle quality report written to {}", report_path)

    def find_existing_subtitle(self, episode: Episode) -> Optional[Path]:
        """
        Check for existing subtitle files in various naming formats.

        Args:
            episode (Episode): Episode to check for subtitles

        Returns:
            Optional[Path]: Path to an existing subtitle file if found, None otherwise
        """
        patterns = self.generate_subtitle_patterns(episode)

        for pattern in patterns:
            filepath = self.series.subtitles_dir / pattern
            if filepath.exists():
                return filepath

        return None

    def generate_subtitle_patterns(self, episode: Episode) -> list[str]:
        """
        Generate various common subtitle filename patterns.

        Args:
            episode (Episode): Episode

        Returns:
            List[str]: List of possible subtitle filenames
        """
        season_num = episode.season_number
        episode_num = episode.episode_number
        patterns = [
            # Standard format: "Show Name - S01E02.srt"
            f"{self.series.name} - S{season_num:02d}E{episode_num:02d}.srt",
            # Season x Episode format: "Show Name - 1x02.srt"
            f"{self.series.name} - {season_num}x{episode_num:02d}.srt",
            # Separate season/episode: "Show Name - Season 1 Episode 02.srt"
            f"{self.series.name} - Season {season_num} Episode {episode_num:02d}.srt",
            # Compact format: "ShowName.S01E02.srt"
            f"{self.series.name.replace(' ', '')}.S{season_num:02d}E{episode_num:02d}.srt",
            # Numbered format: "Show Name 102.srt"
            f"{self.series.name} {season_num:01d}{episode_num:02d}.srt",
            # Dot format: "Show.Name.1x02.srt"
            f"{self.series.name.replace(' ', '.')}.{season_num}x{episode_num:02d}.srt",
            # Underscore format: "Show_Name_S01E02.srt"
            f"{self.series.name.replace(' ', '_')}_S{season_num:02d}E{episode_num:02d}.srt",
        ]

        return patterns

    def _download_candidate_subtitle(self, subtitle: Subtitle) -> Path | None:
        try:
            downloaded = self.client.download_and_save(subtitle)
        except Exception as exc:
            logger.warning("Failed to download subtitle candidate id={} error={}", subtitle.subtitle_id, exc)
            return None
        return Path(downloaded)

    def _episode_runtime_minutes(self, episode_key: EpisodeKey) -> float | None:
        try:
            details = self.series.get_episode_detail(episode_key, keys=["runtime"])
        except Exception:
            return None
        runtime = details.get("runtime")
        if runtime in (None, "", 0):
            return None
        try:
            return float(runtime)
        except (TypeError, ValueError):
            return None

    def _neighbor_context(self, episode_key: EpisodeKey):
        mapping = subtitle_paths_by_episode(
            self.series.subtitles_dir,
            include_quarantined=True,
        )
        neighbors: list[EpisodeKey] = []
        if episode_key.episode_number > 1:
            neighbors.append(EpisodeKey(episode_key.season_number, episode_key.episode_number - 1))
        neighbors.append(EpisodeKey(episode_key.season_number, episode_key.episode_number + 1))

        neighbor_metrics = {}
        neighbor_lines = {}
        for neighbor in neighbors:
            neighbor_path = mapping.get(neighbor)
            if not neighbor_path:
                continue
            expected_runtime = self._episode_runtime_minutes(neighbor)
            metrics, lines = evaluate_subtitle_signals(
                neighbor_path,
                expected_runtime_minutes=expected_runtime,
                settings=self.quality_settings,
            )
            neighbor_metrics[neighbor] = metrics
            neighbor_lines[neighbor] = lines
        return neighbor_metrics, neighbor_lines

    def _metadata_episode_mismatch(self, episode: Episode, subtitle: Subtitle | None) -> bool:
        if subtitle is None:
            return False
        season = getattr(subtitle, "season_number", None)
        ep_no = getattr(subtitle, "episode_number", None)
        if season is None or ep_no is None:
            return False
        try:
            return int(season) != int(episode.season_number) or int(ep_no) != int(episode.episode_number)
        except (TypeError, ValueError):
            return False

    def _evaluate_existing_subtitle(self, episode: Episode, subtitle_path: Path) -> dict:
        return self._evaluate_candidate_subtitle(
            episode=episode,
            subtitle=None,
            subtitle_path=subtitle_path,
            rank=0,
            source="existing",
        )

    def _evaluate_candidate_subtitle(
        self,
        *,
        episode: Episode,
        subtitle: Subtitle | None,
        subtitle_path: Path,
        rank: int,
        source: str = "api",
    ) -> dict:
        episode_key = episode.key()
        expected_runtime = self._episode_runtime_minutes(episode_key)
        candidate_metrics, candidate_lines = evaluate_subtitle_signals(
            subtitle_path,
            expected_runtime_minutes=expected_runtime,
            settings=self.quality_settings,
        )
        neighbor_metrics, neighbor_lines = self._neighbor_context(episode_key)
        diagnostics = candidate_quality_diagnostics(
            episode_key=episode_key,
            candidate_metrics=candidate_metrics,
            candidate_lines=candidate_lines,
            neighbor_metrics_by_episode=neighbor_metrics,
            neighbor_lines_by_episode=neighbor_lines,
            settings=self.quality_settings,
            metadata_episode_mismatch=self._metadata_episode_mismatch(episode, subtitle),
        )
        subtitle_json = subtitle.to_json() if subtitle is not None else None
        subtitle_info = {
            "subtitle_id": getattr(subtitle, "subtitle_id", None),
            "download_count": getattr(subtitle, "download_count", None),
            "new_download_count": getattr(subtitle, "new_download_count", None),
            "votes": getattr(subtitle, "votes", None),
            "ratings": getattr(subtitle, "ratings", None),
            "from_trusted": getattr(subtitle, "from_trusted", None),
            "season_number": getattr(subtitle, "season_number", None),
            "episode_number": getattr(subtitle, "episode_number", None),
            "release": getattr(subtitle, "release", None),
            "file_name": getattr(subtitle, "file_name", None),
        }
        return {
            "rank": int(rank),
            "source": source,
            "subtitle_path": str(subtitle_path),
            "subtitle_json": subtitle_json,
            "subtitle": subtitle_info,
            "quality": diagnostics,
        }

    def _persist_quality(
        self,
        *,
        episode_key: EpisodeKey,
        selected_attempt: dict,
        attempts: list[dict],
    ):
        verdict = str(selected_attempt.get("selection_verdict", "pass"))
        payload = {
            "episode": str(episode_key),
            "verdict": verdict,
            "selected_subtitle_path": selected_attempt.get("subtitle_path"),
            "selected_subtitle": selected_attempt.get("subtitle", {}),
            "selected_candidate_score": selected_attempt.get("candidate_score"),
            "selected_metadata_penalty": selected_attempt.get("metadata_penalty"),
            "selected_hard_fail_reasons": (
                selected_attempt.get("quality", {}).get("hard_fail_reasons", [])
            ),
            "selected_metrics": selected_attempt.get("quality", {}).get("metrics", {}),
            "attempts": [
                {
                    "rank": attempt.get("rank"),
                    "source": attempt.get("source"),
                    "subtitle_path": attempt.get("subtitle_path"),
                    "subtitle": attempt.get("subtitle", {}),
                    "candidate_score": attempt.get("candidate_score"),
                    "metadata_penalty": attempt.get("metadata_penalty"),
                    "quality": attempt.get("quality", {}),
                }
                for attempt in attempts
            ],
            "settings": {
                "enabled": self.quality_settings.enabled,
                "max_candidates": self.quality_settings.max_candidates,
                "runtime_ratio_max": self.quality_settings.runtime_ratio_max,
                "runtime_ratio_min": self.quality_settings.runtime_ratio_min,
                "overlap_containment_threshold": self.quality_settings.overlap_containment_threshold,
            },
            "evaluated_at": current_timestamp(),
        }
        save_episode_quality(self.series.subtitles_dir, episode_key, payload)
        if verdict == "quarantined":
            append_quarantine_event(
                self.series.subtitles_dir,
                {
                    "episode": str(episode_key),
                    "verdict": verdict,
                    "selected_subtitle_path": selected_attempt.get("subtitle_path"),
                    "selected_hard_fail_reasons": (
                        selected_attempt.get("quality", {}).get("hard_fail_reasons", [])
                    ),
                    "timestamp": current_timestamp(),
                },
            )
        self.quality_rows.append(
            {
                "episode": str(episode_key),
                "verdict": verdict,
                "selected_subtitle_path": selected_attempt.get("subtitle_path"),
                "selected_candidate_score": selected_attempt.get("candidate_score"),
                "hard_fail_reasons": (
                    selected_attempt.get("quality", {}).get("hard_fail_reasons", [])
                ),
                "attempt_count": len(attempts),
            }
        )

    def print_subtitles_table(self, episode: Episode, subtitles: list[opensubtitles.Subtitle]):
        table = Table(title=f"{self.series.name} - {episode.short_str()} Available subtitles")
        table.add_column("Id", justify="right", style="cyan", no_wrap=True)
        table.add_column("Total Downloads", justify="right", style="cyan", no_wrap=True)
        table.add_column("Votes", justify="right", style="cyan", no_wrap=True)
        table.add_column("Ratings", justify="right", style="cyan", no_wrap=True)
        table.add_column("Trusted", justify="center", style="cyan", no_wrap=True)
        table.add_column("Release", justify="left", style="green", no_wrap=False)

        for subtitle in subtitles:
            table.add_row(str(subtitle.subtitle_id),
                          str(subtitle.download_count + subtitle.new_download_count),
                          str(subtitle.votes),
                          str(subtitle.ratings),
                          ":white_check_mark:" if subtitle.from_trusted else None,
                          subtitle.release)

        console.print(table)
