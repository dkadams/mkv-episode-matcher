import shutil
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
from mkv_episode_matcher.episode import Episode
from mkv_episode_matcher.series import SeriesDirectoryProcessor, Series, \
    get_specified_episodes

console = Console()

def download_subtitles(config):
    def series_downloader(series):
        console.print(f"[bold green]Downloading subtitles for: {series.name}")

        episodes = get_specified_episodes(config, series)

        downloader = OpenSubtitlesDownloader(config, series)
        for episode in episodes:
            downloader.download(episode)

        console.print(f"[bold green]Subtitles downloaded")

    SeriesDirectoryProcessor(config).process_series(series_downloader)

class OpenSubtitlesDownloader:
    def __init__(self, config: Configuration, series: Series):
        if not config.has_required_settings():
            console.print("[bold red]Error: missing configuration settings. Run mkv-episode-matcher config")
            return

        self.config = config
        self.series = series
        
        api_config = config.stored["api"]
        open_subtitles_api_key = api_config.get("open_subtitles_api_key")
        open_subtitles_user_agent = api_config.get("open_subtitles_user_agent")
        open_subtitles_username = api_config.get("open_subtitles_username")
        open_subtitles_password = api_config.get("open_subtitles_password")

        self.client = OpenSubtitles(open_subtitles_user_agent, open_subtitles_api_key)
        self.client.login(open_subtitles_username, open_subtitles_password)

        self.subtitle_dir = self.series.dot_dir / "subtitles"

    def download(self, episode: Episode):
        console.print(f"Preparing to download series: {self.series.name} - {episode.short_str()}...")

        existing_subtitle = self.find_existing_subtitle(episode)
        if existing_subtitle and not self.config.args.refresh:
            console.print(f"[bold green]\tsubtitle exists: {existing_subtitle}. Skipping download.[/bold green]")
            return

        response = self.client.search(parent_tmdb_id=self.series.detail["id"],
                                      tmdb_id=episode.tmdb_id, languages="en")
        if len(response.data) == 0:
            console.print(f"No subtitles found for {self.series.name} - {episode.short_str()}")
            return

        console.print(f"[green]Found {len(response.data)} subtitles for {self.series.name} - {episode.short_str()}")
        subtitles = sorted(response.data,
                           key=lambda subtitle: subtitle.download_count
                                                + subtitle.new_download_count, reverse=True)
        self.print_subtitles_table(episode, subtitles)

        # TODO configurable selection for selected_subtitle. Most downloaded is
        # what's selected here.
        selected_subtitle = subtitles[0]

        srt_filename = f"{self.series.name} - {episode.short_str()}.srt"
        srt_filepath = self.subtitle_dir / srt_filename

        srt_file = self.client.download_and_save(selected_subtitle)
        if not self.subtitle_dir.exists():
            self.subtitle_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(srt_file, srt_filepath)
        logger.info(f"Subtitle saved to {srt_filepath}")

        opensubs_filepath = srt_filepath.with_suffix(".opensubtitles")
        opensubs_json = selected_subtitle.to_json()
        with open(opensubs_filepath, "w") as json_out:
            json_out.write(opensubs_json)
        logger.info(f"Subtitle metadata saved to {opensubs_filepath}")

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
            filepath = self.subtitle_dir / pattern
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

