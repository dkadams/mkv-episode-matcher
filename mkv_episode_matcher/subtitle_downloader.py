import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from opensubtitlescom import OpenSubtitles, opensubtitles
from rich.console import Console
from rich.table import Table

console = Console()

def download_subtitles(config):
    series_dirs = [Path(dir).resolve() for dir in config.args.series_dirs]
    for series_dir in series_dirs:
        console.print(f"[bold green]Downloaded Subtitles Series: {series_dir}[/bold green]")

        series_dot_dir = series_dir / ".mkv-episode-matcher"
        series_file = series_dot_dir / "series.tmdb.json"
        if not series_file.exists():
            console.print(f"[bold red]Error:[/bold red] Series data not initialized for {series_dir}. run `mkv-episode-matcher init {series_dir}` first.")

        with open(series_file, 'r') as file:
            series_detail = json.load(file)

        seasons_by_number = get_seasons_by_number(series_detail)
        episodes = get_episodes_to_download(config, series_detail["name"], seasons_by_number)

        downloader = OpenSubtitlesDownloader(config, series_dir, series_detail["name"])
        for episode in episodes:
            downloader.download(episode)

        console.print(f"[bold green]Subtitles downloaded")

def get_episodes_to_download(config, series_name:str, seasons_by_number:dict[int, "Season"]) -> set["Episode"]:
    result = set()
    specs = get_specs(config)
    for spec in specs:
        season_number, episode_spec = spec
        season = seasons_by_number.get(season_number)
        if not season:
            console.print(f"[bold red]Error: Season: {season_number} "
                          f"not found for {config.args.series_name} "
                          f"(tried to match {spec}).")
            continue

        try:
            episodes = season.episodes_matching(episode_spec)
            if episodes:
                result.update(episodes)
            else:
                console.print(f"[orange1]Warn: No episodes matching {spec} "
                              f"found for {series_name} "
                              f"season: {season_number}.")
        except UnknownEpisodeError as e:
            console.print(f"[orange1]Error: Episode {e.episode_number} "
                          f"for specifier {spec} "
                          f"does not exist for {series_name} "
                          f"season: {season_number}.")

    return result


def get_specs(config):
    # Normalize filtering to a list of episode specifiers
    return (config.args.episodes_specifiers
            or [(season_number, None) for season_number in config.args.season_numbers])

def get_seasons_by_number(series_detail):
    season_detail = [season for key, season in series_detail.items()
                     if key.startswith("season/")]

    result = {}
    for season in season_detail:
        season_number = season["season_number"]
        episodes = get_episodes(season)
        result[season_number] = Season(season_number, episodes)
    return result

def get_episodes(season_detail):
    result = {}
    for episode_detail in season_detail["episodes"]:
        episode_number = episode_detail["episode_number"]
        episode_id = episode_detail["id"]
        result[episode_number] = Episode(season_detail["season_number"],
                                         episode_number, episode_id)
    return result

@dataclass(eq=True, frozen=True)
class Episode:
    season_number: int
    episode_number: int
    tmdb_id: int

    def short_str(self):
        return f"S{self.season_number:02d}E{self.episode_number:02d}"

@dataclass(eq=True, frozen=True)
class Season:
    season_number: int
    episodes: dict[int, Episode]

    def episodes_matching(self, episode_spec):
        if episode_spec is None:
            return self.episodes.values()
        elif isinstance(episode_spec, int):
            return self.get_episodes([episode_spec])
        elif isinstance(episode_spec, list):
            return self.get_episodes(episode_spec)
        elif isinstance(episode_spec, tuple):
            start, end = episode_spec
            return self.get_episodes_in_range(start, end)
        else:
            raise ValueError(f"Invalid episode specifier: {episode_spec}")

    def get_episodes(self, episode_numbers: list[int]):
        episodes = []
        for episode_number in episode_numbers:
            episode = self.episodes[int(episode_number)]
            if episode:
                episodes.append(episode)
            else:
                console.print(f"[red]Episode not found: {episode_number}")
                raise UnknownEpisodeError(self, episode_number)
        return episodes

    def get_episodes_in_range(self, start, end) -> list[Episode]:
        if start is None:
            return [episode for episode in self.episodes.values()
                    if episode.episode_number <= end]
        elif end is None:
            return [episode for episode in self.episodes.values()
                    if episode.episode_number >= start]
        else:
            episode_number_range = range(start, end + 1)
            return [episode for episode in self.episodes.values()
                    if episode.episode_number in episode_number_range]

class UnknownEpisodeError(Exception):
    def __init__(self, season, episode_number):
        self.message = f"Unknown episode: {season}:{episode_number}"
        self.season = season
        self.episode_number = episode_number

    def __str__(self):
        return self.message


class OpenSubtitlesDownloader:
    def __init__(self, config, series_dir, series_name):
        if not config.has_required_settings():
            console.print("[bold red]Error: missing configuration settings. Run mkv-episode-matcher config")
            return

        api_config = config.stored["api"]
        open_subtitles_api_key = api_config.get("open_subtitles_api_key")
        open_subtitles_user_agent = api_config.get("open_subtitles_user_agent")
        open_subtitles_username = api_config.get("open_subtitles_username")
        open_subtitles_password = api_config.get("open_subtitles_password")

        self.client = OpenSubtitles(open_subtitles_user_agent, open_subtitles_api_key)
        self.client.login(open_subtitles_username, open_subtitles_password)

        self.subtitle_dir = series_dir / ".mkv-episode-matcher" / "subtitles"
        self.series_name = series_name

    def download(self, episode: Episode):
        console.print(f"Preparing to download series: {self.series_name} - {episode.short_str()}...")

        existing_subtitle = self.find_existing_subtitle(episode)
        if existing_subtitle:
            console.print(f"[bold green]\tsubtitle exists: {existing_subtitle}. Skipping download.[/bold green]")
            return

        # Default to standard format for new downloads
        srt_filepath = str(
            self.subtitle_dir / f"{self.series_name} - {episode.short_str()}.srt"
        )

        response = self.client.search(tmdb_id=episode.tmdb_id, languages="en")
        if len(response.data) == 0:
            console.print(f"No subtitles found for {self.series_name} - {episode.short_str()}")
            return

        console.print(f"[green]Found {len(response.data)} subtitles for {self.series_name} - {episode.short_str()}")
        # TODO configurable selection for subtitles
        subtitles = sorted(response.data,
                           key=lambda subtitle: subtitle.download_count
                                                + subtitle.new_download_count, reverse=True)
        self.print_subtitles_table(episode, subtitles)

        selected_subtitle = subtitles[0]
        srt_file = self.client.download_and_save(selected_subtitle)
        if not self.subtitle_dir.exists():
            self.subtitle_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(srt_file, srt_filepath)
        logger.info(f"Subtitle saved to {srt_filepath}")

        opensubs_filepath = srt_filepath + ".opensubtitles"
        opensubs_json = selected_subtitle.to_json()
        with open(opensubs_filepath, "w") as json_out:
            json_out.write(opensubs_json)
        logger.info(f"Subtitle metadata saved to {opensubs_filepath}")

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
            f"{self.series_name} - S{season_num:02d}E{episode_num:02d}.srt",
            # Season x Episode format: "Show Name - 1x02.srt"
            f"{self.series_name} - {season_num}x{episode_num:02d}.srt",
            # Separate season/episode: "Show Name - Season 1 Episode 02.srt"
            f"{self.series_name} - Season {season_num} Episode {episode_num:02d}.srt",
            # Compact format: "ShowName.S01E02.srt"
            f"{self.series_name.replace(' ', '')}.S{season_num:02d}E{episode_num:02d}.srt",
            # Numbered format: "Show Name 102.srt"
            f"{self.series_name} {season_num:01d}{episode_num:02d}.srt",
            # Dot format: "Show.Name.1x02.srt"
            f"{self.series_name.replace(' ', '.')}.{season_num}x{episode_num:02d}.srt",
            # Underscore format: "Show_Name_S01E02.srt"
            f"{self.series_name.replace(' ', '_')}_S{season_num:02d}E{episode_num:02d}.srt",
        ]

        return patterns

    def print_subtitles_table(self, episode: Episode, subtitles: list[opensubtitles.Subtitle]):
        table = Table(title=f"{self.series_name} - {episode.short_str()} Available subtitles")
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

