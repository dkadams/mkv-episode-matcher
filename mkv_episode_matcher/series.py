import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from loguru import logger
from rich.console import Console

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import episode_str, _get_episodes, Season, \
    Episode, get_specs, UnknownEpisodeError

console = Console()

@dataclass(eq=True, frozen=True, order=True)
class Series:
    dir: Path
    dot_dir: Path

    detail: dict
    name: str

    index_dir: Path = None

    @lru_cache
    @staticmethod
    def from_dir(series_dir: Path):
        logger.info(f"Processing {series_dir}")

        series_dot_dir = series_dir / ".mkv-episode-matcher"
        series_file = series_dot_dir / "series.tmdb.json"
        if not series_file.exists():
            console.print(f"[bold red]Error:[/bold red] "
                          f"Series data not initialized for {series_dir}. "
                          f"run `mkv-episode-matcher init {series_dir}`.")
            return None

        logger.info(f"Loading series data from {series_file}")
        with open(series_file, 'r') as file:
            series_detail = json.load(file)

        series_name = series_detail["name"]
        logger.info(f"Processing series: {series_name}")

        settings_file = series_dot_dir / "settings.json"
        settings = {}
        if settings_file.exists():
            with open(settings_file, 'r') as file:
                settings = json.load(file)

        index_dir_setting = settings.get("index-dir")
        index_dir = Path(index_dir_setting) if index_dir_setting else series_dot_dir / "indexes"
        return Series(series_dir, series_dot_dir, series_detail, series_name,
                      index_dir)

    def get_episode_detail(self, episode: tuple[int, int], keys=None) -> dict[str, str | int]:
        season_number, episode_number = episode
        season_detail = self.detail[f"season/{season_number}"]
        episode_detail = next((ep for ep in season_detail["episodes"]
                               if ep["episode_number"] == episode_number), None)
        if not episode_detail:
            raise ValueError(f"Error: No episode detail found for "
                             f"{self.name} "
                             f"episode: {episode_str(*episode)}")

        return {k: episode_detail[k] for k in keys
                or ["id", "season_number", "episode_number", "runtime"]}

class SeriesDirectoryProcessor:
    def __init__(self, config: Configuration):
        self.config = config
        self.series_dirs = [Path(dir).resolve()
                            for dir in self.config.args.series_dirs]

    def process_series(self, process_func: Callable[[Series], None]):
        for series_dir in self.series_dirs:
            if not (series := Series.from_dir(series_dir)):
                continue
            process_func(series)


def get_series(path):
    # Start searching from the leaf directory and move upwards
    candidates = itertools.chain([path] if path.is_dir() else [],
                                 path.parents)
    series_dir = next((dir for dir in candidates
                       if (dir / ".mkv-episode-matcher").is_dir()), None)
    if not series_dir:
        console.print(f"[orange1]No series (.mkv-episode-matcher) directory "
                      f"found in {path} or its parents.")
        return None

    return Series.from_dir(series_dir)


def get_seasons_by_number(series: Series):
    season_detail = [season for key, season in series.detail.items()
                     if key.startswith("season/")]

    result = {}
    for season in season_detail:
        season_number = season["season_number"]
        episodes = _get_episodes(season)
        result[season_number] = Season(season_number, episodes)
    return result


def get_specified_episodes(config, series:Series) -> set["Episode"]:
    seasons_by_number = get_seasons_by_number(series)

    result = set()
    specs = get_specs(config)
    if specs is None:
        return {episode for season in seasons_by_number.values()
                        for episode in season.episodes.values()}

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
                              f"found for {series.name} "
                              f"season: {season_number}.")
        except UnknownEpisodeError as e:
            console.print(f"[orange1]Error: Episode {e.episode_number} "
                          f"for specifier {spec} "
                          f"does not exist for {series.name} "
                          f"season: {season_number}.")

    return result
