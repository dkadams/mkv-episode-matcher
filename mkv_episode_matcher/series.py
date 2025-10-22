import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from loguru import logger
from rich.console import Console

from mkv_episode_matcher.config import Configuration

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

        index_dir = settings.get("index-dir") or series_dot_dir / "indexes.chromadb"
        return Series(series_dir, series_dot_dir, series_detail, series_name,
                      index_dir)

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
