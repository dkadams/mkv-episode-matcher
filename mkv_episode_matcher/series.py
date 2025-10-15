import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from rich.console import Console

from mkv_episode_matcher.config import Configuration

console = Console()

class SeriesDirectoryProcessor:
    def __init__(self, config: Configuration):
        self.config = config
        self.series_dirs = [Path(dir).resolve()
                            for dir in self.config.args.series_dirs]

    def process_series(self, process_func):
        for series_dir in self.series_dirs:
            logger.info(f"Processing {series_dir}")

            series_dot_dir = series_dir / ".mkv-episode-matcher"
            series_file = series_dot_dir / "series.tmdb.json"
            if not series_file.exists():
                console.print(f"[bold red]Error:[/bold red] Series data not initialized for {series_dir}. run `mkv-episode-matcher init {series_dir}` first.")
                continue

            logger.info(f"Loading series data from {series_file}")
            with open(series_file, 'r') as file:
                series_detail = json.load(file)

            series_name = series_detail["name"]
            logger.info(f"Processing series: {series_name}")

            series = Series(series_dir, series_dot_dir,
                            series_detail, series_name)
            process_func(series)

@dataclass
class Series:
    dir: Path
    dot_dir: Path

    detail: dict
    name: str
