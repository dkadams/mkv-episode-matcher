import itertools
from pathlib import Path

from rich.console import Console

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.indexed_episode_matcher import IndexedEpisodeMatcher
from mkv_episode_matcher.series import get_series

console = Console()

def match_episodes(config: Configuration):
    path_with_series = [(path, get_series(path))
                        for path in map(Path, config.args.video_files)]
    path_with_series.sort(key=lambda x: x[1])
    for series, group in itertools.groupby(path_with_series, key=lambda x: x[1]):
        paths = [path for path, _ in group]
        if not series:
            console.print("[orange1]No series found for paths: f{paths}. "
                          "Initialize series with mkv-episode-matcher init")
            continue

        console.print(f"[bold green]Processing series: {series.name}, paths: {paths}")
        IndexedEpisodeMatcher(config, series).match(paths)


