import itertools
from pathlib import Path

from rich.console import Console

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.indexed_episode_matcher import IndexedEpisodeMatcher
from mkv_episode_matcher.series import Series

console = Console()

def match_episodes(config: Configuration):
    last_matcher = None
    for path_str in config.args.video_files:
        path = Path(path_str)
        if not (series := get_series(path)):
            continue

        if not last_matcher or last_matcher.series != series:
            last_matcher = IndexedEpisodeMatcher(config, series)
        last_matcher.match(path)

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
