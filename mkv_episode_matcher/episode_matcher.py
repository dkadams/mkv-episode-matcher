import itertools
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import episode_str
from mkv_episode_matcher.indexed_episode_matcher import IndexedEpisodeMatcher, \
    MatchResult
from mkv_episode_matcher.series import get_series, Series

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
        results = IndexedEpisodeMatcher(config, series).match(paths)
        display_results(series, results)


def display_results(series: Series, results: List[MatchResult]):
    table = Table(title=f"Matches for '{series.name}'")
    table.add_column("Filename")
    table.add_column("Episode Id")
    table.add_column("Correct", justify="center")
    table.add_column("# Matches", style="bold")
    table.add_column("#1", style="magenta")
    table.add_column("#2")
    table.add_column("#3")
    table.add_column("#4")
    table.add_column("#5")

    correct = 0
    known_episode_count = 0
    for result in results:
        if result.known_episode:
            known_episode_count += 1
            actual = episode_str(*result.known_episode)
            correct_match = (len(result.matches) > 0
                             and result.known_episode == result.matches[0][1:])
            if correct_match:
                correct += 1
        else:
            correct_match = False
            actual = "-"

        def prefix(match):
            if match[1:] == result.known_episode:
                return "[bold green]"
            else:
                return ""

        def format(match):
            fmt_distance = "-"
            match match[0]:
                case float() as distance:
                    fmt_distance = f"{distance:.5f}"
                case (min_distance, count):
                    fmt_distance = f"min d: {min_distance:.5f} #: {count}"

            return f"{episode_str(match[1], match[2])}\n " + fmt_distance

        formatted_matches = [prefix(m) + format(m) for m in result.matches]
        if len(formatted_matches) < 5:
            formatted_matches.extend(["-"] * (5 - len(formatted_matches)))

        table.add_row(result.file.name,
                      actual,
                      "[bold green]*" if correct_match else "",
                      str(len(result.matches)),
                      formatted_matches[0],
                      formatted_matches[1],
                      formatted_matches[2],
                      formatted_matches[3],
                      formatted_matches[4])

    console.print(table)
    if known_episode_count > 0:
        console.print(f"Correct: {correct}/{known_episode_count} "
                      f"({correct/known_episode_count*100:.2f}%)")
