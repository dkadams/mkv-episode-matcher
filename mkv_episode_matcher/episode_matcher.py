import itertools
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import episode_str, EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import IndexedEpisodeMatcher, \
    MatchResult, Match
from mkv_episode_matcher.series import get_series, Series, get_seasons_by_number

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
        if config.args.display_by_episode:
            display_results_by_episode(series, results)
        if config.args.display_by_file:
            display_results_by_file(series, results)


def display_results_by_file(series: Series, results: List[MatchResult]):
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
                             and result.known_episode == result.matches[0].episode)
            if correct_match:
                correct += 1
        else:
            correct_match = None
            actual = "-"

        def prefix(match: Match):
            if match.key == result.known_episode:
                return "[bold green]"
            else:
                return ""

        formatted_matches = [prefix(m) + str(m.episode) + "\n" + str(m.score)
                             for m in result.matches]

        if len(formatted_matches) < 5:
            formatted_matches.extend(["-"] * (5 - len(formatted_matches)))

        correct_marker = ""
        match correct_match:
            case True:
                correct_marker = "[bold green]*"
            case False:
                correct_marker = "[bold red]X"

        table.add_row(str(result.file),
                      actual,
                      correct_marker,
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

def display_results_by_episode(series: Series, results: List[MatchResult]):
    matches_by_episode = get_matches_by_episode(results, series)

    for episode, matches in matches_by_episode.items():
        if len(matches) == 0:
            console.print(f"[bold red]No matches found for {episode_str(*episode)}")
        else:
            match_table = Table(title=f"{str(episode)}")
            match_table.add_column("File")
            match_table.add_column("Score")
            for file, match in matches:
                match_table.add_row(str(file), str(match.score))
            console.print(match_table)

def get_matches_by_episode(results: list[MatchResult],
    series: Series) -> dict[EpisodeKey, list[tuple[Path, Match]]]:
    matches_by_episode = {episode.key(): []
           for season in sorted(get_seasons_by_number(series).values())
           for episode in sorted(season.episodes.values())}

    matches_by_episode = {}
    for result in results:
        for match in result.matches:
            matches = matches_by_episode.setdefault(match.key(), [])
            matches.append((result.file, match))

    # Sort each episode's matches by score
    for matches in matches_by_episode.values():
        matches.sort(key=lambda m: m[1].score)

    # Sort the result dict by episode key
    return {k: matches_by_episode[k]
            for k in sorted(matches_by_episode, key=lambda k: k)}

