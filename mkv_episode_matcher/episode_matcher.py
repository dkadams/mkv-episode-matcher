import datetime
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import timedelta, datetime
from pathlib import Path
from typing import Protocol, Callable, Any, Self

from rich.console import Console
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import IndexedEpisodeMatcher, \
    IntervalMatch, Video
from mkv_episode_matcher.series import get_series, Series

console = Console()

class Score(Protocol):
    def __str__(self) -> str: ...
    def __lt__(self, other) -> bool: ...

@dataclass(frozen=True, eq=True, order=True)
class Distance(Score):
    distance: float

@dataclass(eq=True)
class CountAndDistance(Score):
    count: int
    min_distance: float

    def merge(self, match: IntervalMatch):
        self.count += 1
        self.min_distance = min(self.min_distance, match.distance)

    def key(self) -> tuple[int, float]:
        """
        DESCENDING matches, ASCENDING distance.
        More matches, less distance = better match.
        """
        return -self.count, self.min_distance

    def __lt__(self, other: Self) -> bool:
        return self.key() < other.key()

    def __str__(self):
        return f"#: {self.count} min(d): {self.min_distance:.5f}"

@dataclass(frozen=True, eq=True, order=True)
class AggregateMatch:
    video: Video
    episode: EpisodeKey
    score: Score
    interval_matches: list[IntervalMatch]

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
        matches = IndexedEpisodeMatcher(config, series).match(paths)
        dump_results(matches, series)

        aggregated = aggregate_matches(matches)
        if config.args.display_by_episode:
            episode_matches = group_matches(config, aggregated,
                                            lambda match: match.episode)
            display_results_by_episode(episode_matches)

        if config.args.display_by_file:
            video_matches = group_matches(config, aggregated,
                                          lambda match: match.video)
            display_results_by_video(series, video_matches)

def dump_results(results: list[IntervalMatch], series: Series):
    match_dir = series.ensure_matches_dir()
    ts = (datetime.now().astimezone().isoformat(timespec="milliseconds")
          .replace(":", ""))
    result_file = match_dir / f"{ts}.jsonl"
    with open(result_file, "w", encoding="utf-8") as out:
        for result in results:
            out.write(json.dumps(asdict(result), ensure_ascii=False,
                                 # Convert non-JSON-friendly objects (e.g., Path) to strings
                                 default=str,
                                 # Remove spaces from separators to compact output
                                 separators=(",", ":")))
            out.write("\n")

def aggregate_matches(matches: list[IntervalMatch]) -> list[AggregateMatch]:
    aggregated = {}
    for match in matches:
        grouping = (match.video, match.episode)
        grouped = aggregated.setdefault(grouping, (CountAndDistance(0, 1000000),
                                                   []))
        grouped[0].merge(match)
        grouped[1].append(match)

    return [AggregateMatch(video, episode, score, interval_matches)
            for (video, episode), (score, interval_matches) in aggregated.items()]

def group_matches(config: Configuration, matches: list[AggregateMatch],
    keyfunc: Callable[[AggregateMatch], Any]) -> dict[Video|EpisodeKey, list[AggregateMatch]]:
    results = {}
    for match in matches:
        grouped = results.setdefault(keyfunc(match), [])
        grouped.append(match)

    for aggregated in results.values():
        aggregated.sort(key=lambda match: match.score)
        # argparse stores this option as num_matches (--num-matches / -n)
        aggregated[:] = aggregated[:config.args.num_matches]

    return results

def display_results_by_episode(matches_by_episode: dict[EpisodeKey, list[AggregateMatch]]):
    for episode, matches in matches_by_episode.items():
        if len(matches_by_episode) > 0:
            match_table = Table(title=f"{str(episode)}")
            match_table.add_column("File")
            match_table.add_column("Score")
            match_table.add_column("File Duration")
            for match in matches:
                duration = match.video.video_info.minutes
                match_table.add_row(str(match.video.file),
                                    str(match.score),
                                    str(timedelta(seconds=duration))
                                    )
            console.print(match_table)


def display_results_by_video(series: Series,
    matches_by_video: dict[Video, list[AggregateMatch]]):
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
    for video, agg_match in matches_by_video:
        known_episode = video.video_info.known_episode
        if known_episode:
            known_episode_count += 1
            actual = str(known_episode)
            correct_match = (len(agg_match.matches) > 0
                             and known_episode == agg_match.episode)
            if correct_match:
                correct += 1
        else:
            correct_match = None
            actual = "-"

        def prefix(match: IntervalMatch):
            if match.episode == known_episode:
                return "[bold green]"
            else:
                return ""

        formatted_matches = [prefix(m) + str(m.episode) + "\n" + str(m.score)
                             for m in agg_match.matches]

        if len(formatted_matches) < 5:
            formatted_matches.extend(["-"] * (5 - len(formatted_matches)))

        correct_marker = ""
        match correct_match:
            case True:
                correct_marker = "[bold green]*"
            case False:
                correct_marker = "[bold red]X"

        table.add_row(str(agg_match.file),
                      actual,
                      correct_marker,
                      str(len(agg_match.matches)),
                      formatted_matches[0],
                      formatted_matches[1],
                      formatted_matches[2],
                      formatted_matches[3],
                      formatted_matches[4])

    console.print(table)
    if known_episode_count > 0:
        console.print(f"Correct: {correct}/{known_episode_count} "
                      f"({correct/known_episode_count*100:.2f}%)")
