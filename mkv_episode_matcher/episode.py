import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, NamedTuple, Self

from guessit import guessit
from rich.console import Console

console = Console()

class EpisodeKey(tuple[int, int]):
    season_number: int
    episode_number: int

    def __new__(cls, season_number, episode_number):
        return super().__new__(cls, (season_number, episode_number))

    @property
    def season_number(self):
        return self[0]

    @property
    def episode_number(self):
        return self[1]

    def __str__(self):
        return _episode_str(self.season_number, self.episode_number)

    @staticmethod
    def from_path(file: Path) -> Optional["EpisodeKey"]:
        def from_opensubs() -> Optional["EpisodeKey"]:
            """
            subtitle_downloader.py saves off a JSON file with API response data
            when it downloads a subtitle from OpenSubtitles. The subtitle filename
            contains the season and episode number we requested, but the API data
            seems more authoritative.
            """
            opensubs_file = file.with_suffix(".opensubtitles")
            if not opensubs_file.exists():
                return None

            with open(opensubs_file, "r") as json_in:
                opensubs_data = json.load(json_in)

            season_number = int(opensubs_data["season_number"])
            episode_number = int(opensubs_data["episode_number"])
            if season_number and episode_number:
                return EpisodeKey(season_number, episode_number)
            else:
                return None

        def from_guessit() -> Optional["EpisodeKey"]:
            matches = guessit(file.name)
            if not matches:
                return None

            season_number = matches.get("season")
            episode_number = matches.get("episode")
            if season_number and episode_number:
                return EpisodeKey(season_number, episode_number)
            else:
                return None

        return from_opensubs() or from_guessit()

    @staticmethod
    def from_str(episode: str) -> "EpisodeKey":
        parts = episode.split('S')[1].split('E')

        season_number = int(parts[0])
        episode_number = int(parts[1])

        return EpisodeKey(season_number, episode_number)

def get_specs(config):
    # Normalize filtering to a list of episode specifiers
    if config.args.episodes_specifiers:
        return config.args.episodes_specifiers
    elif config.args.season_numbers:
        return [(season_number, None)
                for season_number in config.args.season_numbers]
    else:
        return None

def _get_episodes(season_detail):
    result = {}
    for episode_detail in season_detail["episodes"]:
        episode_number = episode_detail["episode_number"]
        episode_id = episode_detail["id"]
        result[episode_number] = Episode(season_detail["season_number"],
                                         episode_number, episode_id)
    return result

@dataclass(eq=True, frozen=True, order=True)
class Episode:
    season_number: int
    episode_number: int
    tmdb_id: int

    def short_str(self):
        return _episode_str(self.season_number, self.episode_number)

    def key(self) -> EpisodeKey:
        return EpisodeKey(self.season_number, self.episode_number)

def _episode_str(season_number: int | str,
                episode_number: int | str | Tuple[int, int] | List[int]) -> str:
    if isinstance(episode_number, tuple) or isinstance(episode_number, list):
        separator = "-" if len(episode_number) == 2 else ","
        ep_str = separator.join(f"{int(e):02d}" for e in episode_number)
    else:
        ep_str = f"{int(episode_number):02d}"
    return f"S{int(season_number):02d}E{ep_str}"

@dataclass(eq=True, frozen=True, order=True)
class Season:
    season_number: int
    episodes: dict[int, Episode] = field(compare=False)

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
