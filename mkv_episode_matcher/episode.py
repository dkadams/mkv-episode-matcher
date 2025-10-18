from dataclasses import dataclass

from rich.console import Console

from mkv_episode_matcher.series import Series

console = Console()

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


def get_specs(config):
    # Normalize filtering to a list of episode specifiers
    if config.args.episodes_specifiers:
        return config.args.episodes_specifiers
    elif config.args.season_numbers:
        return [(season_number, None)
                for season_number in config.args.season_numbers]
    else:
        return None

def get_seasons_by_number(series: Series):
    season_detail = [season for key, season in series.detail.items()
                     if key.startswith("season/")]

    result = {}
    for season in season_detail:
        season_number = season["season_number"]
        episodes = _get_episodes(season)
        result[season_number] = Season(season_number, episodes)
    return result

def _get_episodes(season_detail):
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
        return episode_str(self.season_number, self.episode_number)

def episode_str(season_number, episode_number):
    return f"S{season_number:02d}E{episode_number:02d}"

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
