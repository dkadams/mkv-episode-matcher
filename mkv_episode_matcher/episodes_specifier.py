import argparse
import re
from typing import List, Optional, Tuple, Union

# Public types for consumers/tests
EpisodeSpecifier = Union[int, List[int], Tuple[Optional[int], Optional[int]], None]
EpisodeTuple = Tuple[int, EpisodeSpecifier]


def _parse_positive_int(text: str) -> int:
    if not text.isdigit():
        raise ValueError(f"expected a non-negative integer, got '{text}'")
    value = int(text)
    if value < 0:
        # This branch is actually unreachable due to isdigit(), but kept for clarity
        raise ValueError(f"expected a non-negative integer, got '{text}'")
    return value


def _parse_episode_specifier(spec: str) -> EpisodeSpecifier:
    spec = spec.strip()
    if spec == "*":
        return None

    # comma-separated list of positive integers
    if "," in spec:
        parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
        if not parts:
            raise ValueError("empty comma-separated list is not allowed")
        return [_parse_positive_int(p) for p in parts]

    # ranges
    if "-" in spec:
        left, right = spec.split("-")

        if left != "" and right != "": # bounded range: A-B
            a = _parse_positive_int(left)
            b = _parse_positive_int(right)
            return a, b
        elif left == "": # unbounded to the left: -N
            return None, _parse_positive_int(right)
        elif right == "": # unbounded to the right: N-
            return _parse_positive_int(left), None
        else:
            raise ValueError(f"invalid range format '{spec}'")

    # single positive integer
    return _parse_positive_int(spec)


class EpisodesSpecifierAction(argparse.Action):
    """Custom argparse action to parse --episodes arguments.

    Each value must be of the form {season}:{episode_specifier} where season is an integer
    and episode_specifier is one of:
      - positive integer -> int
      - comma separated list of integers -> list[int]
      - bounded/unbounded integer range (A-B, -B, A-) -> tuple[int|None, int|None]
      - * -> None

    The parsed result stored in dest is a list of tuples: List[Tuple[int, EpisodeSpecifier]]
    """

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        # values may be a single string or a list of strings depending on nargs
        if isinstance(values, str):
            values_list = [values]
        else:
            values_list = list(values)

        result: List[EpisodeTuple] = []
        for raw in values_list:
            try:
                season_part, spec_part = raw.split(":", 1)
            except ValueError:
                raise argparse.ArgumentError(self, f"invalid episode value '{raw}': expected season:specifier")

            season = self.get_season(season_part)
            spec_value = self.get_spec_part(spec_part)

            result.append((season, spec_value))

        setattr(namespace, self.dest, result)

    def get_season(self, season_part: str) -> int:
        season_part = season_part.strip()
        try:
            season = int(season_part)
        except ValueError:
            raise argparse.ArgumentError(self,
                                         f"invalid season '{season_part}': must be an integer")
        if season < 0:
            raise argparse.ArgumentError(self,
                                         f"invalid season '{season_part}': must be a non-negative integer")
        return season

    def get_spec_part(self, spec_part: str) -> int | list[int] | tuple[
        int | None, int | None] | None:
        spec_part = spec_part.strip()
        try:
            spec_value = _parse_episode_specifier(spec_part)
        except ValueError as e:
            raise argparse.ArgumentError(self,
                                         f"invalid episode specifier '{spec_part}': {e}")
        return spec_value
