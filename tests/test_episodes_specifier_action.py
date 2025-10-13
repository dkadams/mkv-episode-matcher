import argparse
import pytest

from mkv_episode_matcher.episodes_specifier import EpisodesSpecifierAction


def make_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--episodes",
        dest="episodes_specifiers",
        action=EpisodesSpecifierAction,
        nargs="*",
        metavar="SEASON:SPEC",
    )
    return parser


class TestEpisodesSpecifierAction:
    def test_single_integers(self):
        parser = make_parser()
        ns = parser.parse_args(["--episodes", "5:1", "0:0"])  # type: ignore[arg-type]
        assert ns.episodes_specifiers == [(5, 1), (0, 0)]

    def test_comma_separated_list(self):
        parser = make_parser()
        ns = parser.parse_args(["--episodes", "2:1,2,3", "4:0,5"])  # type: ignore[arg-type]
        assert ns.episodes_specifiers == [(2, [1, 2, 3]), (4, [0, 5])]

    def test_ranges(self):
        parser = make_parser()
        ns = parser.parse_args(["--episodes", "4:0-4", "2:-4", "5:0-", "2:1-2"])  # type: ignore[arg-type]
        assert ns.episodes_specifiers == [
            (4, (0, 4)),
            (2, (None, 4)),
            (5, (0, None)),
            (2, (1, 2)),
        ]

    def test_star_all(self):
        parser = make_parser()
        ns = parser.parse_args(["--episodes", "0:*"])  # type: ignore[arg-type]
        assert ns.episodes_specifiers == [(0, None)]

    @pytest.mark.parametrize(
        "args",
        [
            ["--episodes", "x:1"],  # non-integer season
            ["--episodes", "1:x"],  # invalid spec
            ["--episodes", "1:,"] , # empty list
            ["--episodes", "1:3--4"],  # bad range
            ["--episodes", "1:-"],  # just a hyphen
            ["--episodes", "1:"],  # missing spec
            ["--episodes", "1"],  # missing colon
        ],
    )
    def test_errors(self, args):
        parser = make_parser()
        with pytest.raises(SystemExit) as se:
            parser.parse_args(args)  # type: ignore[arg-type]
        assert se.value.code == 2
