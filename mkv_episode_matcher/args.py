import argparse

from mkv_episode_matcher import __version__
from mkv_episode_matcher.config import edit_config, CONFIG_FILE
from mkv_episode_matcher.episode_matcher import match_episodes
from mkv_episode_matcher.series_initializer import init_series
from mkv_episode_matcher.episodes_specifier import EpisodesSpecifierAction
from mkv_episode_matcher.subtitle_downloader import download_subtitles
from mkv_episode_matcher.subtitle_index import index_subtitles


def build_args_parser():
    parser = get_root_parser()

    # get parent parsers
    config_parser = get_config_parser()
    series_dir_parser = get_series_dir_parser()
    episode_parser = get_episode_parser()

    subparsers = parser.add_subparsers(required=True)

    add_config_parser(subparsers, config_parser)
    add_init_series(subparsers, config_parser, series_dir_parser)
    add_fetch_subs(subparsers, config_parser, series_dir_parser, episode_parser)
    add_index_subs(subparsers, config_parser, series_dir_parser, episode_parser)
    add_match(subparsers, config_parser)

    # fetch/match/rename
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't rename any files, just show what would happen",
    )

    # match
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Set confidence threshold for episode matching (0.0-1.0)",
    )

    return parser

def get_root_parser() -> argparse.ArgumentParser:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Automatically match and rename your MKV TV episodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Generic
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version number and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser


def add_config_parser(subparsers, config_parser):
    config_parser = subparsers.add_parser("config", aliases=["onboard"],
                                          parents=[config_parser],
                                          help="Interactive configuration")
    config_parser.set_defaults(func=edit_config)


def add_init_series(subparsers, config_parser, series_dir_parser):
    init_show_parser = subparsers.add_parser("init-series",
                                             parents=[config_parser,
                                                      series_dir_parser],
                                             help="Initialize a directory as containing episodes of a series")
    init_show_parser.add_argument("--name", dest="series_name",
                                  help="The name of the series (default: name of the directory)")
    init_show_parser.add_argument("--id", dest="series_id",
                                  help="The TMDB id of the series")
    init_show_parser.add_argument("--refresh", action="store_true",
                                  help="Refresh of the series details")
    init_show_parser.set_defaults(func=init_series)


def add_fetch_subs(subparsers, config_parser, series_dir_parser, episode_parser):
    fetch_subs_parser = subparsers.add_parser("fetch-subs",
                                              parents=[config_parser,
                                                       series_dir_parser,
                                                       episode_parser],
                                              help="Fetch subtitles for a series")
    fetch_subs_parser.set_defaults(func=download_subtitles)

def add_index_subs(subparsers, config_parser, series_dir_parser, episode_parser):
    index_subs_parser = subparsers.add_parser("index-subs",
                                              parents=[config_parser,
                                                       series_dir_parser,
                                                       episode_parser],
                                              help="Index subtitles for a series")
    index_subs_parser.set_defaults(func=index_subtitles)


def add_match(subparsers, config_parser):
    match_parser = subparsers.add_parser("match",
                                         parents=[config_parser],
                                         help="Match episodes of a series")

    match_parser.add_argument('video_files',
                              nargs='+',
                              help="Path to one or more video files to match, "
                                   "or directories to recursively search for video files")
    match_parser.add_argument('--extension', '-e',
                              nargs='+',
                              default=".mkv",
                              help="File extension to match (default: .mkv)")
    match_parser.set_defaults(func=match_episodes)

def get_series_dir_parser() -> argparse.ArgumentParser:
    series_dir_parser = argparse.ArgumentParser(add_help=False)
    series_dir_parser.add_argument('series_dirs',
                                   nargs='+',
                                   help="Path to the root directory of one or more series")
    return series_dir_parser


def get_config_parser() -> argparse.ArgumentParser:
    config_parser = argparse.ArgumentParser(add_help=False)

    config_parser.add_argument("--config", "-c", dest="config_file",
                               help="Configuration file path",
                               default=CONFIG_FILE)

    # Config value overrides
    config_parser.add_argument("--tmdb-api-key",
                               help="TMDB API key")
    config_parser.add_argument("--open_subtitles_api_key",
                               help="OpenSubtitles API key")
    config_parser.add_argument("--open_subtitles_user_agent",
                               help="OpenSubtitles User Agent")
    config_parser.add_argument("--open_subtitles_username",
                               help="OpenSubtitles Username")
    config_parser.add_argument("--open_subtitles_password",
                               help="OpenSubtitles Password")

    return config_parser


def get_episode_parser():
    episode_parser = argparse.ArgumentParser(add_help=False)
    episode_group = episode_parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--seasons", dest="season_numbers",
        type=int,
        default=None,
        nargs="+",
        help="Specify the seasons to be processed (default: all seasons).")

    episode_group.add_argument(
        "--episodes",
        dest="episodes_specifiers",
        action=EpisodesSpecifierAction,
        default=None,
        nargs="+",
        metavar="SEASON:SPEC",
        help=(
            "Specify episodes as season:specifier where specifier is one of: "
            "N (int), A,B,C (list), A-B or -B or A- (range), or * (all). "
            "Examples: 5:1 8:2 | 2:1,2,3 4:3-4 | 1:* 2:-4 | 5:2-"
        ),
    )

    return episode_parser
