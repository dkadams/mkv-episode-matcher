import argparse

from mkv_episode_matcher import __version__
from mkv_episode_matcher.annoy_subtitle_index import AnnoySubtitleIndex
from mkv_episode_matcher.chroma_subtitle_index import ChromaSubtitleIndex
from mkv_episode_matcher.config import edit_config, CONFIG_FILE
from mkv_episode_matcher.dataset_collector import collect_dataset
from mkv_episode_matcher.dataset_evaluator import evaluate_dataset
from mkv_episode_matcher.episode_matcher import match_episodes
from mkv_episode_matcher.episodes_specifier import EpisodesSpecifierAction
from mkv_episode_matcher.hnswlib_subtitle_index import HnswlibSubtitleIndex
from mkv_episode_matcher.series_initializer import init_series
from mkv_episode_matcher.subtitle_downloader import download_subtitles
from mkv_episode_matcher.subtitle_index import index_subtitles
from mkv_episode_matcher.transcriber_benchmark import (
    BENCHMARK_BACKEND_CHOICES,
    benchmark_transcribers,
)
from mkv_episode_matcher.transcribers import (
    WhisperTranscriber,
    FasterWhisperTranscriber,
    WhispercppCliTranscriber,
    WhisperKitCliTranscriber,
    ParakeetMlxCliTranscriber,
)

def build_args_parser():
    parser = get_root_parser()

    # get parent parsers
    config_parser = get_config_parser()
    series_dir_parser = get_series_dir_parser()
    episode_parser = get_episode_parser()
    index_parser = get_index_parser()

    subparsers = parser.add_subparsers(required=True)

    add_config_parser(subparsers, config_parser)
    add_init_series(subparsers, config_parser, series_dir_parser)
    add_fetch_subs(subparsers, config_parser, series_dir_parser, episode_parser)

    add_index_subs(
        subparsers,
        config_parser,
        series_dir_parser,
        episode_parser,
        index_parser,
    )
    add_collect_dataset(
        subparsers,
        config_parser,
        series_dir_parser,
        episode_parser,
        index_parser,
    )
    add_evaluate_dataset(subparsers, config_parser)
    add_match(subparsers, config_parser, index_parser)
    add_benchmark_transcribers(subparsers, config_parser)

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
    parser.add_argument(
        "--log-dir",
        default=argparse.SUPPRESS,
        help=(
            "Directory where log files are written for this run "
            "(overrides config; otherwise uses [logging].log_dir or "
            "~/.mkv-episode-matcher/logs)"
        ),
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
                                  help="Refresh the series details")
    init_show_parser.add_argument("--segment-duration",
                                  type=int,
                                  help="The number of seconds to use for segmenting episodes (default: 30)",
                                  default=30)
    init_show_parser.add_argument("--random-seed",
                                  type=int,
                                  help="The random seed to use for segmenting episodes (default: 12345)",
                                  default=12345)
    init_show_parser.set_defaults(func=init_series)


def add_fetch_subs(subparsers, config_parser, series_dir_parser, episode_parser):
    fetch_subs_parser = subparsers.add_parser("fetch-subs",
                                              parents=[config_parser,
                                                       series_dir_parser,
                                                       episode_parser],
                                              help="Fetch subtitles for a series")
    fetch_subs_parser.add_argument("--refresh", action="store_true",
                                   help="Download subtitles even if they already exist")
    fetch_subs_parser.set_defaults(func=download_subtitles)

def add_index_subs(subparsers, config_parser, series_dir_parser, episode_parser,
    index_parser):
    index_subs_parser = subparsers.add_parser("index-subs",
                                              parents=[config_parser,
                                                       series_dir_parser,
                                                       episode_parser,
                                                       index_parser],
                                              help="Index subtitles for a series")

    index_subs_parser.add_argument("--rebuild",
                                   action="store_true",
                                   help="Rebuild the index from scratch")
    index_subs_parser.set_defaults(func=index_subtitles)

def add_collect_dataset(subparsers, config_parser, series_dir_parser, episode_parser,
    index_parser):
    collect_parser = subparsers.add_parser(
        "collect-dataset",
        parents=[config_parser, series_dir_parser, episode_parser, index_parser],
        help="Collect labeled transcription segments and subtitles for evaluation",
    )

    collect_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to write the collected dataset",
    )

    collect_parser.add_argument(
        "--misalign-profiles",
        nargs="+",
        choices=["left", "right", "random"],
        default=["left", "right", "random"],
        help="Generate misaligned transcription variants for selected profiles",
    )

    collect_parser.add_argument(
        "--misalign-min-seconds",
        type=float,
        default=1.0,
        help="Minimum absolute audio shift in seconds for misaligned variants",
    )

    collect_parser.add_argument(
        "--misalign-max-seconds",
        type=float,
        default=3.0,
        help="Maximum absolute audio shift in seconds for misaligned variants",
    )

    collect_parser.add_argument(
        "--misalign-seed",
        type=int,
        default=None,
        help="Random seed for deterministic misalignment generation",
    )

    collect_parser.add_argument(
        "--misalign-per-segment",
        action="store_true",
        help="Apply new offset per segment for left/right profiles",
    )

    collect_parser.add_argument(
        "--include-aligned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include aligned transcript variants in the collected dataset",
    )

    collect_parser.add_argument(
        "--segment-duration",
        type=int,
        default=None,
        help="Override the segment duration for this collection run",
    )

    collect_parser.add_argument(
        "--segments-per-minute",
        type=float,
        default=.5,
        help="Number of segments to extract per minute (default: .5)",
    )

    collect_parser.add_argument(
        "--no-transcription-cache",
        action="store_true",
        help="Don't read or create transcribed text cache",
    )

    xscriber_group = collect_parser.add_mutually_exclusive_group()
    xscriber_group.add_argument(
        "--whisper", dest="transcriber",
        action="store_const", const=WhisperTranscriber,
        help="Use Whisper for transcription.")
    xscriber_group.add_argument(
        "--faster-whisper", dest="transcriber",
        action="store_const", const=FasterWhisperTranscriber,
        help="Use Faster Whisper for transcription.")
    xscriber_group.add_argument(
        "--whispercpp-cli", dest="transcriber",
        action="store_const", const=WhispercppCliTranscriber,
        help="Use whisper.cpp's CLI for transcription.")
    xscriber_group.add_argument(
        "--whisperkit-cli", dest="transcriber",
        action="store_const", const=WhisperKitCliTranscriber,
        help="Use WhisperKit's CLI for transcription.")
    xscriber_group.add_argument(
        "--parakeet-mlx", dest="transcriber",
        action="store_const", const=ParakeetMlxCliTranscriber,
        help="Use parakeet-mlx for transcription.")

    collect_parser.set_defaults(
        func=collect_dataset,
        transcriber=ParakeetMlxCliTranscriber,
    )

def add_evaluate_dataset(subparsers, config_parser):
    evaluate_parser = subparsers.add_parser(
        "evaluate-dataset",
        parents=[config_parser],
        help="Evaluate transcript-to-subtitle matching quality from a collected dataset",
    )

    evaluate_parser.add_argument(
        "dataset_dir",
        help="Path to a dataset directory created by collect-dataset",
    )
    evaluate_parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["aligned", "left", "right", "random"],
        default=["aligned", "left", "right", "random"],
        help="Filter manifest entries to specific variant profiles",
    )
    evaluate_parser.add_argument(
        "--report-by-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-profile metrics in output",
    )
    evaluate_parser.add_argument(
        "--output",
        help="Optional path to write JSON report output",
    )
    evaluate_parser.add_argument(
        "--segment-duration",
        type=int,
        default=None,
        help="Override segment duration in seconds (defaults to meta.json or 30)",
    )
    evaluate_parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k values to report",
    )
    evaluate_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N manifest records (videos)",
    )
    evaluate_parser.add_argument(
        "--show-failures",
        type=int,
        default=20,
        help="Maximum failed examples to include in report output",
    )
    evaluate_parser.set_defaults(func=evaluate_dataset)


def add_match(subparsers, config_parser, index_parser):
    match_parser = subparsers.add_parser("match",
                                         parents=[config_parser, index_parser],
                                         help="Match episodes of a series")

    match_parser.add_argument('video_files',
                              nargs='+',
                              help="Path to one or more video files to match, "
                                   "or directories to recursively search for video files")
    match_parser.add_argument('--extension', '-e',
                              nargs='+',
                              default=".mkv",
                              help="File extension to match (default: .mkv)")

    match_parser.add_argument('--no-transcription-cache',
                              action="store_true",
                              help="Don't read or create transcribed text cache")

    match_parser.add_argument('--display-by-episode', '-E',
                              dest="display_by_episode",
                              action="store_true",
                              help="Display results by episode")
    match_parser.add_argument('--display-by-file', '-F',
                              dest="display_by_file",
                              action="store_true",
                              help="Display results by file")

    match_parser.add_argument('--segments-per-minute',
                              type=float,
                              default=.5,
                              help="Number of segments to extract per minute (default: .5)")

    match_parser.add_argument('--num-matches','-n',
                              type=int,
                              default=5,
                              help="Number of matches to show (default: 5)")

    xscriber_group = match_parser.add_mutually_exclusive_group()
    xscriber_group.add_argument(
        "--whisper", dest="transcriber",
        action="store_const", const=WhisperTranscriber,
        help="Use Whisper for transcription.")
    xscriber_group.add_argument(
        "--faster-whisper", dest="transcriber",
        action="store_const", const=FasterWhisperTranscriber,
        help="Use Faster Whisper for transcription.")
    xscriber_group.add_argument(
        "--whispercpp-cli", dest="transcriber",
        action="store_const", const=WhispercppCliTranscriber,
        help="Use whisper.cpp's CLI for transcription.")
    xscriber_group.add_argument(
        "--whisperkit-cli", dest="transcriber",
        action="store_const", const=WhisperKitCliTranscriber,
        help="Use WhisperKit's CLI for transcription.")
    xscriber_group.add_argument(
        "--parakeet-mlx", dest="transcriber",
        action="store_const", const=ParakeetMlxCliTranscriber,
        help="Use parakeet-mlx for transcription.")


    match_parser.set_defaults(func=match_episodes,
                              transcriber=ParakeetMlxCliTranscriber,
                              display_by_episode=True,)


def add_benchmark_transcribers(subparsers, config_parser):
    parser = subparsers.add_parser(
        "benchmark-transcribers",
        parents=[config_parser],
        help="Benchmark transcription backends against one or more input paths",
    )
    parser.add_argument(
        "video_files",
        nargs="+",
        help="Path to one or more video files, or directories to recursively search",
    )
    parser.add_argument(
        "--extension",
        "-e",
        nargs="+",
        default=[".mkv"],
        help="File extension(s) to include when scanning directories (default: .mkv)",
    )
    parser.add_argument(
        "--segments-per-minute",
        type=float,
        default=.5,
        help="Number of segments to extract per minute (default: .5)",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=None,
        help="Override segment duration in seconds (default: series setting or 30)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override random seed (default: series setting or 12345)",
    )
    parser.add_argument(
        "--thread-workers",
        type=int,
        default=10,
        help="Thread worker count for subprocess backends (default: 10)",
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        default=8,
        help="Process worker count for Python model backends (default: 8)",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=BENCHMARK_BACKEND_CHOICES,
        help="Backend(s) to benchmark (repeatable). Default: all backends.",
    )
    parser.set_defaults(func=benchmark_transcribers)

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
    config_parser.add_argument(
        "--set-log-dir",
        dest="set_log_dir",
        help="Persist log directory in config.ini under [logging].log_dir",
    )

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

def get_index_parser():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--chroma", dest="index_type",
        action="store_const", const=ChromaSubtitleIndex,
        help="Use chroma for indexes.")
    group.add_argument(
        "--annoy", dest="index_type",
        action="store_const", const=AnnoySubtitleIndex,
        help="Use Annoy for indexes.")
    group.add_argument(
        "--hnswlib", dest="index_type",
        action="store_const", const=HnswlibSubtitleIndex,
        help="Use hnswlib for indexes.")

    parser.set_defaults(index_type=HnswlibSubtitleIndex)
    return parser
