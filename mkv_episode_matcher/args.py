import argparse

from mkv_episode_matcher import __version__
from mkv_episode_matcher.annoy_subtitle_index import AnnoySubtitleIndex
from mkv_episode_matcher.config import edit_config, CONFIG_FILE
from mkv_episode_matcher.dataset_collector import collect_dataset
from mkv_episode_matcher.dataset_evaluator import evaluate_dataset
from mkv_episode_matcher.episode_matcher import match_episodes
from mkv_episode_matcher.episodes_specifier import EpisodesSpecifierAction
from mkv_episode_matcher.hnswlib_subtitle_index import HnswlibSubtitleIndex
from mkv_episode_matcher.series_initializer import init_series
from mkv_episode_matcher.subtitle_downloader import download_subtitles
from mkv_episode_matcher.subtitle_index import index_subtitles
from mkv_episode_matcher.subtitle_quality import (
    DEFAULT_SUBTITLE_QUALITY_ENABLED,
    DEFAULT_SUBTITLE_QUALITY_MAX_CANDIDATES,
    DEFAULT_SUBTITLE_QUALITY_OVERLAP_CONTAINMENT_THRESHOLD,
    DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MAX,
    DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MIN,
)
from mkv_episode_matcher.multi_episode_assignment import (
    DEFAULT_MULTI_EPISODE_CANDIDATE_K,
    DEFAULT_MULTI_EPISODE_CANDIDATE_K_RETRY,
    DEFAULT_MULTI_EPISODE_DURATION_RATIO_THRESHOLD,
    DEFAULT_MULTI_EPISODE_MIN_EXTRA_MINUTES,
    DEFAULT_MULTI_EPISODE_MIN_EXTRA_SEGMENTS,
    DEFAULT_MULTI_EPISODE_MIN_SIDE_SEGMENTS,
    DEFAULT_MULTI_EPISODE_MISS_PENALTY,
    DEFAULT_MULTI_EPISODE_MODE,
    DEFAULT_MULTI_EPISODE_PAIR_MARGIN,
    DEFAULT_MULTI_EPISODE_SECOND_HALF_HORIZON_MULTIPLIER,
    DEFAULT_MULTI_EPISODE_SEGMENTS_RATIO_THRESHOLD,
    DEFAULT_MULTI_EPISODE_SPLIT_SEARCH_WINDOW_SECONDS,
    MULTI_EPISODE_MODES,
)
from mkv_episode_matcher.transcriber_benchmark import (
    BENCHMARK_BACKEND_CHOICES,
    benchmark_transcribers,
)
from mkv_episode_matcher.transcribers import (
    FasterWhisperTranscriber,
    ParakeetMlxTranscriber,
    WhispercppTranscriber,
    get_default_transcriber_name,
    get_default_transcriber_type,
)


class NoAbbrevArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def build_args_parser():
    parser = get_root_parser()

    # get parent parsers
    config_parser = get_config_parser()
    series_dir_parser = get_series_dir_parser()
    episode_parser = get_episode_parser()
    index_parser = get_index_parser()

    subparsers = parser.add_subparsers(
        required=True,
        parser_class=NoAbbrevArgumentParser,
    )

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
    parser = NoAbbrevArgumentParser(
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
    init_show_parser.add_argument(
        "--subtitle-overlap-seconds",
        type=int,
        default=5,
        help="Subtitle window overlap in seconds for indexing/matching (default: 5)",
    )
    init_show_parser.add_argument("--random-seed",
                                  type=int,
                                  help="The random seed to use for segmenting episodes (default: 12345)",
                                  default=12345)
    init_show_parser.add_argument(
        "--window-expansion-mode",
        choices=["always", "two-stage"],
        default="always",
        help="Neighbor-window expansion mode for retrieval (default: always)",
    )
    init_show_parser.add_argument(
        "--window-neighbor-radius",
        type=int,
        default=1,
        help="Neighbor-window radius for retrieval (default: 1)",
    )
    init_show_parser.add_argument(
        "--low-info-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter low-information transcript segments during matching/evaluation",
    )
    init_show_parser.add_argument(
        "--low-info-min-words",
        type=int,
        default=8,
        help="Minimum token count before transcript segments are considered low-information",
    )
    init_show_parser.add_argument(
        "--low-info-cue-ratio",
        type=float,
        default=0.25,
        help="Cue-token ratio threshold used for low-information filtering",
    )
    init_show_parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=10,
        help="Maximum candidates retrieved per interval query (default: 10)",
    )
    init_show_parser.add_argument(
        "--subtitle-quality",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SUBTITLE_QUALITY_ENABLED,
        help="Enable subtitle quality checks during fetch-subs",
    )
    init_show_parser.add_argument(
        "--subtitle-quality-max-candidates",
        type=int,
        default=DEFAULT_SUBTITLE_QUALITY_MAX_CANDIDATES,
        help="Maximum candidate subtitles to evaluate per episode",
    )
    init_show_parser.add_argument(
        "--subtitle-quality-runtime-ratio-max",
        type=float,
        default=DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MAX,
        help="Hard-fail threshold when subtitle runtime ratio exceeds this maximum",
    )
    init_show_parser.add_argument(
        "--subtitle-quality-runtime-ratio-min",
        type=float,
        default=DEFAULT_SUBTITLE_QUALITY_RUNTIME_RATIO_MIN,
        help="Hard-fail threshold when subtitle runtime ratio drops below this minimum",
    )
    init_show_parser.add_argument(
        "--subtitle-quality-overlap-containment-threshold",
        type=float,
        default=DEFAULT_SUBTITLE_QUALITY_OVERLAP_CONTAINMENT_THRESHOLD,
        help="Neighbor subtitle line-containment threshold considered suspicious",
    )
    init_show_parser.add_argument(
        "--multi-episode-mode",
        choices=sorted(MULTI_EPISODE_MODES),
        default=DEFAULT_MULTI_EPISODE_MODE,
        help="Multi-episode handling mode for matching/evaluation",
    )
    init_show_parser.add_argument(
        "--multi-episode-duration-ratio-threshold",
        type=float,
        default=DEFAULT_MULTI_EPISODE_DURATION_RATIO_THRESHOLD,
        help="Duration ratio threshold for auto multi-episode detection",
    )
    init_show_parser.add_argument(
        "--multi-episode-segments-ratio-threshold",
        type=float,
        default=DEFAULT_MULTI_EPISODE_SEGMENTS_RATIO_THRESHOLD,
        help="Transcribed segment count ratio threshold for auto multi-episode detection",
    )
    init_show_parser.add_argument(
        "--multi-episode-min-extra-minutes",
        type=float,
        default=DEFAULT_MULTI_EPISODE_MIN_EXTRA_MINUTES,
        help="Minimum extra minutes above expected single runtime for multi detection",
    )
    init_show_parser.add_argument(
        "--multi-episode-min-extra-segments",
        type=int,
        default=DEFAULT_MULTI_EPISODE_MIN_EXTRA_SEGMENTS,
        help="Minimum extra transcribed segments above expected single runtime for multi detection",
    )
    init_show_parser.add_argument(
        "--multi-episode-split-search-window-seconds",
        type=int,
        default=DEFAULT_MULTI_EPISODE_SPLIT_SEARCH_WINDOW_SECONDS,
        help="Local split search window around expected split (seconds)",
    )
    init_show_parser.add_argument(
        "--multi-episode-min-side-segments",
        type=int,
        default=DEFAULT_MULTI_EPISODE_MIN_SIDE_SEGMENTS,
        help="Minimum segment count required on each side of a candidate split",
    )
    init_show_parser.add_argument(
        "--multi-episode-candidate-k",
        type=int,
        default=DEFAULT_MULTI_EPISODE_CANDIDATE_K,
        help="Per-chunk episode candidate count before consecutive pair selection",
    )
    init_show_parser.add_argument(
        "--multi-episode-candidate-k-retry",
        type=int,
        default=DEFAULT_MULTI_EPISODE_CANDIDATE_K_RETRY,
        help="Retry per-chunk episode candidate count if no consecutive pair is found",
    )
    init_show_parser.add_argument(
        "--multi-episode-second-half-horizon-multiplier",
        type=float,
        default=DEFAULT_MULTI_EPISODE_SECOND_HALF_HORIZON_MULTIPLIER,
        help="Multiplier for second-half neighbor-window horizon",
    )
    init_show_parser.add_argument(
        "--multi-episode-pair-margin",
        type=float,
        default=DEFAULT_MULTI_EPISODE_PAIR_MARGIN,
        help="Required margin between best and second-best pair score",
    )
    init_show_parser.add_argument(
        "--multi-episode-miss-penalty",
        type=float,
        default=DEFAULT_MULTI_EPISODE_MISS_PENALTY,
        help="Penalty applied when an episode is missing for a chunk segment",
    )
    init_show_parser.set_defaults(func=init_series)


def add_fetch_subs(subparsers, config_parser, series_dir_parser, episode_parser):
    fetch_subs_parser = subparsers.add_parser("fetch-subs",
                                              parents=[config_parser,
                                                       series_dir_parser,
                                                       episode_parser],
                                              help="Fetch subtitles for a series")
    fetch_subs_parser.add_argument("--refresh", action="store_true",
                                   help="Download subtitles even if they already exist")
    fetch_subs_parser.add_argument(
        "--subtitle-quality",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable subtitle quality checks and candidate retry selection",
    )
    fetch_subs_parser.add_argument(
        "--subtitle-quality-max-candidates",
        type=int,
        default=None,
        help="Maximum candidate subtitles to evaluate per episode (default: series setting or 3)",
    )
    fetch_subs_parser.add_argument(
        "--subtitle-quality-runtime-ratio-max",
        type=float,
        default=None,
        help="Runtime ratio hard-fail upper bound (default: series setting or 1.45)",
    )
    fetch_subs_parser.add_argument(
        "--subtitle-quality-runtime-ratio-min",
        type=float,
        default=None,
        help="Runtime ratio hard-fail lower bound (default: series setting or 0.55)",
    )
    fetch_subs_parser.add_argument(
        "--subtitle-quality-overlap-containment-threshold",
        type=float,
        default=None,
        help="Neighbor overlap containment threshold (default: series setting or 0.35)",
    )
    fetch_subs_parser.add_argument(
        "--subtitle-quality-report",
        default=None,
        help="Optional JSON path to write subtitle quality summary report",
    )
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
    index_subs_parser.add_argument(
        "--subtitle-overlap-seconds",
        type=int,
        default=None,
        help="Override subtitle window overlap in seconds (default: series setting or 5)",
    )
    index_subs_parser.add_argument(
        "--include-quarantined-subs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include subtitle files marked as quarantined by subtitle quality checks",
    )
    index_subs_parser.set_defaults(func=index_subtitles)

def add_collect_dataset(subparsers, config_parser, series_dir_parser, episode_parser,
    index_parser):
    default_backend = get_default_transcriber_name()
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
        "--subtitle-overlap-seconds",
        type=int,
        default=None,
        help="Override subtitle window overlap in seconds for this collection run",
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
        "--whispercpp",
        dest="transcriber",
        action="store_const",
        const=WhispercppTranscriber,
        help=f"Use whisper.cpp CLI for transcription (default on this platform: {default_backend}).",
    )
    xscriber_group.add_argument(
        "--parakeet-mlx",
        dest="transcriber",
        action="store_const",
        const=ParakeetMlxTranscriber,
        help=f"Use parakeet-mlx for transcription (macOS only; default on this platform: {default_backend}).",
    )
    xscriber_group.add_argument(
        "--faster-whisper",
        dest="transcriber",
        action="store_const",
        const=FasterWhisperTranscriber,
        help=(
            "Use faster-whisper batched transcription "
            f"(default on this platform: {default_backend})."
        ),
    )

    collect_parser.set_defaults(
        func=collect_dataset,
        transcriber=get_default_transcriber_type(),
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
        "--subtitle-overlap-seconds",
        type=int,
        default=None,
        help="Override subtitle window overlap in seconds (defaults to meta.json, series, or 5)",
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
    evaluate_parser.add_argument(
        "--errors-output",
        help="Optional JSONL/CSV path to write per-segment top-1 mismatches",
    )
    evaluate_parser.add_argument(
        "--multi-failures-output",
        help=(
            "Optional HTML path to write detailed multi-episode assignment "
            "fallback diagnostics"
        ),
    )
    evaluate_parser.add_argument(
        "--include-match-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include matched subtitle window text/time details for top predictions "
            "in mismatch and failure outputs"
        ),
    )
    evaluate_parser.add_argument(
        "--include-quarantined-subs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include subtitle files marked as quarantined by subtitle quality checks",
    )
    evaluate_parser.add_argument(
        "--window-expansion-mode",
        choices=["always", "two-stage"],
        default=None,
        help="Neighbor-window expansion mode (default: series/meta setting or always)",
    )
    evaluate_parser.add_argument(
        "--window-neighbor-radius",
        type=int,
        default=None,
        help="Neighbor-window radius (default: series/meta setting or 1)",
    )
    evaluate_parser.add_argument(
        "--low-info-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Filter low-information transcript segments before scoring",
    )
    evaluate_parser.add_argument(
        "--low-info-min-words",
        type=int,
        default=None,
        help="Minimum words threshold for low-information filtering",
    )
    evaluate_parser.add_argument(
        "--low-info-cue-ratio",
        type=float,
        default=None,
        help="Cue-token ratio threshold for low-information filtering",
    )
    evaluate_parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=None,
        help="Maximum candidates retrieved per queried window",
    )
    evaluate_parser.add_argument(
        "--support-window-bonus",
        type=float,
        default=0.012,
        help="Support bonus subtracted per additional supporting window",
    )
    evaluate_parser.add_argument(
        "--support-offset-penalty",
        type=float,
        default=0.003,
        help="Penalty per window of offset from mapped interval in support-aware scoring",
    )
    evaluate_parser.add_argument(
        "--multi-episode-mode",
        choices=sorted(MULTI_EPISODE_MODES),
        default=None,
        help="Multi-episode handling mode (default: series/meta setting or auto)",
    )
    evaluate_parser.add_argument(
        "--multi-episode-duration-ratio-threshold",
        type=float,
        default=None,
        help="Duration ratio threshold for auto multi-episode detection",
    )
    evaluate_parser.add_argument(
        "--multi-episode-segments-ratio-threshold",
        type=float,
        default=None,
        help="Transcribed segment ratio threshold for auto multi-episode detection",
    )
    evaluate_parser.add_argument(
        "--multi-episode-min-extra-minutes",
        type=float,
        default=None,
        help="Minimum extra minutes above expected single runtime for multi detection",
    )
    evaluate_parser.add_argument(
        "--multi-episode-min-extra-segments",
        type=int,
        default=None,
        help="Minimum extra segments above expected single runtime for multi detection",
    )
    evaluate_parser.add_argument(
        "--multi-episode-split-search-window-seconds",
        type=int,
        default=None,
        help="Local split search window around expected split in seconds",
    )
    evaluate_parser.add_argument(
        "--multi-episode-min-side-segments",
        type=int,
        default=None,
        help="Minimum segment count required on each side of a split",
    )
    evaluate_parser.add_argument(
        "--multi-episode-candidate-k",
        type=int,
        default=None,
        help="Per-chunk episode candidate count for pair selection",
    )
    evaluate_parser.add_argument(
        "--multi-episode-candidate-k-retry",
        type=int,
        default=None,
        help="Retry candidate count if no consecutive pair is found",
    )
    evaluate_parser.add_argument(
        "--multi-episode-second-half-horizon-multiplier",
        type=float,
        default=None,
        help="Second-half neighbor-window horizon multiplier",
    )
    evaluate_parser.add_argument(
        "--multi-episode-pair-margin",
        type=float,
        default=None,
        help="Required score margin between top two candidate pairs",
    )
    evaluate_parser.add_argument(
        "--multi-episode-miss-penalty",
        type=float,
        default=None,
        help="Penalty for chunk segments missing a candidate episode hit",
    )
    evaluate_parser.set_defaults(func=evaluate_dataset)


def add_match(subparsers, config_parser, index_parser):
    default_backend = get_default_transcriber_name()
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
    match_parser.add_argument(
        "--subtitle-overlap-seconds",
        type=int,
        default=None,
        help="Override subtitle window overlap in seconds (default: series setting or 5)",
    )
    match_parser.add_argument(
        "--window-expansion-mode",
        choices=["always", "two-stage"],
        default=None,
        help="Neighbor-window expansion mode (default: series setting or always)",
    )
    match_parser.add_argument(
        "--window-neighbor-radius",
        type=int,
        default=None,
        help="Neighbor-window radius (default: series setting or 1)",
    )
    match_parser.add_argument(
        "--low-info-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Filter low-information transcript segments before indexed matching",
    )
    match_parser.add_argument(
        "--low-info-min-words",
        type=int,
        default=None,
        help="Minimum words threshold for low-information filtering",
    )
    match_parser.add_argument(
        "--low-info-cue-ratio",
        type=float,
        default=None,
        help="Cue-token ratio threshold for low-information filtering",
    )
    match_parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=None,
        help="Maximum candidates retrieved per queried window",
    )
    match_parser.add_argument(
        "--multi-episode-mode",
        choices=sorted(MULTI_EPISODE_MODES),
        default=None,
        help="Multi-episode handling mode (default: series setting or auto)",
    )
    match_parser.add_argument(
        "--multi-episode-duration-ratio-threshold",
        type=float,
        default=None,
        help="Duration ratio threshold for auto multi-episode detection",
    )
    match_parser.add_argument(
        "--multi-episode-segments-ratio-threshold",
        type=float,
        default=None,
        help="Transcribed segment ratio threshold for auto multi-episode detection",
    )
    match_parser.add_argument(
        "--multi-episode-min-extra-minutes",
        type=float,
        default=None,
        help="Minimum extra minutes above expected single runtime for multi detection",
    )
    match_parser.add_argument(
        "--multi-episode-min-extra-segments",
        type=int,
        default=None,
        help="Minimum extra segments above expected single runtime for multi detection",
    )
    match_parser.add_argument(
        "--multi-episode-split-search-window-seconds",
        type=int,
        default=None,
        help="Local split search window around expected split in seconds",
    )
    match_parser.add_argument(
        "--multi-episode-min-side-segments",
        type=int,
        default=None,
        help="Minimum segment count required on each side of a split",
    )
    match_parser.add_argument(
        "--multi-episode-candidate-k",
        type=int,
        default=None,
        help="Per-chunk episode candidate count for pair selection",
    )
    match_parser.add_argument(
        "--multi-episode-candidate-k-retry",
        type=int,
        default=None,
        help="Retry candidate count if no consecutive pair is found",
    )
    match_parser.add_argument(
        "--multi-episode-second-half-horizon-multiplier",
        type=float,
        default=None,
        help="Second-half neighbor-window horizon multiplier",
    )
    match_parser.add_argument(
        "--multi-episode-pair-margin",
        type=float,
        default=None,
        help="Required score margin between top two candidate pairs",
    )
    match_parser.add_argument(
        "--multi-episode-miss-penalty",
        type=float,
        default=None,
        help="Penalty for chunk segments missing a candidate episode hit",
    )

    match_parser.add_argument('--num-matches','-n',
                              type=int,
                              default=5,
                              help="Number of matches to show (default: 5)")
    match_parser.add_argument(
        "--transcribe-workers",
        "--xscribe-workers",
        dest="transcribe_workers",
        type=int,
        default=4,
        help="Worker count for transcription jobs (default: 4)",
    )
    match_parser.add_argument(
        "--io-workers",
        dest="io_workers",
        type=int,
        default=2,
        help="Worker count for audio extraction I/O stage (default: 2)",
    )

    xscriber_group = match_parser.add_mutually_exclusive_group()
    xscriber_group.add_argument(
        "--whispercpp",
        dest="transcriber",
        action="store_const",
        const=WhispercppTranscriber,
        help=f"Use whisper.cpp CLI for transcription (default on this platform: {default_backend}).",
    )
    xscriber_group.add_argument(
        "--parakeet-mlx",
        dest="transcriber",
        action="store_const",
        const=ParakeetMlxTranscriber,
        help=f"Use parakeet-mlx for transcription (macOS only; default on this platform: {default_backend}).",
    )
    xscriber_group.add_argument(
        "--faster-whisper",
        dest="transcriber",
        action="store_const",
        const=FasterWhisperTranscriber,
        help=(
            "Use faster-whisper batched transcription "
            f"(default on this platform: {default_backend})."
        ),
    )


    match_parser.set_defaults(func=match_episodes,
                              transcriber=get_default_transcriber_type(),
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
        "--transcribe-workers",
        "--xscribe-workers",
        dest="transcribe_workers",
        type=int,
        default=4,
        help="Worker count for transcription jobs (default: 4)",
    )
    parser.add_argument(
        "--io-workers",
        dest="io_workers",
        type=int,
        default=2,
        help="Worker count for audio extraction I/O stage (default: 2)",
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
        "--annoy", dest="index_type",
        action="store_const", const=AnnoySubtitleIndex,
        help="Use Annoy for indexes.")
    group.add_argument(
        "--hnswlib", dest="index_type",
        action="store_const", const=HnswlibSubtitleIndex,
        help="Use hnswlib for indexes.")

    parser.set_defaults(index_type=HnswlibSubtitleIndex)
    return parser
