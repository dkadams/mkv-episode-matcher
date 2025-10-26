import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, \
    as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Self

from rich.console import Console

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import episode_from_path, EpisodeKey
from mkv_episode_matcher.extract_text_segments_worker import \
    _init_text_extractor_worker, _extract_text_segments_worker
from mkv_episode_matcher.series import Series

console = Console()

@dataclass(frozen=True, eq=True)
class Score:
    """
    Represents how well an episode matched a subtitle file.
        -count: Number of segments matched
        -min_distance: Minimum distance of all the matched segments
    """
    count: int
    min_distance: float

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
class Match:
    season: int
    episode: int
    score: Score

    def key(self) -> EpisodeKey:
        return EpisodeKey(self.season, self.episode)

@dataclass(frozen=True, eq=True)
class MatchResult:
    file: Path
    matches: list[Match]
    known_episode: tuple[int, int]

class IndexedEpisodeMatcher:
    def __init__(self, config: Configuration, series: Series):
        self.config = config
        self.series = series

        index_cls = config.args.index_type.reader_type
        self.index = index_cls(config, series)

        self.text_extractor_model = "small.en"
        self.segment_duration = 30
        self.segment_count = 10

        self.extracted_text_dir = self.series.dot_dir / "extracted-text"
        self.extracted_text_dir.mkdir(exist_ok=True)

        self.cache = TextSegmentCache(config, self.extracted_text_dir,
                                      self.segment_duration, self.segment_count)

    def match(self, paths) -> List[MatchResult]:
        files = list(self._collect_files(paths))
        if not files:
            return []

        text_segments_map = self._ensure_text_segments(files)

        query_results: Dict[Path, List[Match]] = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {
                executor.submit(self.index.query_intervals, text_segments_map[file]): file
                for file in files
            }
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                query_results[file] = future.result()

        results = []
        for file in files:
            matches = query_results[file]
            actual_episode = episode_from_path(file)
            results.append(MatchResult(file, matches, actual_episode))

        return results

    @staticmethod
    def _collect_files(paths: Iterable[Path]) -> Iterable[Path]:
        for path in paths:
            if path.is_file():
                yield path
            else:
                yield from (candidate for candidate in path.rglob("*.mkv")
                            if candidate.is_file())

    def _ensure_text_segments(self, files: List[Path]) -> Dict[Path, List[Tuple[int, str]]]:
        if not files:
            return {}

        cached_segments = self.cache.get_cached_segments(files)
        missing_files = files - cached_segments.keys()
        extracted_segments = self.extract_segments(missing_files) if missing_files else {}

        return cached_segments | extracted_segments

    def extract_segments(self, missing_files: Iterable[Path]) -> Dict[Path, list[tuple[int, str]]]:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=4,
            initializer=_init_text_extractor_worker,
            initargs=(self.config.args.transcriber, self.text_extractor_model,
                      self.segment_duration, self.segment_count),
            mp_context=ctx,
        ) as executor:
            results = executor.map(_extract_text_segments_worker, missing_files)

            text_segments = {file: result
                             for file, result in zip(missing_files, results)}

            self.cache.write_cache(text_segments)
            return text_segments

class TextSegmentCache:
    def __init__(self, config: Configuration, extracted_text_dir: Path,
                 segment_duration: int, segment_count: int):
        self.config = config
        self.extracted_text_dir = extracted_text_dir
        self.segment_duration = segment_duration
        self.segment_count = segment_count

    def _ensure_cache_dir(self) -> Path:
        cache_dir = self.extracted_text_dir / f"dur{self.segment_duration}s_count{self.segment_count}"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def get_cached_segments(self, files: List[Path]) -> Dict[Path, List[Tuple[int, str]]]:
        if self.config.args.no_transcription_cache:
            return dict()

        cache_dir = self._ensure_cache_dir()
        text_segments: Dict[Path, List[Tuple[int, str]]] = {}

        for file in files:
            cache_file = cache_dir / file.with_suffix(".json").name
            if not cache_file.exists():
                continue
            with open(cache_file, "r") as json_in:
                text_segments[file] = json.load(json_in)

        return text_segments

    def write_cache(self, text_segments: Dict[Path, List[Tuple[int, str]]]):
        cache_dir = self._ensure_cache_dir()

        for file, segments in text_segments.items():
            cache_file = cache_dir / file.with_suffix(".json").name
            if not cache_file.exists():
                with open(cache_file, "w") as json_out:
                    json.dump(segments, json_out)
