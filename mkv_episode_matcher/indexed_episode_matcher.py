import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, \
    as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from guessit import guessit
from rich.console import Console

from mkv_episode_matcher.annoy_subtitle_index import AnnoySubtitleIndexReader
from mkv_episode_matcher.chroma_subtitle_index import (
    ChromaSubtitleIndex,
    ChromaSubtitleIndexReader,
)
from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.extract_text_segments_worker import \
    _init_text_extractor_worker, _extract_text_segments_worker
from mkv_episode_matcher.hnswlib_subtitle_index import \
    HnswlibSubtitleIndexReader
from mkv_episode_matcher.series import Series, get_series

console = Console()

@dataclass
class MatchResult:
    file: Path
    matches: List[Tuple[Tuple[float, int] or float, str, str]]
    known_episode: Tuple[int, int]

class IndexedEpisodeMatcher:
    def __init__(self, config: Configuration, series: Series):
        self.config = config
        self.series = series

        if config.args.index_format == "chroma":
            self.index = ChromaSubtitleIndexReader(config, series)
        elif config.args.index_format == "annoy":
            self.index = AnnoySubtitleIndexReader(config, series)
        elif config.args.index_format == "hnswlib":
            self.index = HnswlibSubtitleIndexReader(config, series)
        else:
            raise Exception(f"Unknown index format: {config.args.index_format}")

        self.text_extractor_model = "small.en"
        self.segment_duration = 30
        self.segment_count = 10

        self.extracted_text_dir = self.series.dot_dir / "extracted-text"
        self.extracted_text_dir.mkdir(exist_ok=True)

        self.cache = TextSegmentCache(config, self.extracted_text_dir,
                                      self.segment_duration, self.segment_count)

    def match(self, paths):
        files = list(self._collect_files(paths))
        if not files:
            return []

        text_segments_map = self._ensure_text_segments(files)

        query_results: Dict[
            Path, List[Tuple[Tuple[float, int] or float, str, str]]
        ] = {}
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
            info = guessit(file.name)
            actual = info.get("season"), info.get("episode") if info else None
            results.append(MatchResult(file, matches, actual))

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
            initargs=(self.text_extractor_model, self.segment_duration, self.segment_count,),
            mp_context=ctx,
        ) as executor:
            results = executor.map(_extract_text_segments_worker, missing_files)

            text_segments = {file: result
                             for file, result in zip(missing_files, results)}

            self.cache.write_cache(text_segments)
            return text_segments

def match_debug(config: Configuration):
    extract_file = Path(config.args.extract_file)
    series = get_series(extract_file)
    index = ChromaSubtitleIndex(config, series)

    info = guessit(str(extract_file))
    if not info:
        raise ValueError("Unable to guess info from file (not a labeled episode?)")

    with open(config.args.extract_file, "r") as f:
        extracts = json.load(f)

    combined = {}
    for offset, extracted_text in extracts:
        start_ms = offset * 30 * 1000
        query = {"$and": [
        #     {"start_ms": start_ms},
             {"season_number": info.get("season")},
             {"episode_number": info.get("episode")}
        ]}
        #query = {"start_ms": start_ms}
        console.print(f"Query: {query}")
        result = index.intervals.get(where=query)
        combined[start_ms] = (extracted_text, result["documents"], result["metadatas"])
        break


    console.print(json.dumps(combined, indent=2))

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

    def get_cached_segments(self, files: List[Path]) -> Tuple[Dict[Path, List[Tuple[int, str]]], List[Path]]:
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
