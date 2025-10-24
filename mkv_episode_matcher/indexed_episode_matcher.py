import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from guessit import guessit
from loguru import logger
from rich.console import Console

from mkv_episode_matcher.annoy_subtitle_index import AnnoySubtitleIndexReader
from mkv_episode_matcher.chroma_subtitle_index import (
    ChromaSubtitleIndex,
    ChromaSubtitleIndexReader,
)
from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.hnswlib_subtitle_index import HnswlibSubtitleIndexReader
from mkv_episode_matcher.series import Series, get_series
from mkv_episode_matcher.text_segment_extractor import TextSegmentExtractor

console = Console()

_PROCESS_TEXT_EXTRACTOR = None


def _init_text_extractor_worker(model_name: str):
    """Initializer for the process pool so Whisper loads only in child processes."""
    global _PROCESS_TEXT_EXTRACTOR
    if _PROCESS_TEXT_EXTRACTOR is None:
        _PROCESS_TEXT_EXTRACTOR = TextSegmentExtractor(model_name)


def _extract_text_segments_worker(
    file_path_str: str,
    duration: int,
    count: int,
    cache_dir_str: str,
):
    """Extract text segments for a single file inside a worker process."""
    file_path = Path(file_path_str)
    cache_dir = Path(cache_dir_str)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / file_path.with_suffix(".json").name

    if cache_file.exists():
        with open(cache_file, "r") as json_in:
            return json.load(json_in)

    segments = _PROCESS_TEXT_EXTRACTOR.get_text_segments(file_path, duration, count)
    with open(cache_file, "w") as json_out:
        json.dump(segments, json_out)
    return segments

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

    def match(self, paths):
        files = list(self._collect_files(paths))
        if not files:
            return []

        for file in files:
            logger.info(f"Processing file: {file}")

        text_segments_map = self._ensure_text_segments(files)

        query_workers = self._determine_thread_workers(len(files))
        query_results: Dict[
            Path, List[Tuple[Tuple[float, int] or float, str, str]]
        ] = {}
        with ThreadPoolExecutor(max_workers=query_workers) as executor:
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

    def match_intervals(self, file):
        """
        match against the extracted intervals, targeting the query to only
        subtitle segments that start at the same time as the extracted text
        :param file: the file to match
        :return: (distance, season, episode) tuples
        """
        text_segments = self.extract_text_segments(file)
        return self.index.query_intervals(text_segments)

    def extract_text_segments(self, file) -> List[Tuple[int, str]]:
        return self._ensure_text_segments([file])[file]

    def _collect_files(self, paths: Iterable[Path]) -> Iterable[Path]:
        for path in paths:
            if path.is_file():
                yield path
            else:
                yield from (
                    candidate for candidate in path.rglob("*.mkv") if candidate.is_file()
                )

    def _ensure_text_segments(self, files: List[Path]) -> Dict[Path, List[Tuple[int, str]]]:
        if not files:
            return {}

        cache_dir = self._ensure_cache_dir()
        text_segments: Dict[Path, List[Tuple[int, str]]] = {}
        missing_files: List[Path] = []

        for file in files:
            cache_file = cache_dir / file.with_suffix(".json").name
            if cache_file.exists():
                with open(cache_file, "r") as json_in:
                    text_segments[file] = json.load(json_in)
            else:
                missing_files.append(file)

        if missing_files:
            process_workers = self._determine_process_workers(len(missing_files))
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=process_workers,
                initializer=_init_text_extractor_worker,
                initargs=(self.text_extractor_model,),
                mp_context=ctx,
            ) as executor:
                future_to_file = {
                    executor.submit(
                        _extract_text_segments_worker,
                        str(file),
                        self.segment_duration,
                        self.segment_count,
                        str(cache_dir),
                    ): file
                    for file in missing_files
                }
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    text_segments[file] = future.result()

        return text_segments

    def _ensure_cache_dir(self) -> Path:
        cache_dir = self.extracted_text_dir / f"dur{self.segment_duration}s_count{self.segment_count}"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def _determine_process_workers(self, task_count: int) -> int:
        cpu_count = os.cpu_count() or 1
        return max(1, min(task_count, cpu_count))

    def _determine_thread_workers(self, task_count: int) -> int:
        cpu_count = os.cpu_count() or 1
        return max(1, min(task_count, cpu_count))

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
