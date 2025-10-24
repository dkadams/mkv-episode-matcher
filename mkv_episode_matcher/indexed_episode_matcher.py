import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

        self.text_extractor = TextSegmentExtractor("small.en")

        self.extracted_text_dir = self.series.dot_dir / "extracted-text"
        self.extracted_text_dir.mkdir(exist_ok=True)

    def match(self, paths):
        files = [path if path.is_file() else path.rglob('**/*.mkv')
                 for path in paths]

        results = []
        for file in files:
            logger.info(f"Processing file: {file}")

            matches = self.match_intervals(file)

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
        duration = 30
        count = 10

        cache_dir = self.extracted_text_dir / f"dur{duration}s_count{count}"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / file.with_suffix(".json").name

        if cache_file.exists():
            with open(cache_file, "r") as json_in:
                text_segments = json.load(json_in)
        else:
            text_segments = self.text_extractor.get_text_segments(file, duration, count)
            with open(cache_file, "w") as json_out:
                json_out.write(json.dumps(text_segments))

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
