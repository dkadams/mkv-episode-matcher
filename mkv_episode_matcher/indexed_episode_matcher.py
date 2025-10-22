import json
import sys
from pathlib import Path
from typing import List, Tuple

from guessit import guessit
from loguru import logger
from rich.console import Console
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import episode_str
from mkv_episode_matcher.series import Series, get_series
from mkv_episode_matcher.chroma_subtitle_index import ChromaSubtitleIndex
from mkv_episode_matcher.text_segment_extractor import TextSegmentExtractor

console = Console()

class IndexedEpisodeMatcher:
    def __init__(self, config: Configuration, series: Series):
        self.config = config
        self.series = series
        self.index = ChromaSubtitleIndex(config, series)
        self.text_extractor = TextSegmentExtractor("small.en")

        self.extracted_text_dir = self.series.dot_dir / "extracted-text"
        self.extracted_text_dir.mkdir(exist_ok=True)

    def match(self, paths):
        table = Table(title=f"Matches for '{self.series.name}'")
        table.add_column("Filename")
        table.add_column("Episode Id")
        table.add_column("# Matches", style="bold")
        table.add_column("#1", style="magenta")
        table.add_column("#2")
        table.add_column("#3")
        table.add_column("#4")
        table.add_column("#5")

        correct = 0
        known_episode_count = 0
        files = [path if path.is_file() else path.rglob('**/*.mkv')
                 for path in paths]
        for file in files:
            logger.info(f"Processing file: {file}")

            #matches = self.match_file_full(file)
            matches = self.match_intervals(file)
            formatted_matches = [f"{episode_str(m[1], m[2])} - {m[0]}"
                                 for m in matches]

            info = guessit(file.name)
            if info:
                known_episode_count += 1
                actual_episode = info.get("season"), info.get("episode")
                actual = episode_str(*actual_episode)
                if len(matches) > 0 and actual_episode == matches[0][1:]:
                    correct += 1
            else:
                actual = "-"

            if len(formatted_matches) < 5:
                formatted_matches.extend(["-"] * (5 - len(formatted_matches)))
            table.add_row(file.name,
                          actual,
                          str(len(formatted_matches)),
                          formatted_matches[0],
                          formatted_matches[1],
                          formatted_matches[2],
                          formatted_matches[3],
                          formatted_matches[4])

        console.print(table)
        if known_episode_count > 0:
            console.print(f"Correct: {correct}/{known_episode_count} ({correct/known_episode_count*100:.2f}%)")

    def match_file_full(self, file):
        """
        match against all the extracted text as a single query
        :param file: the file to match
        :return: (distance, season, episode) tuples
        """
        text_segments = self.extract_text_segments(file)
        text = " ".join([text for _, text in text_segments])
        return self.index.query_full_text(text)

    def match_intervals(self, file):
        """
        match against the extracted intervals, targeting the query to only
        subtitle segments that start at the same time as the extracted text
        :param file: the file to match
        :return: (distance, season, episode) tuples
        """
        text_segments = self.extract_text_segments(file)

        text_intervals = [(index * 30 * 1000, text)
                          for index, text in text_segments]
        return self.index.query_intervals(text_intervals)


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
