import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, \
    as_completed
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Self, Optional, Type

from loguru import logger
from rich.console import Console
from rich.progress import Progress

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.embedding_model import DEFAULT_MODEL_NAME
from mkv_episode_matcher.embedding_worker import \
    _init_embeddings_extractor_worker, _extract_embeddings_from_transcription
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.transcription_worker import \
    _init_transcription_worker, _extract_text_segments_worker, \
    transcription_file_name
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

PathDict: Type = dict[Path, Path]

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
    transcription: Optional[Path]
    embeddings: Optional[Path]
    matches: list[Match]
    known_episode: Optional[EpisodeKey]

class IndexedEpisodeMatcher:
    def __init__(self, config: Configuration, series: Series):
        self.config = config
        self.series = series

        self.index_cls = config.args.index_type.reader_type

        self.text_extractor_model = "small.en"
        self.segment_duration = 30
        self.segment_count = 10

    def match(self, paths) -> List[MatchResult]:
        before = time.time()
        video_files = list(self._collect_files(paths))
        after = time.time()
        logger.info(f"Collected {len(video_files)} video files in {after - before:.2f}s")

        if not video_files:
            return []

        with Progress() as progress:
            match_progress = progress.add_task(f"Matching: {self.series.name}", total=100.0)

            transcriptions = self.get_transcriptions(progress, video_files)
            progress.update(match_progress, advance=33.3333)

            embeddings = self.get_embeddings(progress, transcriptions)
            progress.update(match_progress, advance=33.3333)

            query_results = self.get_query_results(progress, embeddings)
            progress.update(match_progress, advance=33.3333)

            match_results = [
                MatchResult(file,
                            transcription := transcriptions[file],
                            embedding := embeddings[transcription],
                            query_results[embedding],
                            EpisodeKey.from_path(file))
                for file in video_files
            ]


        return match_results

    def get_transcriptions(self, progress: Progress, files: list[Path]) -> PathDict:
        if not files:
            return {}

        transcription_progress = progress.add_task(f"Transcribing videos for: {self.series.name}",
                                                   total=len(files))

        with ThreadPoolExecutor(max_workers=10) as executor:
            chunks = chunked(files, 5)
            future_to_chunks = {executor.submit(self.get_cached_transcriptions, chunk): chunk
                                for chunk in chunks}
            cached = {}
            for future in as_completed(future_to_chunks):
                result = future.result()
                progress.update(transcription_progress, advance=len(result))
                cached.update(result)

        missing_files = files - cached.keys()

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=4,
            initializer=_init_transcription_worker,
            initargs=(self.series, self.config.args.transcriber,
                      self.text_extractor_model,
                      self.segment_duration, self.segment_count),
            mp_context=ctx,
        ) as executor:
            chunks = chunked(missing_files, 3)
            futures = (executor.submit(_extract_text_segments_worker,
                                       chunk) for chunk in chunks)

            transcribed = {}
            for future in as_completed(futures):
                result = future.result()
                transcribed.update(result)
                progress.update(transcription_progress, advance=len(result))

        progress.remove_task(transcription_progress)
        return cached | transcribed

    def get_embeddings(self, progress: Progress,
        transcriptions: PathDict) -> PathDict:
        embeddings_progress = progress.add_task(
            f"Extracting embeddings: {self.series.name}",
            total=len(transcriptions))

        with ProcessPoolExecutor(max_workers=10,
                                 initializer=_init_embeddings_extractor_worker,
                                 initargs=(self.config, self.series,
                                           DEFAULT_MODEL_NAME,
                                           self.segment_duration,
                                           self.segment_count)) as executor:
            chunks = chunked(transcriptions.values(), 5)
            futures = (executor.submit(_extract_embeddings_from_transcription,
                                       chunk) for chunk in chunks)

            embeddings = {}
            for future in as_completed(futures):
                result = future.result()
                progress.update(embeddings_progress, advance=len(result))
                embeddings.update(result)

        progress.remove_task(embeddings_progress)
        return embeddings

    def get_query_results(self, progress: Progress,
        embeddings: PathDict) -> dict[Path, list[Match]]:
        query_progress = progress.add_task(f"Querying for: {self.series.name}",
                                           total=len(embeddings))

        load_index_progress = progress.add_task(f"Loading index for: {self.series.name}", total=1)
        index = self.index_cls(self.config, self.series)
        progress.update(load_index_progress, advance=1)
        progress.remove_task(load_index_progress)

        query_results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {
                executor.submit(index.query_intervals,
                                embedding): embedding
                for embedding in embeddings.values()
            }
            for future in as_completed(future_to_file):
                progress.update(query_progress, advance=1)
                file = future_to_file[future]
                query_results[file] = future.result()

        progress.remove_task(query_progress)
        return query_results

    @staticmethod
    def _collect_files(paths: Iterable[Path]) -> Iterable[Path]:
        for path in paths:
            if path.is_file():
                yield path
            else:
                yield from (candidate for candidate in path.rglob("*.mkv")
                            if candidate.is_file())

    def get_cached_transcriptions(self, video_files: list[Path]) -> PathDict:
        if self.config.args.no_transcription_cache:
            return {}

        dir = self.series.get_transcription_text_dir(self.segment_duration,
                                                     self.segment_count)
        cached_transcripts = map(lambda file: dir / transcription_file_name(file),
                             video_files)
        return {video: transcript
                for video, transcript in zip(video_files, cached_transcripts)
                if transcript.exists()}


def chunked(iterable: Iterable, n: int) -> Iterable:
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk

