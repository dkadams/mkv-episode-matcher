import hashlib
import json
import math
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, \
    as_completed
from dataclasses import dataclass, asdict
from itertools import islice
from pathlib import Path
from typing import Iterable, Self, Optional, Type

from loguru import logger
from more_itertools import unique
from rich.console import Console
from rich.progress import Progress

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.embedding_model import DEFAULT_MODEL_NAME
from mkv_episode_matcher.embedding_worker import \
    _init_embeddings_extractor_worker, _extract_embeddings_from_transcription
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcribers import SubprocessTranscriber
from mkv_episode_matcher.transcription_worker import \
    _init_transcription_worker, _extract_text_segments_worker
from mkv_episode_matcher.video_helper import get_video_duration_seconds

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

@dataclass(frozen=True, eq=True, order=True)
class VideoInfo:
    file: Path
    path_hash: str
    byte_count: int
    minutes: float
    segments: int

    def asdict(self) -> dict:
        result = asdict(self)
        # Convert non-JSON Encodable Path to str
        result["file"] = str(result["file"])
        return result

class IndexedEpisodeMatcher:
    def __init__(self, config: Configuration, series: Series):
        self.config = config
        self.series = series

        self.index_cls = config.args.index_type.reader_type

        self.text_extractor_model = "small.en"

    def match(self, paths) -> list[MatchResult]:

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

    def get_transcriptions(self, progress: Progress, videos: list[Path]) -> PathDict:
        if not videos:
            return {}

        read_cache_task = progress.add_task(f"Reading video info cache",
                                            total=1)
        video_cache_path = self.series.ensure_segments_dir() / "video-info-cache.json"
        video_info_dict = {}
        if video_cache_path.exists():
            try:
                with video_cache_path.open("r") as cache:
                    video_info_dict = json.load(cache)
            except json.JSONDecodeError:
                logger.info(f"Video info cache is corrupt, recreating")

        progress.update(read_cache_task, advance=1)
        progress.remove_task(read_cache_task)

        segments_per_minute = self.config.args.segments_per_minute
        with ThreadPoolExecutor(max_workers=10) as threads:
            def get_video_info(path: Path) -> tuple[Path, VideoInfo]:
                logger.info(f"Getting video info for: {path}")
                full_path = path.resolve()
                # We only use the full_path for hashing, so don't return it
                path_hash = hashlib.sha256(str(full_path).encode()).hexdigest()
                byte_count = full_path.stat().st_size
                video_dict = video_info_dict.get(path_hash)
                if video_dict and video_dict["byte_count"] == byte_count:
                    logger.info(f"Found cached video info for: {path}")
                    return path, VideoInfo(full_path, path_hash, byte_count,
                                           video_dict["minutes"],
                                           video_dict["segments"])

                logger.info(f"Retrieving duration for: {path} to create video info")
                # This executes ffmpeg, so we only call it if we don't have a
                # cached entry.
                seconds = get_video_duration_seconds(full_path)
                minutes = seconds / 60.0
                segments = math.ceil(seconds / self.series.segment_duration)
                return path, VideoInfo(full_path, path_hash, byte_count,
                                       minutes, segments)

            def get_cached_segment_count(path: Path) -> tuple[Path, int]:
                logger.info(f"Checking for cached transcription for: {path}")
                transcript = self.series.transcription_file(path)
                if not transcript.exists():
                    return path, 0
                logger.info(f"Found cached transcription for: {path}")
                with transcript.open("r") as f:
                    transcript = json.load(f)
                    return path, len(transcript)

            video_info_task = progress.add_task(f"Retrieving video info",
                                                total=len(videos))
            transcript_cache_task = progress.add_task(f"Checking transcription cache",
                                                      total=len(videos))

            video_info_by_path = {}
            segment_count_by_path = {}

            futures = [threads.submit(get_video_info, path)
                       for path in videos]
            if not self.config.args.no_transcription_cache:
                futures.extend(threads.submit(get_cached_segment_count, path)
                               for path in videos)
            logger.info(f"Submitted video info and transcript cache tasks: {len(futures)}")
            for future in as_completed(futures):
                logger.info(f"Completed video info/transcript cache task: {future}")
                match future.result():
                    case (Path() as path, VideoInfo() as video_info):
                        video_info_by_path[path] = video_info
                        progress.update(video_info_task, advance=1)

                    case (Path() as path, cached_segment_count):
                        segment_count_by_path[path] = cached_segment_count
                        progress.update(transcript_cache_task, advance=1)
            logger.info(f"Completed video info/transcript cache tasks")
        progress.remove_task(video_info_task)
        progress.remove_task(transcript_cache_task)

        write_video_info_task = progress.add_task(f"Writing video info cache",
                                                  total=1)
        video_info_dict = {video_info.path_hash: video_info.asdict()
                           for video_info in video_info_by_path.values()}
        with video_cache_path.open("w") as cache:
            json.dump(video_info_dict, cache)
        progress.update(write_video_info_task, advance=1)
        progress.remove_task(write_video_info_task)

        segment_indexes = self.get_segment_selection(video_info_by_path.values())

        segments_to_transcribe_count = 0
        segments_to_transcribe = {}
        cached_transcripts = {}
        for path in videos:
            video_info = video_info_by_path[path]
            segment_target = math.ceil(video_info.minutes * segments_per_minute)
            cached_segments = segment_count_by_path.get(path, 0)
            segments_needed = segment_target - cached_segments
            if segments_needed > 0:
                start = cached_segments
                end = cached_segments + segments_needed
                segments_to_transcribe[path] = segment_indexes[start:end]
                segments_to_transcribe_count += segments_needed
            else:
                cached_transcripts[path] = self.series.transcription_file(path)

        transcription_progress = progress.add_task(f"Transcribing segments",
                                                   total=len(segments_to_transcribe))

        transcriber_type = self.config.args.transcriber
        # If the transcriber is a subprocess, we don't get any benefit from
        # spawning a separate python process for it.
        if issubclass(transcriber_type, SubprocessTranscriber):
            executor = ThreadPoolExecutor(
                max_workers=10,
                initializer=_init_transcription_worker,
                initargs=(self.config, self.series, transcriber_type,
                          self.text_extractor_model))
        else:
            ctx = multiprocessing.get_context("spawn")
            executor = ProcessPoolExecutor(
                max_workers=8,
                initializer=_init_transcription_worker,
                initargs=(self.config, self.series, transcriber_type,
                          self.text_extractor_model),
                mp_context=ctx)

        with executor as transcribers:
            chunks = chunked(segments_to_transcribe.items(), 3)
            futures = (transcribers.submit(_extract_text_segments_worker,
                                           chunk) for chunk in chunks)

            transcribed = {}
            for future in as_completed(futures):
                result = future.result()
                transcribed.update(result)
                progress.update(transcription_progress, advance=len(result))

        progress.remove_task(transcription_progress)
        return cached_transcripts | transcribed

    def get_segment_selection(self, video_infos: Iterable[VideoInfo]) -> list[int]:
        """
        Pre-compute a random selection of segments to extract from each video.

        The implementation attempts to ensure even sampling when video duration
        varies. As the duration grows, additional segments are extracted only
        from additional period of runtime.

        For example, when extracting segments from 22, 33, and 44 minute
        videos, the same segments will be selected from the first 22 minutes
        for all files. The 33 and 44 minute videos will have the same segments
        extracted from minutes 23-33. The additional segments for the 44 minute
        episode will only occur during minutes 34-44.

        :param video_info_by_path:
        :return: list of segment indexes to extract
        """
        segments_per_minute = self.config.args.segments_per_minute

        file_durations = sorted(unique((video_info.minutes, video_info.segments)
                                       for video_info in video_infos))
        random.seed(self.series.random_seed)
        segment_indexes = []
        last_file_segments = 0
        for file_minutes, file_segments in file_durations:
            target_segments = math.ceil(file_minutes * segments_per_minute)
            segments_needed = target_segments - len(segment_indexes)
            if segments_needed > 0:
                # As the duration goes up, we want to avoid sampling from the
                # range we've already sampled. So we keep moving the start range
                # up.
                segment_range = range(last_file_segments, file_segments)
                # We need to avoid last_file_segments == file_segments, because
                # range() would return None,
                if last_file_segments != file_segments:
                    last_file_segments = file_segments

                selected_segments = random.sample(segment_range,
                                                  segments_needed)
                segment_indexes.extend(selected_segments)

        logger.info(
            f"Selected {len(segment_indexes)} segments for transcription: {segment_indexes}")
        return segment_indexes

    def get_embeddings(self, progress: Progress,
        transcriptions: PathDict) -> PathDict:
        embeddings_progress = progress.add_task(
            f"Extracting embeddings: {self.series.name}",
            total=len(transcriptions))

        with ProcessPoolExecutor(max_workers=10,
                                 initializer=_init_embeddings_extractor_worker,
                                 initargs=(self.config, self.series,
                                           DEFAULT_MODEL_NAME)) as embeddings_extractor:
            chunks = chunked(transcriptions.values(), 5)
            futures = (embeddings_extractor.submit(_extract_embeddings_from_transcription,
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

def chunked(iterable: Iterable, n: int) -> Iterable:
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk

