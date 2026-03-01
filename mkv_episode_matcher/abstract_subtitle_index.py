from abc import ABC, abstractmethod
from concurrent.futures import Executor
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import pysubs2
from loguru import logger
from rich.console import Console
from rich.progress import Progress

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.embedding_model import SentenceTransformerModel
from mkv_episode_matcher.subtitle_embeddings_extractor import SubtitleEmbeddingsExtractor
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.series import Series, get_specified_episodes
from mkv_episode_matcher.subtitle_fixed_intervalizer import \
    SubtitleFixedIntervalizer
from mkv_episode_matcher.windowing import (
    DEFAULT_SUPPORT_OFFSET_PENALTY,
    DEFAULT_SUPPORT_WINDOW_BONUS,
    make_window_config,
    resolve_low_info_cue_ratio,
    resolve_low_info_filter,
    resolve_low_info_min_words,
    resolve_max_results_per_query,
    resolve_subtitle_overlap_seconds,
    resolve_window_expansion_mode,
    resolve_window_neighbor_radius,
)

console = Console()

class AbstractSubtitleIndex(ABC):
    """Base class for indexes that manually manage embeddings."""

    def __init__(self, config: Configuration, series: Series):
        self.config = config
        self.series = series

        self.embedding_model = SentenceTransformerModel()
        self.interval_seconds = self.series.segment_duration
        self.subtitle_overlap_seconds = resolve_subtitle_overlap_seconds(
            self.config.args, self.series
        )
        self.window_config = make_window_config(
            self.interval_seconds, self.subtitle_overlap_seconds
        )
        self.window_profile_key = self.window_config.profile_key
        self.window_expansion_mode = resolve_window_expansion_mode(config.args, series)
        self.window_neighbor_radius = resolve_window_neighbor_radius(config.args, series)
        self.low_info_filter = resolve_low_info_filter(config.args, series)
        self.low_info_min_words = resolve_low_info_min_words(config.args, series)
        self.low_info_cue_ratio = resolve_low_info_cue_ratio(config.args, series)
        self.max_results_per_query = resolve_max_results_per_query(config.args, series)
        self.support_window_bonus = DEFAULT_SUPPORT_WINDOW_BONUS
        self.support_offset_penalty = DEFAULT_SUPPORT_OFFSET_PENALTY

        # Embeddings are shared across multiple index types, so they are stored
        # in the series index directory.
        self.profile_dir = series.index_dir / self.window_profile_key
        embeddings_dir = self.profile_dir / "embeddings"

        self.interval_subs_dir = embeddings_dir / "interval-subs"
        self.interval_subs_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = embeddings_dir / self.embedding_model.dir_name()
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def index_dir(self):
        pass

class AbstractSubtitleIndexWriter(AbstractSubtitleIndex):


    def __init__(self, config: Configuration, series: Series):
        super().__init__(config, series)

        self.sub_intervalizer = SubtitleFixedIntervalizer(config, series,
                                                          self.interval_seconds,
                                                          self.subtitle_overlap_seconds)
        self.embedding_extractor = SubtitleEmbeddingsExtractor(config, series,
                                                               self.embedding_model,
                                                               self.model_dir)

    @abstractmethod
    def build_interval_index(self, interval_dir: Path):
        pass

    def index_series(self):
        logger.info(f"SubtitleEmbeddingProcessor: Indexing: {self.series.name} ({self.series.dir})")

        with Progress() as progress, ThreadPoolExecutor(max_workers=10) as executor:
            if self.config.args.rebuild:
                self.delete_embeddings_files(progress)

            # annoy indexes must be rebuilt from scratch. hnswlib indexes can
            # be incrementally updated, but that hasn't been implemented yet.
            if self.index_dir.exists():
                self.delete_index_files(progress)

            episode_keys = {EpisodeKey(ep.season_number, ep.episode_number)
                            for ep in get_specified_episodes(self.config, self.series)}

            self.intervalize_subs(episode_keys, executor, progress)
            self.extract_embeddings(episode_keys, executor, progress)

            self.build_index(executor, progress)

    def delete_embeddings_files(self, progress: Progress):
        logger.info(f"Removing embeddings for: {self.series.name}, "
                    f"model: {self.embedding_model}. "
                    f"Deleting: {self.interval_subs_dir} and {self.model_dir}")

        def files(path: Path, ext: str) -> list[Path]:
            return [file for file in path.rglob("*")
                    if file.suffix == ext and file.is_file()]
        def dirs(path: Path) -> list[Path]:
            return [dir for dir in path.iterdir()
                    if dir.is_dir()]

        files_to_delete = (files(self.interval_subs_dir, ".srt")
                           + files(self.model_dir, ".npy"))
        dirs_to_delete = dirs(self.model_dir)

        delete_progress = progress.add_task(
            f"Removing embeddings for: {self.series.name}",
            total=len(files_to_delete) + len(dirs_to_delete))

        for embedding_file in files_to_delete:
            embedding_file.unlink()
            progress.update(delete_progress, advance=1)
        for interval_dir in dirs_to_delete:
            interval_dir.rmdir()
            progress.update(delete_progress, advance=1)

        progress.remove_task(delete_progress)

    def delete_index_files(self, progress: Progress):
        logger.info(f"Removing index for: {self.series.name}, "
                    f"deleting: {self.index_dir}")
        index_files = [file for file in self.index_dir.iterdir()
                       if file.suffix in {".idx", ".json"}
                       and file.is_file()]
        delete_progress = progress.add_task(
            f"Removing index for: {self.series.name}",
            total=len(index_files))
        for index_file in index_files:
            index_file.unlink()
            progress.update(delete_progress, advance=1)
        progress.remove_task(delete_progress)

    def intervalize_subs(self, episode_keys: set[EpisodeKey],
        executor: Executor, progress: Progress):
        logger.info(f"Intervalizing subs for: {self.series.name}")

        existing_srts = self.interval_subs_dir.rglob("*.srt")
        existing_keys = {
            key for file in existing_srts
            if (key := EpisodeKey.from_srt_path(file))
        }
        missing_subs = episode_keys - existing_keys

        srt_files = list(self.series.subtitles_dir.rglob("*.srt"))
        eps_and_subs_to_process = [(key, file) for file in srt_files
                                   if (key := EpisodeKey.from_srt_path(file)) in missing_subs]
        if not eps_and_subs_to_process:
          logger.info(f"No subs to intervalize for series: {self.series.name}")
          return

        extract_progress = progress.add_task(
            f"Intervalizing subs for {self.series.name} "
            f"({self.series.dir})", total=len(eps_and_subs_to_process))

        if not self.interval_subs_dir.exists():
            self.interval_subs_dir.mkdir(parents=True, exist_ok=True)

        def extract(episode_key: EpisodeKey, input: Path):
            output = self.interval_subs_dir / f"{episode_key}.srt"
            logger.info(f"Intervalizing subs for: {self.series.name}, "
                        f"episode: {episode_key} "
                        f"from: {input} to {output}")
            self.sub_intervalizer.execute(episode_key, input, output)
            progress.update(extract_progress, advance=1)
            logger.info(f"Intervalized subs for: {self.series.name}, "
                        f"episode: {episode_key} "
                        f"from: {input} to {output}")

        missing_episodes, missing_subs = zip(*eps_and_subs_to_process)
        list(executor.map(extract, missing_episodes, missing_subs))
        progress.remove_task(extract_progress)
        logger.info(f"Intervalized subs for: {self.series.name}")

    def extract_embeddings(self, episode_keys: set[EpisodeKey],
        executor: Executor, progress: Progress):
        logger.info(f"Extracting embeddings for: {self.series.name}")

        subtitle_files = list(self.series.subtitles_dir.rglob("*.srt"))

        extract_progress = progress.add_task(
            f"Extracting embeddings for {self.series.name} "
            f"({self.series.dir})", total=len(subtitle_files))

        interval_subs = [(episode_key, path)
                         for path in self.interval_subs_dir.rglob("*.srt")
                         if (episode_key := EpisodeKey.from_srt_path(path)) in episode_keys]
        logger.info(f"Extracting embeddings for subs: {interval_subs}")

        def extract_embeddings(interval_index: int):
            self.embedding_extractor.execute(interval_index, interval_subs)
            progress.update(extract_progress, advance=1)

        interval_count = max(len(pysubs2.load(str(path), format_="srt"))
                             for _, path in interval_subs)
        intervals = range(interval_count)

        list(executor.map(extract_embeddings, intervals))
        progress.remove_task(extract_progress)
        logger.info(f"Extracted embeddings for: {self.series.name}")

    def build_index(self, executor: Executor, progress: Progress):
        logger.info(f"Building indexes for: {self.series.name}, "
                    f"model: {self.embedding_model}")

        embeddings_files = [file for file in self.model_dir.iterdir()
                         if file.is_file() and file.suffix == ".npy"]
        if not embeddings_files:
            console.print(f"[red]No embeddings files found for series: {self.series.name}.")

        index_progress = progress.add_task(
            f"Indexing {self.series.name} ({self.series.dir})",
            total=len(embeddings_files))

        def index_interval(file: Path):
            logger.info(f"Indexing {file}")

            self.build_interval_index(file)
            progress.update(index_progress, advance=1)

        list(executor.map(index_interval, embeddings_files))
        progress.remove_task(index_progress)
