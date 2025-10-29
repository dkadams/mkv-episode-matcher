import math
from concurrent.futures import Executor
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pysubs2
from loguru import logger
from pysubs2 import SSAFile
from rich.console import Console
from rich.progress import Progress

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.embedding_model import EmbeddingModel
from mkv_episode_matcher.series import Series, get_specified_episodes

console = Console()

class SubtitleIndexHelper:
    """Shared helper for extracting and caching subtitle embeddings."""

    def __init__(
        self,
        config: Configuration,
        series: Series,
        index_dir: Path,
        embedding_model: EmbeddingModel,
        interval_seconds: int = 30,
    ):
        self.config = config
        self.series = series
        # This is the index directory for the specific index type:
        # "{series.index_dir}/hnswlib.index" or "{series.index_dir}/annoy.index".
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self.interval_seconds = interval_seconds
        self.interval_ms = interval_seconds * 1000

        # Embeddings are shared across multiple index types, so they are stored
        # in the series index directory.
        embeddings_dir = series.index_dir / "embeddings"

        self.text_dir = embeddings_dir / "text"
        self.text_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = embeddings_dir / self.embedding_model.dir_name()
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def index_series(self, build_interval_index: Callable[[Path], None]):
        logger.info(f"SubtitleEmbeddingProcessor: Indexing: {self.series.name} ({self.series.dir})")

        with Progress() as progress, ThreadPoolExecutor(max_workers=10) as executor:
            if self.config.args.rebuild:
                self.delete_embeddings_files(progress)

            # annoy indexes must be rebuilt from scratch. hnswlib indexes can
            # be incrementally updated, but that hasn't been implemented yet.
            if self.index_dir.exists():
                self.delete_index_files(progress)

            self.extract_embeddings(executor, progress)

            self.build_index(executor, progress, build_interval_index)

    def delete_embeddings_files(self, progress: Progress):
        logger.info(f"Removing embeddings for: {self.series.name}, "
                    f"model: {self.embedding_model}. "
                    f"Deleting: {self.text_dir} and {self.model_dir}")

        def files(path: Path, ext: str) -> list[Path]:
            return [file for file in path.rglob("*")
                    if file.suffix == ext and file.is_file()]
        def dirs(path: Path) -> list[Path]:
            return [dir for dir in path.iterdir()
                    if dir.is_dir()]

        files_to_delete = (files(self.text_dir, ".txt")
                           + files(self.model_dir, ".npy"))
        dirs_to_delete = dirs(self.text_dir) + dirs(self.model_dir)

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

    def extract_embeddings(self, executor: Executor, progress: Progress):
        logger.info(f"Extracting embeddings for: {self.series.name}")

        subtitle_files = list(self.series.subtitles_dir.rglob("*.srt"))
        episodes = {EpisodeKey(ep.season_number, ep.episode_number)
                    for ep in get_specified_episodes(self.config, self.series)}
        logger.info(f"Extracting embeddings for Episodes: {episodes}")

        extract_progress = progress.add_task(
            f"Extracting embeddings for {self.series.name} "
            f"({self.series.dir})", total=len(subtitle_files))

        def extract_embeddings(file):
            logger.info(f"Extracting from: {file}")
            episode = EpisodeKey.from_path(file)
            logger.info(f"Identified: {file} as episode: {episode}")
            if episode in episodes:
                logger.info(f"Extracting: {file} as episode: {episode}")
                self.extract(file, episode)
            progress.update(extract_progress, advance=1)

        list(executor.map(extract_embeddings, subtitle_files))
        progress.remove_task(extract_progress)
        logger.info(f"Extracted embeddings for: {self.series.name}")

    def extract(self, path: Path, episode: EpisodeKey):
        logger.info(f"Extracting embeddings for episode: {self.series.name} episode: {episode}")

        sub_file = pysubs2.load(str(path), format_="srt")
        metadata = self.series.get_episode_detail(episode)
        if not metadata:
            logger.info(f"No metadata found for episode: {episode}")
            return

        interval_count = math.ceil(metadata["runtime"] * 60 / self.interval_seconds)
        intervals = (
            (i, range(i * self.interval_ms, (i + 1) * self.interval_ms))
            for i in range(interval_count)
        )

        for index, interval in intervals:
            text_file = self._get_text_dir(index) / f"{episode}.txt"
            embeddings_file = self._get_model_dir(index) / f"{episode}.npy"
            if text_file.exists() and embeddings_file.exists():
                logger.info(f"Skipping interval: {interval} for episode: {episode}: .txt & .npy already exist. ")
                continue

            interval_text = self.extract_interval_text(interval, sub_file,
                                                       text_file)
            if interval_text is None:
                continue

            if embeddings_file.exists():
                logger.info(f"Skipping interval: {interval} for episode: {episode}: .npy already exist. ")
                continue

            embeddings = self.embedding_model.encode_document(interval_text)
            np.save(embeddings_file, embeddings.astype(np.float32))

        logger.info(f"Extracted embeddings for episode: {self.series.name} episode: {episode}")

    @staticmethod
    def extract_interval_text(interval: range, sub_file: SSAFile,
        text_file: Path) -> str:
        if text_file.exists():
            with open(text_file, "r") as text_in:
                interval_text = text_in.read()
        else:
            subs = [
                # There are no EOL characters in the transcribed text. So we
                # don't want any in the subtitle text, either.
                sub.plaintext.replace("\n", " ")
                for sub in sub_file
                if sub.start in interval or sub.end in interval
            ]
            if not subs:
                interval_text = None

            interval_text = " ".join(subs)
            with open(text_file, "w") as text_out:
                text_out.write(interval_text)
        return interval_text

    def build_index(self, executor: Executor, progress: Progress,
            build_interval_index: Callable[[Path], None]):
        logger.info(f"Building indexes for: {self.series.name}, "
                    f"model: {self.embedding_model}")

        interval_dirs = [dir for dir in self.model_dir.iterdir()
                         if dir.is_dir()]
        if not interval_dirs:
            console.print(f"[red]No embedding dirs found for series: {self.series.name}.")

        index_progress = progress.add_task(
            f"Indexing {self.series.name} ({self.series.dir})",
            total=len(interval_dirs))

        def index_interval(dir):
            logger.info(f"Indexing {dir}")
            build_interval_index(dir)
            progress.update(index_progress, advance=1)

        list(executor.map(index_interval, interval_dirs))
        progress.remove_task(index_progress)

    def _get_text_dir(self, index: int) -> Path:
        dir = self.text_dir / str(index)
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def _get_model_dir(self, index: int) -> Path:
        dir = self.model_dir / str(index)
        dir.mkdir(parents=True, exist_ok=True)
        return dir

