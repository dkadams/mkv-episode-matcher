import json
import math
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import hnswlib
import numpy as np
import pysubs2
from loguru import logger
from rich.console import Console
from rich.progress import Progress
from sentence_transformers import SentenceTransformer

from mkv_episode_matcher.episode import (
    episode_str,
    episode_tuple, episode_from_path, EpisodeKey,
)
from mkv_episode_matcher.indexed_episode_matcher import Match, Score
from mkv_episode_matcher.series import Series, get_specified_episodes

console = Console()


class HnswlibSubtitleIndex:
    def __init__(self, config, series: Series):
        self.config = config
        self.series = series

        self.index_dir = series.index_dir / "hnswlib.index"
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class HnswlibSubtitleIndexWriter(HnswlibSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)
        self.embeddings_dir = self.index_dir / "embeddings"

    def index_series(self):
        episodes = {
            (ep.season_number, ep.episode_number)
            for ep in get_specified_episodes(self.config, self.series)
        }

        subtitle_files = list(self.series.dir.rglob("*.srt"))

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        with Progress() as progress, ThreadPoolExecutor(max_workers=10) as executor:
            if self.index_dir.exists():
                logger.info(
                    f"Rebuilding index for: {self.series.name}, deleting: {self.index_dir}"
                )
                index_files = [
                    file
                    for file in self.index_dir.iterdir()
                    if file.is_file()
                    and file.suffix in {".bin", ".json"}
                ]
                delete_progress = progress.add_task(
                    f"Removing index for: {self.series.name}", total=len(index_files)
                )
                for index_file in index_files:
                    index_file.unlink()
                    progress.update(delete_progress, advance=1)
                progress.remove_task(delete_progress)

            extract_progress = progress.add_task(
                f"Extracting embeddings for {self.series.name} ({self.series.dir})",
                total=len(subtitle_files),
            )

            def extract_embeddings(file: Path):
                logger.info(f"Indexing: {file}")
                episode = episode_from_path(file)
                logger.info(f"Identified: {file} as episode: {episode}")
                if episode in episodes:
                    logger.info(f"Indexing: {file} as episode: {episode}")
                    self.extract_embeddings(file, episode)
                progress.update(extract_progress, advance=1)

            list(executor.map(extract_embeddings, subtitle_files))
            progress.remove_task(extract_progress)

            interval_dirs = [
                dir_path for dir_path in self.embeddings_dir.iterdir() if dir_path.is_dir()
            ]
            index_progress = progress.add_task(
                f"Indexing {self.series.name} ({self.series.dir})",
                total=len(interval_dirs),
            )

            def index_interval(dir_path: Path):
                logger.info(f"Indexing {dir_path}")
                self.build_interval_index(dir_path)
                progress.update(index_progress, advance=1)

            list(executor.map(index_interval, interval_dirs))
            progress.remove_task(index_progress)

    def extract_embeddings(self, path: Path, episode: tuple[int, int]):
        epi_str = episode_str(*episode)
        logger.info(
            f"Indexing episode: {self.series.name} episode: {epi_str}"
        )

        sub_file = pysubs2.load(path, format_="srt")
        metadata = self.series.get_episode_detail(episode)

        interval_count = math.ceil(metadata["runtime"] * 60 / 30)
        intervals = (
            (i, range(i * 30 * 1000, (i + 1) * 30 * 1000)) for i in range(interval_count)
        )

        for index, interval in intervals:
            subs = [
                sub.plaintext
                for sub in sub_file
                if sub.start in interval or sub.end in interval
            ]
            if not subs:
                continue

            interval_text = " ".join(subs)

            interval_dir = self.embeddings_dir / str(index)
            interval_dir.mkdir(parents=True, exist_ok=True)
            text_file = interval_dir / f"{epi_str}.txt"
            if not text_file.exists():
                with open(text_file, "w") as text_out:
                    text_out.write(interval_text)

            embeddings_file = interval_dir / f"{epi_str}.npy"
            if not embeddings_file.exists():
                embeddings = self.model.encode_document(interval_text)
                np.save(embeddings_file, embeddings.astype(np.float32))

    def build_interval_index(self, interval_dir: Path):
        embedding_files = list(interval_dir.glob("*.npy"))
        embedding_files.sort(key=lambda f: f.stem)

        if not embedding_files:
            logger.warning(f"No embeddings found for interval dir: {interval_dir}")
            return

        index_directory = {
            index: episode_tuple(f.stem) for index, f in enumerate(embedding_files)
        }

        interval = interval_dir.stem
        with open(self.index_dir / f"{interval}.json", "w") as json_out:
            json.dump(index_directory, json_out)

        dim = self.model.get_sentence_embedding_dimension()
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(embedding_files), ef_construction=200, M=16)

        vectors = []
        labels = []
        for episode_index, file in enumerate(embedding_files):
            with open(file, "rb") as f:
                vector = np.load(f)
            vectors.append(vector)
            labels.append(episode_index)

        if vectors:
            data = np.vstack(vectors).astype(np.float32)
            index.add_items(data, np.array(labels))

        index.set_ef(200)
        index.save_index(str(self.index_dir / f"{interval}.bin"))

class HnswlibSubtitleIndexReader(HnswlibSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)

        self.indexes = self.load_indexes()

    def query_intervals(self, text_segments: list[tuple[int, str]]) -> list[Match]:
        distances_by_episode: dict[EpisodeKey, Score]  = {}
        for interval, text in text_segments:
            index_entry = self.indexes.get(interval)
            if index_entry is None:
                logger.warning(f"No index found for interval: {interval}")
                continue

            directory, index = index_entry

            # Avoid asking for more results than are available. Doing so causes
            # hnswlib to throw this RuntimeError:
            #   Cannot return the results in a contiguous 2D array. Probably
            #       ef or M is too small
            neighbor_count = min(5, len(directory))
            if neighbor_count == 0:
                logger.warning(
                    f"Index metadata empty for interval: {interval}, skipping query"
                )
                continue

            query = self.model.encode_query(text).astype(np.float32)
            labels, distances = index.knn_query(query, k=neighbor_count,
                                                num_threads=1, filter=None)
            ids = labels[0]
            dists = distances[0]

            for item_id, distance in zip(ids, dists):
                if item_id == -1:
                    continue
                key = directory[item_id]
                cur = distances_by_episode.setdefault(key, Score(0, 1000000))
                score = Score(cur.count + 1, min(cur.min_distance, distance))
                distances_by_episode[key] = score

        ordered_ep_id = sorted(distances_by_episode,
                                key=lambda k: distances_by_episode[k])

        return [Match(ep_id[0], ep_id[1], distances_by_episode[ep_id])
                for ep_id in ordered_ep_id[:5]]

    def load_indexes(self) -> dict[int, tuple[dict[int, EpisodeKey], hnswlib.Index]]:
        if not self.index_dir.exists():
            console.print(
                f"[bold red]No index for series: {self.series.name}"
                f" Use mkv-episode-matcher index-subs to build indexes"
            )

        intervals = [
            int(file.stem)
            for file in self.index_dir.iterdir()
            if file.is_file() and file.suffix == ".bin"
        ]
        return {interval: self.get_index(interval) for interval in intervals}

    def get_index(
        self, interval: int
    ) -> tuple[dict[int, EpisodeKey], hnswlib.Index] | None:
        index_file = self.index_dir / f"{interval}.bin"
        directory_file = self.index_dir / f"{interval}.json"
        if not (index_file.exists() and directory_file.exists()):
            logger.warning(
                f"Incomplete or missing index for interval: {interval}: "
                f"{index_file} and/or {directory_file} not found."
            )
            return None

        logger.info(f"Loading index for interval: {interval}: {index_file}")
        dim = self.model.get_sentence_embedding_dimension()
        index = hnswlib.Index(space="cosine", dim=dim)
        with open(directory_file, "r") as json_in:
            data = json.load(json_in)
            index_directory = {
                int(id): EpisodeKey(season, episode)
                for id, (season, episode) in data.items()
            }

        if not index_directory:
            logger.warning(f"Empty index metadata for interval: {interval}")
            return None

        index.load_index(str(index_file), max_elements=len(index_directory))
        index.set_ef(200)
        return index_directory, index

HnswlibSubtitleIndex.reader_type = HnswlibSubtitleIndexReader
HnswlibSubtitleIndex.writer_type = HnswlibSubtitleIndexWriter
