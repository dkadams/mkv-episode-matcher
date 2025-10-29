import json
from pathlib import Path

import hnswlib
import numpy as np
from loguru import logger
from rich.console import Console

from mkv_episode_matcher.abstract_subtitle_index import AbstractSubtitleIndex, \
    AbstractSubtitleIndexWriter
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import Match, Score
from mkv_episode_matcher.series import Series

console = Console()


class HnswlibSubtitleIndex(AbstractSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)

    @property
    def index_dir(self):
        return self.series.index_dir / "hnswlib.index"


class HnswlibSubtitleIndexWriter(HnswlibSubtitleIndex, AbstractSubtitleIndexWriter):

    def build_interval_index(self, interval_dir: Path):
        embedding_files = list(interval_dir.glob("*.npy"))
        embedding_files.sort(key=lambda f: f.stem)

        if not embedding_files:
            logger.warning(f"No embeddings found for interval dir: {interval_dir}")
            return

        index_directory = {index: EpisodeKey.from_str(f.stem)
                           for index, f in enumerate(embedding_files)}

        interval = interval_dir.stem
        with open(self.index_dir / f"{interval}.json", "w") as json_out:
            json.dump(index_directory, json_out)

        dim = self.embedding_model.get_sentence_embedding_dimension()
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
        index.save_index(str(self.index_dir / f"{interval}.idx"))

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

            query = self.embedding_model.encode_query(text).astype(np.float32)
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
            if file.is_file() and file.suffix == ".idx"
        ]
        return {interval: self.get_index(interval) for interval in intervals}

    def get_index(
        self, interval: int
    ) -> tuple[dict[int, EpisodeKey], hnswlib.Index] | None:
        index_file = self.index_dir / f"{interval}.idx"
        directory_file = self.index_dir / f"{interval}.json"
        if not (index_file.exists() and directory_file.exists()):
            logger.warning(
                f"Incomplete or missing index for interval: {interval}: "
                f"{index_file} and/or {directory_file} not found."
            )
            return None

        logger.info(f"Loading index for interval: {interval}: {index_file}")
        dim = self.embedding_model.get_sentence_embedding_dimension()
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
