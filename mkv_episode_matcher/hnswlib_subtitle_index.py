from pathlib import Path

import hnswlib
import numpy as np
from loguru import logger
from rich.console import Console

from mkv_episode_matcher.abstract_subtitle_index import AbstractSubtitleIndex, \
    AbstractSubtitleIndexWriter
from mkv_episode_matcher.embeddings_extractor import EmbeddingsExtractor
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

    def build_interval_index(self, embeddings_file: Path):
        embedding_entries = np.load(embeddings_file)

        dim = self.embedding_model.get_sentence_embedding_dimension()
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(embedding_entries),
                         ef_construction=200, M=16)

        # use one thread for now, since we're already parallelizing the build
        index.add_items(embedding_entries["embedding"], embedding_entries["id"],
                        num_threads=1)

        index.set_ef(200)
        interval_index = embeddings_file.stem
        index.save_index(str(self.index_dir / f"{interval_index}.idx"))

class HnswlibSubtitleIndexReader(HnswlibSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)

        self.indexes = self.load_indexes()

    def query_intervals(self, text_segments: list[tuple[int, str]]) -> list[Match]:
        distances_by_episode: dict[EpisodeKey, Score] = {}
        for interval, text in text_segments:
            directory, index = self.indexes.get(interval)
            if index is None:
                logger.warning(f"No index found for interval: {interval}")
                continue

            # Avoid asking for more results than are available. Doing so causes
            # hnswlib to throw this RuntimeError:
            #   Cannot return the results in a contiguous 2D array. Probably
            #       ef or M is too small
            neighbor_count = min(5, index.get_current_count())
            if neighbor_count == 0:
                logger.warning(
                    f"Index metadata empty for interval: {interval}, skipping query"
                )
                continue

            query = self.embedding_model.encode_query(text).astype(np.float32)
            # We only passed a query vector, so we can squeeze the results since
            # they will only ever have one dimension.
            ids, distances = map(np.squeeze, index.knn_query(query, k=neighbor_count,
                                             num_threads=1, filter=None))
            logger.info(f"Query: {interval} -> {ids} -> {distances}")
            episode_keys = [directory[id] for id in ids if id != -1]
            if len(episode_keys) == 0 or len(distances) == 0:
                continue

            for key, distance in zip(episode_keys, distances):
                cur = distances_by_episode.setdefault(key, Score(0, 1000000))
                score = Score(cur.count + 1, min(cur.min_distance, distance))
                distances_by_episode[key] = score

        ordered_keys = sorted(distances_by_episode,
                              key=lambda k: distances_by_episode[k])

        return [Match(key.season_number, key.episode_number, distances_by_episode[key])
                for key in ordered_keys[:5]]

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

    def get_index(self, interval: int) -> tuple[dict[int, EpisodeKey], hnswlib.Index] | None:
        embedding_file = self.model_dir / f"{interval}.npy"
        directory = EmbeddingsExtractor.get_directory(embedding_file)

        index_file = self.index_dir / f"{interval}.idx"
        logger.info(f"Loading index for interval: {interval}: {index_file}")
        dim = self.embedding_model.get_sentence_embedding_dimension()
        index = hnswlib.Index(space="cosine", dim=dim)

        index.load_index(str(index_file))
        index.set_ef(200)
        return directory, index

HnswlibSubtitleIndex.reader_type = HnswlibSubtitleIndexReader
HnswlibSubtitleIndex.writer_type = HnswlibSubtitleIndexWriter
