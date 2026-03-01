from pathlib import Path

import hnswlib
import numpy as np
from loguru import logger
from rich.console import Console

from mkv_episode_matcher.abstract_subtitle_index import AbstractSubtitleIndex, \
    AbstractSubtitleIndexWriter
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import IntervalMatch
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.subtitle_embeddings_extractor import \
    SubtitleEmbeddingsExtractor
from mkv_episode_matcher.windowing import (
    neighbor_window_indexes,
    map_segment_index_to_window_index,
)

console = Console()


class HnswlibSubtitleIndex(AbstractSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)
        if not self.index_dir.exists():
            self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_dir(self):
        return self.profile_dir / "hnswlib.index"

class HnswlibSubtitleIndexWriter(HnswlibSubtitleIndex, AbstractSubtitleIndexWriter):

    def build_interval_index(self, embeddings_file: Path):
        embedding_entries = np.load(embeddings_file)

        dim = self.embedding_model.get_sentence_embedding_dimension()
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(embedding_entries),
                         ef_construction=200, M=16)

        # use one thread for now, since we're already parallelizing the build
        index.add_items(embedding_entries["embedding"], embedding_entries["id"])

        logger.info(f"Built index for interval: {embeddings_file.stem}. items: {index.get_current_count()}")
        index.set_ef(200)
        interval_index = embeddings_file.stem
        index_path = str(self.index_dir / f"{interval_index}.idx")
        logger.info(f"Saving index for interval: {interval_index} to: {index_path}")
        index.save_index(index_path)
        logger.info(f"Saved index for interval?: {interval_index} to?: {index_path}")

class HnswlibSubtitleIndexReader(HnswlibSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)

        self.indexes = self.load_indexes()

    def query_intervals(self, embeddings_path: Path,
        max_results_per_query: int = 10) -> list[IntervalMatch]:
        embeddings = np.load(embeddings_path)

        results: list[IntervalMatch] = []
        for segment_index, embedding in zip(embeddings["interval_index"],
                                            embeddings["embedding"]):
            mapped_interval = map_segment_index_to_window_index(
                int(segment_index),
                self.series.segment_duration,
                self.window_config.stride_seconds,
            )
            candidate_intervals = [
                idx
                for idx in neighbor_window_indexes(mapped_interval)
                if idx in self.indexes
            ]
            if not candidate_intervals:
                logger.warning(f"No index found for segment: {segment_index} mapped to interval: {mapped_interval}")
                continue

            per_episode: dict[EpisodeKey, IntervalMatch] = {}
            for interval_idx in candidate_intervals:
                directory, index = self.indexes[interval_idx]

                # Avoid asking for more results than are available. Doing so causes
                # hnswlib to throw this RuntimeError:
                #   Cannot return the results in a contiguous 2D array. Probably
                #       ef or M is too small
                neighbor_count = min(max_results_per_query, index.get_current_count())
                if neighbor_count == 0:
                    logger.warning(
                        f"Index empty for interval: {interval_idx}, skipping."
                    )
                    continue

                ids_by_q, dists_by_q = index.knn_query(
                    embedding, k=neighbor_count, num_threads=1, filter=None
                )
                # knn_query supports multiple queries, but we only have one. So
                # there'll only be one result.
                ids, distances = ids_by_q[0], dists_by_q[0]
                for id, distance in zip(ids, distances):
                    episode = directory[id]
                    candidate = IntervalMatch(
                        embeddings_path,
                        int(segment_index),
                        episode,
                        interval_idx,
                        float(distance),
                    )
                    existing = per_episode.get(episode)
                    if existing is None or candidate.distance < existing.distance or (
                        candidate.distance == existing.distance
                        and candidate.episode_index < existing.episode_index
                    ):
                        per_episode[episode] = candidate

            if per_episode:
                deduped = sorted(
                    per_episode.values(),
                    key=lambda match: (match.distance, match.episode_index, match.episode),
                )[:max_results_per_query]
                results.extend(deduped)
        return results


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
        directory = SubtitleEmbeddingsExtractor.get_directory(embedding_file)

        index_file = self.index_dir / f"{interval}.idx"
        logger.info(f"Loading index for interval: {interval}: {index_file}")
        dim = self.embedding_model.get_sentence_embedding_dimension()
        index = hnswlib.Index(space="cosine", dim=dim)

        index.load_index(str(index_file))
        index.set_ef(200)
        return directory, index

HnswlibSubtitleIndex.reader_type = HnswlibSubtitleIndexReader
HnswlibSubtitleIndex.writer_type = HnswlibSubtitleIndexWriter
