from pathlib import Path

import hnswlib
import numpy as np
from loguru import logger
from rich.console import Console

from mkv_episode_matcher.abstract_subtitle_index import AbstractSubtitleIndex, \
    AbstractSubtitleIndexWriter
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import IntervalMatch
from mkv_episode_matcher.segment_quality import load_low_info_intervals
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.subtitle_embeddings_extractor import \
    SubtitleEmbeddingsExtractor
from mkv_episode_matcher.windowing import (
    distance_with_window_penalty,
    merge_episode_window_hit,
    neighbor_window_indexes,
    map_segment_index_to_window_index,
    should_expand_to_neighbor_windows,
    support_aware_score,
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
        max_results_per_query: int | None = None) -> list[IntervalMatch]:
        embeddings = np.load(embeddings_path)
        max_results = int(max_results_per_query or self.max_results_per_query)

        low_info_intervals = set()
        if self.low_info_filter:
            loaded = load_low_info_intervals(embeddings_path)
            if loaded is None:
                logger.warning(
                    f"Low-info sidecar missing for {embeddings_path}; "
                    "running without low-info filtering."
                )
            else:
                low_info_intervals = loaded

        rows = list(zip(embeddings["interval_index"], embeddings["embedding"]))
        filtered_rows = [
            (segment_index, embedding)
            for segment_index, embedding in rows
            if int(segment_index) not in low_info_intervals
        ]
        if self.low_info_filter and rows and not filtered_rows:
            logger.warning(
                f"All segments were filtered as low-info for {embeddings_path}; "
                "falling back to unfiltered matching."
            )
            rows_to_score = rows
        elif self.low_info_filter:
            rows_to_score = filtered_rows
        else:
            rows_to_score = rows

        results: list[IntervalMatch] = []
        for segment_index, embedding in rows_to_score:
            mapped_interval = map_segment_index_to_window_index(
                int(segment_index),
                self.series.segment_duration,
                self.window_config.stride_seconds,
            )
            ranked = self.query_segment_embedding(
                embedding=embedding,
                mapped_interval=mapped_interval,
                max_results_per_query=max_results,
            )
            if ranked:
                for episode, score, best_window_index in ranked[:max_results]:
                    results.append(
                        IntervalMatch(
                            embeddings_path,
                            int(segment_index),
                            episode,
                            int(best_window_index),
                            float(score),
                        )
                    )
            else:
                logger.warning(
                    f"No index results found for segment: {segment_index} mapped to interval: {mapped_interval}"
                )
        return results

    def query_segment_embedding(
        self,
        embedding: np.ndarray,
        mapped_interval: int,
        max_results_per_query: int | None = None,
        neighbor_radius: int | None = None,
        allowed_episodes: set[EpisodeKey] | None = None,
        expansion_mode: str | None = None,
    ) -> list[tuple[EpisodeKey, float, int]]:
        max_results = int(max_results_per_query or self.max_results_per_query)
        radius = int(self.window_neighbor_radius if neighbor_radius is None else neighbor_radius)
        mode = expansion_mode or self.window_expansion_mode

        per_episode_support = {}
        mapped_window_distances: list[float] = []

        mapped_entry = self.indexes.get(int(mapped_interval))
        if mapped_entry is not None:
            for episode, raw_distance, interval_idx in self._query_interval(
                mapped_entry,
                embedding,
                int(mapped_interval),
                max_results,
            ):
                if allowed_episodes is not None and episode not in allowed_episodes:
                    continue
                mapped_window_distances.append(raw_distance)
                adjusted_distance = distance_with_window_penalty(
                    raw_distance,
                    interval_idx,
                    int(mapped_interval),
                )
                merge_episode_window_hit(
                    per_episode_support,
                    episode,
                    interval_idx,
                    adjusted_distance,
                )

        should_expand = should_expand_to_neighbor_windows(
            mapped_window_distances,
            expansion_mode=mode,
        )
        if should_expand:
            for interval_idx in neighbor_window_indexes(
                int(mapped_interval),
                radius=radius,
            ):
                if interval_idx == int(mapped_interval):
                    continue
                index_entry = self.indexes.get(interval_idx)
                if index_entry is None:
                    continue
                for episode, raw_distance, candidate_interval in self._query_interval(
                    index_entry,
                    embedding,
                    interval_idx,
                    max_results,
                ):
                    if allowed_episodes is not None and episode not in allowed_episodes:
                        continue
                    adjusted_distance = distance_with_window_penalty(
                        raw_distance,
                        candidate_interval,
                        int(mapped_interval),
                    )
                    merge_episode_window_hit(
                        per_episode_support,
                        episode,
                        candidate_interval,
                        adjusted_distance,
                    )

        if not per_episode_support:
            return []

        ranked = []
        for episode, support in per_episode_support.items():
            score, nearest_offset = support_aware_score(
                support,
                int(mapped_interval),
                support_window_bonus=self.support_window_bonus,
                support_offset_penalty=self.support_offset_penalty,
            )
            ranked.append((
                float(score),
                int(nearest_offset),
                episode,
                int(support.best_window_index),
            ))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [
            (episode, score, best_window_index)
            for score, _, episode, best_window_index in ranked[:max_results]
        ]

    @staticmethod
    def _query_interval(index_entry: tuple[dict[int, EpisodeKey], hnswlib.Index],
        embedding: np.ndarray, interval_idx: int, max_results_per_query: int) -> list[tuple[EpisodeKey, float, int]]:
        directory, index = index_entry

        # Avoid asking for more results than are available. Doing so causes
        # hnswlib to throw this RuntimeError:
        #   Cannot return the results in a contiguous 2D array. Probably
        #       ef or M is too small
        neighbor_count = min(max_results_per_query, index.get_current_count())
        if neighbor_count == 0:
            return []

        ids_by_q, dists_by_q = index.knn_query(
            embedding, k=neighbor_count, num_threads=1, filter=None
        )
        # knn_query supports multiple queries, but we only have one. So
        # there'll only be one result.
        ids, distances = ids_by_q[0], dists_by_q[0]
        return [
            (directory[id], float(distance), interval_idx)
            for id, distance in zip(ids, distances)
        ]


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
