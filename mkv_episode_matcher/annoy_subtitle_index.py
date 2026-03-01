from pathlib import Path

import numpy as np
from annoy import AnnoyIndex
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
    distance_with_window_penalty,
    neighbor_window_indexes,
    map_segment_index_to_window_index,
    should_expand_to_neighbor_windows,
)

console = Console()

class AnnoySubtitleIndex(AbstractSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)
        if not self.index_dir.exists():
            self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_dir(self):
        return self.profile_dir / "annoy.index"

class AnnoySubtitleIndexWriter(AnnoySubtitleIndex, AbstractSubtitleIndexWriter):
    def build_interval_index(self, embeddings_file: Path):
        embedding_entries = np.load(embeddings_file)


        index = AnnoyIndex(self.embedding_model.get_sentence_embedding_dimension(),
                           "angular")
        for entry in embedding_entries:
            index.add_item(entry["id"], entry["embedding"])

        # we're already parallelizing the build, so don't use more threads'
        index.build(50, n_jobs=1)
        interval_index = embeddings_file.stem
        index.save(str(self.index_dir / f"{interval_index}.idx"))
        index.unload()

class AnnoySubtitleIndexReader(AnnoySubtitleIndex):
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

            per_episode: dict[EpisodeKey, IntervalMatch] = {}
            mapped_window_distances: list[float] = []

            mapped_entry = self.indexes.get(mapped_interval)
            if mapped_entry is not None:
                for episode, raw_distance, interval_idx in self._query_interval(
                    mapped_entry,
                    embedding,
                    mapped_interval,
                    max_results_per_query,
                ):
                    mapped_window_distances.append(raw_distance)
                    adjusted_distance = distance_with_window_penalty(
                        raw_distance,
                        interval_idx,
                        mapped_interval,
                    )
                    candidate = IntervalMatch(
                        embeddings_path,
                        int(segment_index),
                        episode,
                        interval_idx,
                        adjusted_distance,
                    )
                    self._merge_best_by_episode(per_episode, candidate)

            if should_expand_to_neighbor_windows(mapped_window_distances):
                for interval_idx in neighbor_window_indexes(mapped_interval):
                    if interval_idx == mapped_interval:
                        continue
                    index_entry = self.indexes.get(interval_idx)
                    if index_entry is None:
                        continue
                    for episode, raw_distance, candidate_interval in self._query_interval(
                        index_entry,
                        embedding,
                        interval_idx,
                        max_results_per_query,
                    ):
                        adjusted_distance = distance_with_window_penalty(
                            raw_distance,
                            candidate_interval,
                            mapped_interval,
                        )
                        candidate = IntervalMatch(
                            embeddings_path,
                            int(segment_index),
                            episode,
                            candidate_interval,
                            adjusted_distance,
                        )
                        self._merge_best_by_episode(per_episode, candidate)

            if per_episode:
                deduped = sorted(
                    per_episode.values(),
                    key=lambda match: (match.distance, match.episode_index, match.episode),
                )[:max_results_per_query]
                results.extend(deduped)
            else:
                logger.warning(
                    f"No index results found for segment: {segment_index} mapped to interval: {mapped_interval}"
                )

        return results

    @staticmethod
    def _merge_best_by_episode(per_episode: dict[EpisodeKey, IntervalMatch], candidate: IntervalMatch):
        existing = per_episode.get(candidate.episode)
        if existing is None or candidate.distance < existing.distance or (
            candidate.distance == existing.distance
            and candidate.episode_index < existing.episode_index
        ):
            per_episode[candidate.episode] = candidate

    @staticmethod
    def _query_interval(index_entry: tuple[dict[int, EpisodeKey], AnnoyIndex],
        embedding: np.ndarray, interval_idx: int, max_results_per_query: int) -> list[tuple[EpisodeKey, float, int]]:
        directory, index = index_entry
        ids, distances = index.get_nns_by_vector(
            embedding,
            max_results_per_query,
            include_distances=True,
        )
        return [
            (directory[id], float(distance), interval_idx)
            for id, distance in zip(ids, distances)
        ]

    def load_indexes(self):
        if not self.index_dir.exists():
            console.print(f"[bold red]No index for series: {self.series.name}"
                          f" Use mkv-episode-matcher index-subs to build indexes")

        intervals = [int(file.stem) for file in self.index_dir.iterdir()
                     if file.is_file() and file.suffix == ".idx"]
        return {interval: self.get_index(interval) for interval in intervals}

    def get_index(self, interval: int) -> tuple[dict[int, EpisodeKey], AnnoyIndex] | None:
        embedding_file = self.model_dir / f"{interval}.npy"
        index_file = self.index_dir / f"{interval}.idx"
        if not (index_file.exists() and embedding_file.exists()):
            # This might happen if there's a mismatch between the video file
            # lengths and the episode metadata.
            logger.warning(
                f"Incomplete or missing index for interval: {interval}: "
                f"f{index_file} and/or {embedding_file} not found.")
            return None

        logger.info(f"Loading index for interval: {interval}: {index_file}")
        index = AnnoyIndex(self.embedding_model.get_sentence_embedding_dimension(), "angular")
        index.load(str(index_file))
        logger.info(f"Loading directory for interval: {interval} "
                    f"from: {embedding_file}")
        directory = SubtitleEmbeddingsExtractor.get_directory(embedding_file)

        return directory, index

AnnoySubtitleIndex.reader_type = AnnoySubtitleIndexReader
AnnoySubtitleIndex.writer_type = AnnoySubtitleIndexWriter
