from pathlib import Path

import numpy as np
from annoy import AnnoyIndex
from loguru import logger
from rich.console import Console

from mkv_episode_matcher.abstract_subtitle_index import AbstractSubtitleIndex, \
    AbstractSubtitleIndexWriter
from mkv_episode_matcher.embeddings_extractor import EmbeddingsExtractor
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import Match, Score
from mkv_episode_matcher.series import Series

console = Console()

class AnnoySubtitleIndex(AbstractSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)

    @property
    def index_dir(self):
        return self.series.index_dir / "annoy.index"

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

    def query_intervals(self, text_segments: list[tuple[int, str]]) -> list[Match]:
        scores_by_episode: dict[EpisodeKey, Score] = {}
        for interval, text in text_segments:
            index_entry = self.indexes.get(interval)
            if index_entry is None:
                logger.warning(f"No index found for interval: {interval}")
                continue

            directory, index = index_entry
            query = self.embedding_model.encode_query(text)
            ids, distances = index.get_nns_by_vector(query, 5, include_distances=True)
            logger.info(f"Query: {interval} -> {ids} -> {distances}")
            for id, distance in zip(ids, distances):
                episode_id = directory[id]

                cur = scores_by_episode.setdefault(episode_id, Score(0, 1000000))
                score = Score(cur.count + 1, min(cur.min_distance, distance))
                scores_by_episode[episode_id] = score

        ordered_ep_id = sorted(scores_by_episode,
                               # Order by frequency DESCENDING, distance ASCENDING
                               key=lambda k: scores_by_episode[k])

        return [Match(ep_id[0], ep_id[1], scores_by_episode[ep_id])
                for ep_id in ordered_ep_id[:5]] # only return the top 5 results

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
        directory = EmbeddingsExtractor.get_directory(embedding_file)

        return directory, index

AnnoySubtitleIndex.reader_type = AnnoySubtitleIndexReader
AnnoySubtitleIndex.writer_type = AnnoySubtitleIndexWriter
