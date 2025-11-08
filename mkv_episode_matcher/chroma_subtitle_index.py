from pathlib import Path

import chromadb
import numpy as np
from loguru import logger
from rich.console import Console

from mkv_episode_matcher.abstract_subtitle_index import (
    AbstractSubtitleIndex,
    AbstractSubtitleIndexWriter,
)
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import Match, Score
from mkv_episode_matcher.series import Series

console = Console()


class ChromaSubtitleIndex(AbstractSubtitleIndex):
    COLLECTION_NAME = "intervals"

    def __init__(self, config, series: Series):
        super().__init__(config, series)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.index_dir))
        self.collection = self._ensure_collection()

    @property
    def index_dir(self):
        return self.series.index_dir / "chroma.index"

    def _ensure_collection(self):
        return self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


class ChromaSubtitleIndexWriter(ChromaSubtitleIndex, AbstractSubtitleIndexWriter):
    def delete_index_files(self, progress):
        logger.info(
            f"Removing Chroma collection for: {self.series.name}, "
            f"deleting: {self.index_dir}"
        )
        self.client.delete_collection(self.COLLECTION_NAME)

        # Recreate an empty collection for subsequent indexing.
        self.collection = self._ensure_collection()

    def build_interval_index(self, embeddings_file: Path):
        interval_index = int(embeddings_file.stem)
        logger.info(f"Upserting Chroma embeddings for interval: {interval_index}")

        embedding_entries = np.load(embeddings_file)
        if embedding_entries.size == 0:
            logger.warning(
                f"No Chroma embeddings found in {embeddings_file}, skipping interval"
            )
            return

        embeddings: list[list[float]] = (
            embedding_entries["embedding"].astype(np.float32).tolist()
        )

        season_numbers = embedding_entries["episode_key"]["season_number"]
        episode_numbers = embedding_entries["episode_key"]["episode_number"]
        metadatas: list[dict[str, int]] = [
            {
                "season_number": int(season),
                "episode_number": int(episode),
                "interval": interval_index,
            }
            for season, episode in zip(season_numbers, episode_numbers)
        ]

        # Ensure stale entries for the interval are cleared before inserting.
        self.collection.delete(where={"interval": interval_index})

        ids = [
            f"{interval_index}:{int(season)}:{int(episode)}"
            for season, episode in zip(season_numbers, episode_numbers)
        ]
        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)


class ChromaSubtitleIndexReader(ChromaSubtitleIndex):
    def query_intervals(self, embeddings: Path | np.ndarray) -> list[Match]:
        if isinstance(embeddings, Path):
            embeddings = np.load(embeddings)

        if not isinstance(embeddings, np.ndarray):
            raise ValueError(f"Invalid embeddings: {embeddings}")

        scores_by_episode: dict[EpisodeKey, Score] = {}

        for interval, query_vector in zip(embeddings["interval_index"],
                                          embeddings["embedding"]):

            result = self.collection.query(
                query_embeddings=[query_vector],
                where={"interval": int(interval)},
                n_results=5,
                include=["metadatas", "distances"],
            )

            metadatas = result.get("metadatas") or []
            distances = result.get("distances") or []
            if not metadatas or not distances:
                logger.warning(f"No Chroma results for interval: {interval}")
                continue

            for metadata, distance in zip(metadatas[0], distances[0]):
                season = metadata.get("season_number")
                episode = metadata.get("episode_number")
                if season is None or episode is None:
                    logger.warning(
                        f"Chroma metadata missing episode information: {metadata}"
                    )
                    continue

                key = EpisodeKey(int(season), int(episode))
                current = scores_by_episode.setdefault(key, Score(0, 1000000))
                scores_by_episode[key] = Score(
                    current.count + 1, min(current.min_distance, float(distance))
                )

        ordered_ep_id = sorted(scores_by_episode, key=lambda k: scores_by_episode[k])
        return [
            Match(ep_id[0], ep_id[1], scores_by_episode[ep_id])
            for ep_id in ordered_ep_id[:5]
        ]


ChromaSubtitleIndex.reader_type = ChromaSubtitleIndexReader
ChromaSubtitleIndex.writer_type = ChromaSubtitleIndexWriter
