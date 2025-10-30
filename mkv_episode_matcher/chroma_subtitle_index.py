import shutil
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

    def build_interval_index(self, interval_dir: Path):
        embedding_files = sorted(interval_dir.glob("*.npy"), key=lambda f: f.stem)
        if not embedding_files:
            logger.warning(f"No embeddings found for interval dir: {interval_dir}")
            return

        interval = int(interval_dir.stem)
        logger.info(f"Upserting Chroma embeddings for interval: {interval}")

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, int]] = []

        for embedding_path in embedding_files:
            vector = np.load(embedding_path).astype(np.float32)
            episode = EpisodeKey.from_str(embedding_path.stem)
            ids.append(f"{interval}:{episode.season_number}:{episode.episode_number}")
            embeddings.append(vector.tolist())
            metadatas.append(
                {
                    "season_number": episode.season_number,
                    "episode_number": episode.episode_number,
                    "interval": interval,
                }
            )

        # Ensure stale entries for the interval are cleared before inserting.
        self.collection.delete(where={"interval": interval})
        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)


class ChromaSubtitleIndexReader(ChromaSubtitleIndex):
    def query_intervals(self, text_segments: list[tuple[int, str]]) -> list[Match]:
        scores_by_episode: dict[EpisodeKey, Score] = {}

        for interval, text in text_segments:
            query_vector = (
                self.embedding_model.encode_query(text).astype(np.float32).tolist()
            )

            result = self.collection.query(
                query_embeddings=[query_vector],
                where={"interval": interval},
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
