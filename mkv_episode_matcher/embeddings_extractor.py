import hashlib
from pathlib import Path

import numpy as np
import pysubs2
from loguru import logger

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.embedding_model import EmbeddingModel
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.series import Series


class EmbeddingsExtractor:
    def __init__(self, config: Configuration, series: Series,
        model: EmbeddingModel, model_dir: Path):
        self.config = config
        self.series = series
        self.model = model
        self.model_dir = model_dir

    def execute(self, interval_index: int,
            interval_subs_paths: list[tuple[EpisodeKey, Path]]):
        dtype = self.get_dtype()

        self.model_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = self.model_dir / f"{interval_index}.npy"
        if embeddings_file.exists():
            existing_embeddings = np.load(embeddings_file).view(dtype)
            existing_hashes = {row.tobytes()
                               for row in existing_embeddings["sha256"]}
        else:
            existing_embeddings = np.array([], dtype=dtype)
            existing_hashes = set()

        logger.info(f"Creating new embeddings for series: {self.series.name} "
                    f"interval: {interval_index}")
        new_embeddings_tuples = []
        for id, (episode_key, subs_path) in enumerate(interval_subs_paths):
            subs_file = pysubs2.load(str(subs_path), format_="srt")
            if interval_index >= len(subs_file):
                continue

            interval_text = subs_file[interval_index].plaintext
            if interval_text is None:
                continue

            digest = hashlib.sha256(interval_text.encode("utf-8")).digest()
            if digest in existing_hashes:
                continue

            embeddings = self.model.encode_document(interval_text)

            digest_array = np.frombuffer(digest, dtype=np.uint8).copy()
            new_embeddings_tuples.append(
                (id, episode_key, embeddings, digest_array))
            existing_hashes.add(digest)

        logger.info(f"Merging existing embeddings with new embeddings for "
                    f"{self.series.name}, interval: {interval_index}")
        new_embeddings = np.array(new_embeddings_tuples, dtype=dtype)
        combined = np.concatenate([existing_embeddings, new_embeddings])

        # Get a list of unique indexes to save from combined. reverse the order
        # so that we prefer the values from the new embeddings.
        _, uniq_idx_reversed = np.unique(combined['episode_key'][::-1],
                                         return_index=True)
        # reverse the index to get the original order.
        uniq_idx = len(combined) - 1 - uniq_idx_reversed

        # Get an ordering *of the indicies* by the episode_key
        order = np.argsort(combined['episode_key'][uniq_idx])

        # create a new array, selecting by the unique index, which are ordered
        # by episode_key.
        merged_embeddings = combined[uniq_idx[order]]

        # rewrite the ids to be sequential and contiguous
        merged_embeddings['id'] = np.arange(len(merged_embeddings))

        np.save(embeddings_file, merged_embeddings)
        logger.info(f"Extracted embeddings for series: {self.series.name},"
                    f" interval: {interval_index}")

    @staticmethod
    def get_directory(embeddings_file: Path) -> dict[int, EpisodeKey]:
        embeddings_entry = np.load(embeddings_file)

        return {id: EpisodeKey(season_number, episode_number)
                for id, season_number, episode_number
                in zip(embeddings_entry["id"],
                       embeddings_entry["episode_key"]["season_number"],
                       embeddings_entry["episode_key"]["episode_number"])}

    def get_dtype(self):
        return np.dtype([
            ("id", np.int32),
            ("episode_key",
             [("season_number", np.int32), ("episode_number", np.int32)]),
            ("embedding", np.float32,
             (self.model.get_sentence_embedding_dimension(),)),
            ("sha256", np.uint8, (32,))
        ])
