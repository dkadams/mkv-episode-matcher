import hashlib
import json
from pathlib import Path

import numpy as np
from loguru import logger

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.embedding_model import EmbeddingModel
from mkv_episode_matcher.series import Series


class TranscriptionEmbeddingsExtractor:
    def __init__(self, config: Configuration, series: Series,
        model: EmbeddingModel):
        self.config = config
        self.series = series
        self.model = model

    def execute(self, transcription_json: Path,
        duration: int, count: int) -> Path:
        dtype = self.get_dtype()

        output_dir = self.series.ensure_transcription_embeddings_dir(duration, count)
        embeddings_file = output_dir / transcription_json.with_suffix(".npy").name
        if embeddings_file.exists():
            existing_embeddings = np.load(embeddings_file).view(dtype)
            existing_hashes = {row.tobytes()
                               for row in existing_embeddings["sha256"]}
        else:
            existing_embeddings = np.array([], dtype=dtype)
            existing_hashes = set()

        with open(transcription_json, 'r') as file:
            transcribed_intervals = json.load(file)

        logger.info(f"Creating query embeddings for file: {transcription_json}")
        new_embeddings_tuples = []
        for id, (interval_index, interval_text) in enumerate(transcribed_intervals.items()):
            digest = hashlib.sha256(interval_text.encode("utf-8")).digest()
            if digest in existing_hashes:
                continue

            embeddings = self.model.encode_document(interval_text)

            digest_array = np.frombuffer(digest, dtype=np.uint8).copy()
            new_embeddings_tuples.append(
                (id, interval_index, embeddings, digest_array)
            )
            existing_hashes.add(digest)

        logger.info(f"Merging existing embeddings with new embeddings for "
                    f"{transcription_json}")
        new_embeddings = np.array(new_embeddings_tuples, dtype=dtype)
        combined = np.concatenate([existing_embeddings, new_embeddings])

        # Get a list of unique indexes to save from combined. reverse the order
        # so that we prefer the values from the new embeddings.
        _, uniq_idx_reversed = np.unique(combined['interval_index'][::-1],
                                         return_index=True)
        # reverse the index to get the original order.
        uniq_idx = len(combined) - 1 - uniq_idx_reversed

        # Get an ordering *of the indicies* by the episode_key
        order = np.argsort(combined['interval_index'][uniq_idx])

        # create a new array, selecting by the unique index, which are ordered
        # by episode_key.
        merged_embeddings = combined[uniq_idx[order]]

        # rewrite the ids to be sequential and contiguous
        merged_embeddings['id'] = np.arange(len(merged_embeddings))

        np.save(embeddings_file, merged_embeddings)
        logger.info(f"Extracted embeddings for: {transcription_json}")

        return embeddings_file

    def get_dtype(self):
        return np.dtype([
            ("id", np.int32),
            ("interval_index", np.int32),
            ("embedding", np.float32,
             (self.model.get_sentence_embedding_dimension(),)),
            ("sha256", np.uint8, (32,))
        ])
