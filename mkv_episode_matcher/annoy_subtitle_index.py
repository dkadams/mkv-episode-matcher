import json
from pathlib import Path

import numpy as np
from annoy import AnnoyIndex
from loguru import logger
from rich.console import Console

from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.embedding_model import EmbeddingModel, SentenceTransformerModel
from mkv_episode_matcher.indexed_episode_matcher import Match, Score
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.abstract_subtitle_index import AbstractSubtitleIndex, \
    AbstractSubtitleIndexWriter

console = Console()

class AnnoySubtitleIndex(AbstractSubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)

    @property
    def index_dir(self):
        return self.series.index_dir / "annoy.index"

class AnnoySubtitleIndexWriter(AnnoySubtitleIndex, AbstractSubtitleIndexWriter):
    def build_interval_index(self, interval_dir: Path):
        embeddings = list(interval_dir.glob("*.npy"))
        embeddings.sort(key=lambda f: f.stem)

        index_directory = {index: EpisodeKey.from_str(f.stem)
                            for index, f in enumerate(embeddings)}

        interval = interval_dir.stem
        with open(self.index_dir / f"{interval}.json", "w") as json_out:
            json.dump(index_directory, json_out)

        index = AnnoyIndex(self.embedding_model.get_sentence_embedding_dimension(),
                           "angular")
        for episode_index, file in enumerate(embeddings):
            with open(file, "rb") as f:
                embeddings = np.load(f)

            logger.info(f"Indexing: {file} -> {interval}[{episode_index}]")
            index.add_item(episode_index, embeddings)

        # we're already parallelizing the build, so don't use more threads'
        index.build(50, n_jobs=1)
        index.save(str(self.index_dir / f"{interval}.idx"))
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
        index_file = self.index_dir / f"{interval}.idx"
        directory_file = self.index_dir / f"{interval}.json"
        if not (index_file.exists() and directory_file.exists()):
            # This might happen if there's a mismatch between the video file
            # lengths and the episode metadata.
            logger.warning(
                f"Incomplete or missing index for interval: {interval}: "
                f"f{index_file} and/or {directory_file} not found.")
            return None

        logger.info(f"Loading index for interval: {interval}: {index_file}")
        index = AnnoyIndex(self.embedding_model.get_sentence_embedding_dimension(), "angular")
        index.load(str(index_file))
        logger.info(f"Loading directory for interval: {interval}: {directory_file}")
        with open(directory_file, "r") as json_in:
            data = json.load(json_in)
            index_directory = {int(id): EpisodeKey(season, episode)
                               for id, (season, episode) in data.items()}

        return index_directory, index

AnnoySubtitleIndex.reader_type = AnnoySubtitleIndexReader
AnnoySubtitleIndex.writer_type = AnnoySubtitleIndexWriter
