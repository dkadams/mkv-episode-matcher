import json
import math
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pysubs2
from annoy import AnnoyIndex
from loguru import logger
from rich.console import Console
from rich.progress import Progress
from sentence_transformers import SentenceTransformer

from mkv_episode_matcher.episode import episode_str, \
    episode_tuple, episode_from_path
from mkv_episode_matcher.series import Series, get_specified_episodes

console = Console()

class AnnoySubtitleIndex:
    def __init__(self, config, series: Series):
        self.config = config
        self.series = series

        self.index_dir = series.index_dir / "annoy.index"
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class AnnoySubtitleIndexWriter(AnnoySubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)

        self.embeddings_dir = self.index_dir / "embeddings"

    def index_series(self):
        episodes = {(ep.season_number, ep.episode_number)
                    for ep in get_specified_episodes(self.config, self.series)}

        subtitle_files = list(self.series.dir.rglob("*.srt"))

        with Progress() as progress, ThreadPoolExecutor(max_workers=10) as executor:
            # annoy indexes must be rebuilt from scratch
            if self.index_dir.exists():
                logger.info(f"Rebuilding index for: {self.series.name}, "
                            f"deleting: {self.index_dir}")
                index_files = [file for file in self.index_dir.iterdir()
                               if file.suffix in {".ann", ".json"}
                               and file.is_file()]
                delete_progress = progress.add_task(
                    f"Removing index for: {self.series.name}",
                    total=len(index_files))
                for index_file in index_files:
                    index_file.unlink()
                    progress.update(delete_progress, advance=1)
                progress.remove_task(delete_progress)

            extract_progress = progress.add_task(
                f"Extracting embeddings for {self.series.name} "
                f"({self.series.dir})", total=len(subtitle_files))

            def extract_embeddings(file):
                logger.info(f"Indexing: {file}")
                episode = episode_from_path(file)
                logger.info(f"Identified: {file} as episode: {episode}")
                if episode in episodes:
                    logger.info(f"Indexing: {file} as episode: {episode}")
                    self.extract_embeddings(file, episode)
                progress.update(extract_progress, advance=1)

            list(executor.map(extract_embeddings, subtitle_files))
            progress.remove_task(extract_progress)

            interval_dirs = [dir for dir in self.embeddings_dir.iterdir()
                             if dir.is_dir()]
            index_progress = progress.add_task(
                f"Indexing {self.series.name} ({self.series.dir})",
                total=len(interval_dirs))
            def index_interval(dir):
                logger.info(f"Indexing {dir}")
                self.build_interval_index(dir)
                progress.update(index_progress, advance=1)

            list(executor.map(index_interval, interval_dirs))
            progress.remove_task(index_progress)

    def extract_embeddings(self, path, episode: tuple[int, int]):
        epi_str = episode_str(*episode)
        logger.info(f"Indexing episode: {self.series.name} episode: {epi_str}")

        sub_file = pysubs2.load(path, format_="srt")
        metadata = self.series.get_episode_detail(episode)

        # TODO maybe this should use the runtime from the subtitle file?
        interval_count = math.ceil(metadata["runtime"] * 60 / 30)
        intervals = ((i, range(i * 30 * 1000, (i + 1) * 30 * 1000))
                     for i in range(interval_count))

        for index, interval in intervals:
            subs = [sub.plaintext for sub in sub_file
                    if sub.start in interval or sub.end in interval]
            # Don't index empty intervals
            if not subs:
                continue

            interval_text = " ".join(subs)

            interval_dir = self.embeddings_dir / str(index)
            if not interval_dir.exists():
                interval_dir.mkdir(parents=True, exist_ok=True)
            text_file = interval_dir / f"{epi_str}.txt"
            if not text_file.exists():
                with open(text_file, "w") as text_out:
                    text_out.write(interval_text)

            embeddings_file = interval_dir / f"{epi_str}.npy"
            if not embeddings_file.exists():
                embeddings = self.model.encode_document(interval_text)
                np.save(embeddings_file, embeddings)

    def build_interval_index(self, interval_dir: Path):
        embeddings = list(interval_dir.glob("*.npy"))
        embeddings.sort(key=lambda f: f.stem)

        index_directory = {index: episode_tuple(f.stem)
                            for index, f in enumerate(embeddings)}

        interval = interval_dir.stem
        with open(self.index_dir / f"{interval}.json", "w") as json_out:
            json.dump(index_directory, json_out)

        index = AnnoyIndex(self.model.get_sentence_embedding_dimension(),
                           "angular")
        for episode_index, file in enumerate(embeddings):
            with open(file, "rb") as f:
                embeddings = np.load(f)

            logger.info(f"Indexing: {file} -> {interval}[{episode_index}]")
            index.add_item(episode_index, embeddings)

        # we're already parallelizing the build, so don't use more threads'
        index.build(50, n_jobs=1)
        index.save(str(self.index_dir / f"{interval}.ann"))
        index.unload()

class AnnoySubtitleIndexReader(AnnoySubtitleIndex):
    def __init__(self, config, series: Series):
        super().__init__(config, series)

        self.indexes = self.load_indexes()

    def query_intervals(self, text_segments: list[tuple[int, str]]) -> list[tuple[tuple[float, int], str, str]]:
        distances_by_episode = {}
        for interval, text in text_segments:
            index_entry = self.indexes.get(interval)
            if index_entry is None:
                logger.warning(f"No index found for interval: {interval}")
                continue

            directory, index = index_entry
            query = self.model.encode_query(text)
            ids, distances = index.get_nns_by_vector(query, 5, include_distances=True)
            logger.info(f"Query: {interval} -> {ids} -> {distances}")
            for id, distance in zip(ids, distances):
                episode_id = directory[id]
                if episode_id in distances_by_episode:
                    cur = distances_by_episode[episode_id]
                    distances_by_episode[episode_id] = (min(cur[0], distance),
                                                        cur[1] + 1)
                else:
                    distances_by_episode[episode_id] = (distance, 1)

        ordered_ep_id = sorted(distances_by_episode,
                               # Order by frequency DESCENDING, distance ASCENDING
                               key=lambda k: (-distances_by_episode[k][1],
                                              distances_by_episode[k][0]))

        return [(distances_by_episode[ep_id], ep_id[0], ep_id[1])
                for ep_id in ordered_ep_id[:5]] # only return the top 5 results

    def load_indexes(self):
        if not self.index_dir.exists():
            console.print(f"[bold red]No index for series: {self.series.name}"
                          f" Use mkv-episode-matcher index-subs to build indexes")

        intervals = [int(file.stem) for file in self.index_dir.iterdir()
                     if file.is_file() and file.suffix == ".ann"]
        return {interval: self.get_index(interval) for interval in intervals}

    def get_index(self, interval: int) -> tuple[dict[int, tuple[int, int]], AnnoyIndex] | None:
        index_file = self.index_dir / f"{interval}.ann"
        directory_file = self.index_dir / f"{interval}.json"
        if not (index_file.exists() and directory_file.exists()):
            # This might happen if there's a mismatch between the video file
            # lengths and the episode metadata.
            logger.warning(
                f"Incomplete or missing index for interval: {interval}: "
                f"f{index_file} and/or {directory_file} not found.")
            return None

        logger.info(f"Loading index for interval: {interval}: {index_file}")
        index = AnnoyIndex(self.model.get_sentence_embedding_dimension(), "angular")
        index.load(str(index_file))
        logger.info(f"Loading directory for interval: {interval}: {directory_file}")
        with open(directory_file, "r") as json_in:
            data = json.load(json_in)
            index_directory = {int(id): (season, episode)
                               for id, (season, episode) in data.items()}

        return index_directory, index
