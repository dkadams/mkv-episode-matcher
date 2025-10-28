import math
from concurrent.futures.thread import ThreadPoolExecutor

import chromadb
import pysubs2
from loguru import logger
from rich.console import Console
from rich.progress import Progress

from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.indexed_episode_matcher import Match, Score
from mkv_episode_matcher.series import Series, get_specified_episodes

console = Console()

class ChromaSubtitleIndex:
    def __init__(self, config, series: Series):
        self.config = config
        self.series = series

        self.index_dir = series.index_dir / "chroma.index"
        self.chromadb = chromadb.PersistentClient(path=self.index_dir)

        self.intervals = self.chromadb.get_or_create_collection(name="intervals",
                                                               metadata={"hnsw:space": "cosine"})

class ChromaSubtitleIndexWriter(ChromaSubtitleIndex):
    def index_series(self):
        episodes = {(ep.season_number, ep.episode_number)
                    for ep in get_specified_episodes(self.config, self.series)}

        subtitle_files = list(self.series.dir.rglob("*.srt"))

        with Progress() as progress, ThreadPoolExecutor(max_workers=10) as executor:
            task = progress.add_task(f"Indexing {self.series.name} ({self.series.dir})",
                                     total=len(subtitle_files))

            def index_file(file):
                logger.info(f"Indexing: {file}")
                episode = EpisodeKey.from_path(file)
                logger.info(f"Identified: {file} as episode: {episode}")
                if episode in episodes:
                    logger.info(f"Indexing: {file} as episode: {episode}")
                    self.index_episode(file, episode)
                progress.update(task, advance=1)

            list(executor.map(index_file, subtitle_files))

    def index_episode(self, path, episode: EpisodeKey):
        logger.info(f"Indexing episode: {self.series.name} episode: {episode}")

        sub_file = pysubs2.load(path, format_="srt")

        metadata = self.series.get_episode_detail(episode)
        interval_count = math.ceil(metadata["runtime"] * 60 / 30)
        intervals = (range(i * 30 * 1000, (i + 1) * 30 * 1000)
                     for i in range(interval_count))
        ids = []
        documents = []
        metadatas = []
        for interval in intervals:
            subs = [sub.plaintext for sub in sub_file
                    if sub.start in interval or sub.end in interval]

            ids.append(f"{str(path)}:{interval.start}:{interval.stop}")
            documents.append("\n".join(subs))
            sub_metadata = metadata.copy()
            sub_metadata["start_ms"] = interval.start
            sub_metadata["end_ms"] = interval.stop
            metadatas.append(sub_metadata)
        self.intervals.upsert(ids=ids, documents=documents, metadatas=metadatas)


class ChromaSubtitleIndexReader(ChromaSubtitleIndex):
    def query_intervals(self, text_segments: list[tuple[int, str]]) -> list[Match]:
        scores_by_episode: dict[EpisodeKey, Score] = {}
        for interval, text in text_segments:
            start_ms = interval * 30 * 1000
            result = self.intervals.query(query_texts=[text],
                                          where={"start_ms": start_ms},
                                          n_results=5,
                                          include=["metadatas", "distances"])

            for md, distance in zip(result["metadatas"][0],
                                    result["distances"][0]):
                key = EpisodeKey(md["season_number"], md["episode_number"])

                cur = scores_by_episode.setdefault(key, Score(0, 1000000))
                score = Score(cur.count + 1, min(cur.min_distance, distance))
                scores_by_episode[key] = score


        ordered_ep_id = sorted(scores_by_episode,
                               key=lambda k: scores_by_episode[k])

        return [Match(ep_id[0], ep_id[1], scores_by_episode[ep_id])
                for ep_id in ordered_ep_id[:5]] # only return the top 5 results

ChromaSubtitleIndex.reader_type = ChromaSubtitleIndexReader
ChromaSubtitleIndex.writer_type = ChromaSubtitleIndexWriter
