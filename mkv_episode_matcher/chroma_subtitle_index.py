import json
import math
import time
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import chromadb
import pysubs2
from guessit import guessit
from loguru import logger
from rich.console import Console
from rich.progress import Progress

from mkv_episode_matcher.episode import get_specified_episodes, episode_str
from mkv_episode_matcher.series import Series

console = Console()

class ChromaSubtitleIndex:
    def __init__(self, config, series: Series):
        self.config = config
        self.series = series

        self.chromadb = chromadb.PersistentClient(path=series.index_dir)

        self.full_episodes = self.chromadb.get_or_create_collection(name="full-episodes",
                                                                    metadata={"hnsw:space": "cosine"})
        self.segments = self.chromadb.get_or_create_collection(name="segments",
                                                               metadata={"hnsw:space": "cosine"})
        self.intervals = self.chromadb.get_or_create_collection(name="intervals",
                                                               metadata={"hnsw:space": "cosine"})

    def index_series(self):
        episodes = {(ep.season_number, ep.episode_number)
                    for ep in get_specified_episodes(self.config, self.series)}

        subtitle_files = list(self.series.dir.rglob("*.srt"))


        with Progress() as progress, ThreadPoolExecutor(max_workers=10) as executor:
            task = progress.add_task(f"Indexing {self.series.name} ({self.series.dir})",
                                     total=len(subtitle_files))

            def index_file(file):
                logger.info(f"Indexing: {file}")
                episode = self.get_episode(file)
                logger.info(f"Identified: {file} as episode: {episode}")
                if episode in episodes:
                    logger.info(f"Indexing: {file} as episode: {episode}")
                    self.index_episode(file, episode, progress)
                progress.update(task, advance=1)

            list(executor.map(index_file, subtitle_files))

    def index_episode(self, path, episode: tuple[int, int], progress: Progress):
        logger.info(f"Indexing episode: {self.series.name} episode: {episode_str(*episode)}")

        sub_file = pysubs2.load(path, format_="srt")

        metadata = self.get_metadata(episode)

        task = progress.add_task(f"Indexing {episode_str(*episode)}", total=2)
        self.index_full_episode(metadata, path, sub_file, episode)
        progress.update(task, advance=1)

        def update_progress(advance: float):
            progress.update(task, advance=advance)
        #self.index_segments(metadata, path, sub_file, episode, update_progress)

        self.index_intervals(metadata, path, sub_file, episode)
        progress.update(task, advance=1)
        progress.update(task, completed=True)
        progress.remove_task(task)

    def get_metadata(self, episode: tuple[int, int]) -> dict[str, str | int]:
        season_number, episode_number = episode
        season_detail = self.series.detail[f"season/{season_number}"]
        episode_detail = next((ep for ep in season_detail["episodes"]
                               if ep["episode_number"] == episode_number), None)
        if not episode_detail:
            raise ValueError(f"Error: No episode detail found for "
                             f"{self.series.name} "
                             f"episode: {episode_str(*episode)}")

        return {k: episode_detail[k] for k in ["id", "season_number",
                                               "episode_number", "runtime"]}

    def index_full_episode(self, metadata: dict[str, str | int], path: Path,
        sub_file: pysubs2.SSAFile, episode: tuple[int, int]):
        full_episode = "\n".join([line.plaintext for line in sub_file])

        self.upsert("full", self.full_episodes, episode,
                    ids=[str(path)], documents=[full_episode], metadatas=[metadata])


    def index_segments(self, metadata: dict[str, str | int], path: Path,
        sub_file: pysubs2.SSAFile, episode: tuple[int, int], upgdate_progress):
        ids = []
        documents = []
        metadatas = []
        for sub in sub_file:
            ids.append(f"{str(path)}:{sub.start}:{sub.end}")
            documents.append(sub.plaintext)
            sub_metadata = metadata.copy()
            sub_metadata["start_ms"] = sub.start
            sub_metadata["end_ms"] = sub.end
            metadatas.append(sub_metadata)

        chunk_size = 20
        chunk_count = len(ids) / chunk_size
        for chunk_number, (id_chunk, doc_chunk, meta_chunk) in enumerate(zip(
                chunked(ids, chunk_size),
                chunked(documents, chunk_size),
                chunked(metadatas, chunk_size))):
            self.upsert(f"segments[chunk#{chunk_number}]", self.segments, episode,
                        ids=id_chunk, documents=doc_chunk, metadatas=meta_chunk)
            upgdate_progress(1 / chunk_count)

    def index_intervals(self, metadata: dict[str, str | int], path: Path,
            sub_file: pysubs2.SSAFile, episode: tuple[int, int]):
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

        self.upsert(f"intervals", self.intervals, episode,
                    ids=ids, documents=documents, metadatas=metadatas)

    @staticmethod
    def upsert(name, collection, episode, ids, documents, metadatas):
        logger.info(f"Upserting {episode_str(*episode)} into {name}")
        before = time.time()
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        after = time.time()
        logger.info(f"Upserted {episode_str(*episode)} into {name} in {after - before} seconds")

    @staticmethod
    def get_episode(file: Path) -> Optional[tuple[int, int]]:
        def from_opensubs():
            opensubs_file = file.with_suffix(".opensubtitles")
            if not opensubs_file.exists():
                return None

            with open(opensubs_file, "r") as json_in:
                opensubs_data = json.load(json_in)

            season_number = int(opensubs_data["season_number"])
            episode_number = int(opensubs_data["episode_number"])
            return season_number, episode_number

        def from_guessit():
            matches = guessit(file.name)
            season_number = matches.get("season")
            episode_number = matches.get("episode")
            return season_number, episode_number

        return from_opensubs() or from_guessit()

    def query_intervals(self, text_segments: list[tuple[int, str]]) -> list[tuple[tuple[float, int], str, str]]:
        distances_by_episode = {}
        for start_ms, text in text_segments:
            result = self.intervals.query(query_texts=[text],
                                          where={"start_ms": start_ms},
                                          n_results=5,
                                          include=["metadatas", "distances"])

            for md, distance in zip(result["metadatas"][0],
                                    result["distances"][0]):
                episode_id = (md["season_number"], md["episode_number"])

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

    def query_full_text(self, text: str) -> list[tuple[float, str, str]]:
        result = self.full_episodes.query(query_texts=[text],
                                          n_results=5,
                                          include=["metadatas", "distances"])

        return [(distance, md["season_number"], md["episode_number"])
                for md, distance
                in zip(result["metadatas"][0], result["distances"][0])]

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]
