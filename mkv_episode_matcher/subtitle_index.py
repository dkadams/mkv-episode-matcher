import glob
import json
from pathlib import Path
from typing import Optional

import chromadb
import pysubs2
from guessit import guessit
from loguru import logger

from rich.console import Console

from mkv_episode_matcher.episode import get_specified_episodes
from mkv_episode_matcher.series import SeriesDirectoryProcessor, Series

console = Console()

def index_subtitles(config):
    def subtitles_indexer(series):
        console.print(f"[bold green]Indexing subtitles for: {series.name}")
        FullEpisodeSubtitleIndex(config, series).index_series()
        console.print(f"[bold green]Indexing complete for: {series.name}")

    SeriesDirectoryProcessor(config).process_series(subtitles_indexer)


class FullEpisodeSubtitleIndex:
    def __init__(self, config, series: Series):
        self.config = config
        self.series = series

        db_path = series.dot_dir / 'indexes.chromadb'
        self.chromadb = chromadb.PersistentClient(path=db_path)

        self.full_episodes = self.chromadb.get_or_create_collection(name="full-episodes",
                                                                    metadata={"hnsw:space": "cosine"})

    def upsert(self, path, episode: tuple[int, int]):
        sub_file = pysubs2.load(path, format_="srt")
        full_episode = "\n".join([line.plaintext for line in sub_file])

        season_number = episode[0]
        episode_number = episode[1]
        season_detail = self.series.detail[f"season/{season_number}"]
        episode_detail = next((ep for ep in season_detail["episodes"]
                                if ep["episode_number"] == episode_number),
                              None)
        metadata = {k: episode_detail[k] for k in ["id", "season_number",
                                                   "episode_number", "runtime"]}
        logger.info(f"Found episode detail: {episode_detail}/{type(episode_detail)}")
        if not episode_detail:
            raise ValueError(f"Error: No episode detail found for "
                              f"{self.series.name} season: {season_number} "
                              f"episode: {episode_number}")

        logger.info(f"Indexing episode: {self.series.name} season: {season_number} episode: {episode_number}")
        self.full_episodes.upsert(ids=[str(path)],
                                  documents=[full_episode],
                                  metadatas=[metadata])

    def index_series(self):
        episodes = {(ep.season_number, ep.episode_number)
                    for ep in get_specified_episodes(self.config, self.series)}

        for file in self.series.dir.rglob("*.srt"):
            logger.info(f"Found file: {file}")

            episode = self.get_episode(file)
            logger.info(f"Found subtitle: {file}, episode: {episode}")
            if episode in episodes:
                self.upsert(file, episode)

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


