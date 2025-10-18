from guessit import guessit
from loguru import logger
from rich.console import Console
from rich.table import Table

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import episode_str
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.subtitle_index import FullEpisodeSubtitleIndex
from mkv_episode_matcher.text_chunk_extractor import TextChunkExtractor

console = Console()

class IndexedEpisodeMatcher:
    def __init__(self, config: Configuration, series: Series):
        self.config = config
        self.series = series
        self.index = FullEpisodeSubtitleIndex(config, series)
        self.text_extractor = TextChunkExtractor(30, 10, "small.en")

    def match(self, path):
        table = Table(title=f"Matches for '{self.series.name}'")
        table.add_column("Filename")
        table.add_column("Episode Id")
        table.add_column("# Matches", style="bold")
        table.add_column("#1", style="magenta")
        table.add_column("#2")
        table.add_column("#3")
        table.add_column("#4")
        table.add_column("#5")

        files = [path] if path.is_file() else path.rglob('**/*.mkv')
        for file in files:
            logger.info(f"Processing file: {file}")

            info = guessit(file.name)
            actual = episode_str(info.get("season"), info.get("episode")) if info else "-"

            matches = [f"{episode_str(m[1], m[2])} - {m[0]:.2}"
                       for m in self.match_file(file)]
            if len(matches) < 5:
                matches.extend(["-"] * (5 - len(matches)))
            table.add_row(file.name,
                          actual,
                          str(len(matches)),
                          matches[0],
                          matches[1],
                          matches[2],
                          matches[3],
                          matches[4])

        console.print(table)

    def match_file(self, file):
        text = self.text_extractor.get_text(file)
        result = self.index.full_episodes.query(query_texts=[text],
                                                n_results=5,
                                                include=["metadatas", "distances"])

        return [(distance, md["season_number"], md["episode_number"])
                for md, distance
                in zip(result["metadatas"][0], result["distances"][0])]

