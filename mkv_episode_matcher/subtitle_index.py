from rich.console import Console

from mkv_episode_matcher.chroma_subtitle_index import ChromaSubtitleIndex
from mkv_episode_matcher.series import SeriesDirectoryProcessor

console = Console()

def index_subtitles(config):
    def subtitles_indexer(series):
        console.print(f"[bold green]Indexing subtitles for: {series.name}")
        ChromaSubtitleIndex(config, series).index_series()
        console.print(f"[bold green]Indexing complete for: {series.name}")

    SeriesDirectoryProcessor(config).process_series(subtitles_indexer)
