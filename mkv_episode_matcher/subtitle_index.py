from rich.console import Console

from mkv_episode_matcher.annoy_subtitle_index import AnnoySubtitleIndexWriter
from mkv_episode_matcher.chroma_subtitle_index import ChromaSubtitleIndexWriter
from mkv_episode_matcher.hnswlib_subtitle_index import HnswlibSubtitleIndexWriter
from mkv_episode_matcher.series import SeriesDirectoryProcessor

console = Console()

def index_subtitles(config):
    def subtitles_indexer(series):
        console.print(f"[bold green]Indexing subtitles for: {series.name}, format: {config.args.index_format}")
        if config.args.index_format == "chroma":
            index_writer = ChromaSubtitleIndexWriter(config, series)
        elif config.args.index_format == "annoy":
            index_writer = AnnoySubtitleIndexWriter(config, series)
        elif config.args.index_format == "hnswlib":
            index_writer = HnswlibSubtitleIndexWriter(config, series)
        else:
            raise Exception(f"Unknown index format: {config.args.index_format}")

        index_writer.index_series()
        console.print(f"[bold green]Indexing complete for: {series.name}")

    SeriesDirectoryProcessor(config).process_series(subtitles_indexer)
