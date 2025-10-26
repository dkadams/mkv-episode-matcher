from rich.console import Console

from mkv_episode_matcher.series import SeriesDirectoryProcessor

console = Console()

def index_subtitles(config):
    def subtitles_indexer(series):
        index_cls = config.args.index_type.writer_type
        console.print(f"[bold green]Indexing subtitles for: {series.name}, type: {index_cls.__name__}")
        index_writer = index_cls(config, series)

        index_writer.index_series()
        console.print(f"[bold green]Indexing complete for: {series.name}")

    SeriesDirectoryProcessor(config).process_series(subtitles_indexer)
