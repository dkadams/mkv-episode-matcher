import json
from pathlib import Path

from rich.console import Console
from loguru import logger

from mkv_episode_matcher.series import SERIES_DEFAULT_SETTINGS
from mkv_episode_matcher.series_ui import print_series_results, series_id_prompt
from mkv_episode_matcher.tmdb_client import fetch_series_detail, search_series, \
    fetch_season_details
from mkv_episode_matcher.windowing import make_window_config

console = Console()

def init_series(config):
    series_dirs = [Path(dir).resolve() for dir in config.args.series_dirs]
    for series_dir in series_dirs:
        console.print(f"[bold green]Initializing Series: {series_dir}[/bold green]")
        init = SeriesInitializer(config, series_dir)
        init.init_tmdb()
        console.print(f"[bold green]Series initialized: {series_dir}[/bold green]")

class SeriesInitializer:
    def __init__(self, config, series_dir):
        self.config = config

        self.series_dir = series_dir
        self.series_dot_dir = self.series_dir / ".mkv-episode-matcher"

        self.series_name = self.config.args.series_name or self.series_dir.name

    def init_tmdb(self):
        logger.info(f"Initializing series: {self.series_dir}, series_name: {self.series_name}")

        series_id = self.config.args.series_id or self.get_series_id()
        if not series_id:
            console.print(f"[red]No series id for series: {self.series_name}")
            return

        series_detail = fetch_series_detail(self.config, series_id)
        seasons = [season["season_number"] for season in series_detail["seasons"]]

        series_file = self.series_dot_dir / "series.tmdb.json"
        if series_file.exists() and not self.config.args.refresh:
            console.print(f"[bold orange1]Series data already initialized: {self.series_dir}. Use --refresh to re-initialize.")
            return
        else:
            # The season detail response **includes** the series detail, so we write
            # it as the series detail.
            seasons_detail = fetch_season_details(self.config, series_id, seasons)

            self.series_dot_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing series detail to {series_file}")
            with open(series_file, "w", encoding="utf-8") as out:
                json.dump(seasons_detail, out)

        series_settings = SERIES_DEFAULT_SETTINGS.copy()
        if self.config.args.segment_duration:
            series_settings["segment_duration"] = self.config.args.segment_duration
        if self.config.args.subtitle_overlap_seconds is not None:
            series_settings["subtitle_overlap_seconds"] = self.config.args.subtitle_overlap_seconds
        if self.config.args.random_seed:
            series_settings["random_seed"] = self.config.args.random_seed
        make_window_config(
            int(series_settings["segment_duration"]),
            int(series_settings["subtitle_overlap_seconds"]),
        )

        series_settings_file = self.series_dot_dir / "settings.json"
        logger.info(f"Writing series settings to {series_settings_file}")
        with open(series_settings_file, "w", encoding="utf-8") as out:
            json.dump(series_settings, out)


    def get_series_id(self):
        """
        Search for a TV series and allow the user to select one result.

        - Displays a paginated table (name, first_air_date, overview) using rich.table.Table
        - Uses rich.prompt.Prompt to choose an entry by sequence number
        - Supports pagination via 'next' and 'prev' when there are multiple pages
        """
        page = 1
        pages = {}
        while True:
            response = pages.get(page) or search_series(self.config, self.series_name, page)
            pages[page] = response
            if not (response and response.get("results") and response.get("total_results")):
                console.print(f"[red]Series: {self.series_name} not found")
                return None

            total_results = response.get("total_results", 0)
            results = response.get("results", [])
            total_pages = response.get("total_pages", 1)

            if total_results == 0 or not results:
                console.print(f"[red]Series: {self.series_name} not found")
                return None

            # If there is exactly one result, return it immediately
            if total_results == 1:
                return results[0]["id"]

            print_series_results(page, results, self.series_name, total_pages)

            selection = series_id_prompt(page, results, total_pages)
            if selection == "next":
                page += 1
                continue
            elif selection == "prev":
                page -= 1
                continue
            else:
                pick = int(selection)
                if 1 <= pick <= len(results):
                    return results[pick - 1]["id"]
                # Should not reach here due to choices validation, but guard anyway
                console.print("[red]Invalid selection. Please try again.")
