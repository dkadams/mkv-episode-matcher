from rich.console import Console


from mkv_episode_matcher.__main__ import CONFIG_FILE
from mkv_episode_matcher.config import get_config
from mkv_episode_matcher.tmdb_client import fetch_show_id, get_number_of_seasons
from mkv_episode_matcher.utils import get_subtitles

console = Console()

class SubtitleDownloader:
    def __init__(self):
        self.config = get_config(CONFIG_FILE)

    def download(self, show_name):
        show_id = fetch_show_id(show_name)
        if not show_id:
            console.print(f"[bold red]Error:[/bold red] Could not find show ID for {show_name} on TMDB.")
            return None

        num_seasons = get_number_of_seasons(show_id)
        if not num_seasons:
            console.print(f"[bold red]Error:[/bold red] {num_seasons} for {show_name} on TMDB.")
            return None

        seasons = set(range(1, num_seasons + 1))
        get_subtitles(show_id, seasons=seasons, config=self.config)

        return True
