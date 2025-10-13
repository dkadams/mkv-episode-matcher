from rich.prompt import Prompt
from rich.table import Table
from rich.console import Console

console = Console()

def print_series_results(page: int, results, series_name, total_pages):
    table = Table(
        title=f"Search results for '{series_name}' (page {page} of {total_pages})")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("First Air Date", style="magenta")
    table.add_column("Overview", style="")

    for idx, item in enumerate(results, start=1):
        name = item.get("name") or item.get("original_name") or ""
        first_air_date = item.get("first_air_date") or ""
        overview = item.get("overview") or ""
        table.add_row(str(idx), name, first_air_date, overview)

    console.print(table)


def series_id_prompt(page: int, results, total_pages) -> str:
    # Build prompt choices
    choices = [str(i) for i in range(1, len(results) + 1)]
    show_paging = total_pages and total_pages > 1
    if show_paging and page > 1:
        choices.append("prev")
    if show_paging and page < total_pages:
        choices.append("next")

    # Ask for selection or navigation
    selection = Prompt.ask(
        "Select a series by number" + (
            " or navigate" if len(choices) > len(results) else ""),
        choices=choices,
        show_choices=True,
    )
    return selection
