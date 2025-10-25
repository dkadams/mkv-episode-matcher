# __main__.py (enhanced version)
import sys

from loguru import logger
from rich.console import Console

from mkv_episode_matcher.args import build_args_parser
from mkv_episode_matcher.config import CONFIG_DIR, get_config

# Initialize rich console for better output
console = Console()

# Check if logs directory exists, if not create it
log_dir = CONFIG_DIR / "logs"
if not log_dir.exists():
    log_dir.mkdir(exist_ok=True)
logger.remove()
# Add a new handler for stdout logs
logger.add(
    str(log_dir / "stdout.log"),
    format="{time} {level} {message}",
    level="INFO",
    rotation="10 MB",
)

# Add a new handler for error logs
logger.add(str(log_dir / "stderr.log"), level="ERROR", rotation="10 MB")

@logger.catch
def main():
    """
    Entry point of the application.
    """

    parser = build_args_parser()
    args = parser.parse_args()

    logger.info(f"Command-line arguments: {args}")
    if args.verbose:
        console.print("[bold cyan]Command-line Arguments[/bold cyan]")
        console.print(args)

    if args.func:
        config = get_config(args)
        args.func(config)
    else:
        console.print("[bold red]No subcommand func[/bold red]")


# Run the main function if the script is run directly
if __name__ == "__main__":
    # Log the start of the application
    logger.info("Starting mkv-episode-matcher")

    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        logger.exception("Unhandled exception")
        sys.exit(1)
