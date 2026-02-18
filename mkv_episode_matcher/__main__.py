# __main__.py (enhanced version)
import os
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console

from mkv_episode_matcher.args import build_args_parser
from mkv_episode_matcher.config import get_config, resolve_log_dir

# Disable Hugging Face tokenizers' internal multithreading. This is a transitive
# dependency of SentenceTransformers used for embedding, and the original Whisper
# implementation.
# This avoids "process just got forked" warnings and potential deadlocks
# when the program forks subprocesses (e.g., ffmpeg, whisper-cli).
# The text volume tokenized here is small, so parallelism offers little benefit.
# Performance impact has not been formally benchmarked but is expected to be negligible
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize rich console for better output
console = Console()

def configure_bootstrap_logging():
    """Setup temporary logging to stderr before args/config are available."""
    logger.remove()
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")


def configure_file_logging(log_dir: Path) -> Path:
    """Switch logging from stderr to per-run log files."""
    resolved = Path(log_dir).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        str(resolved / "stdout-{time:YYYYMMDDTHHmmss}.log"),
        format="{time} {level} {message}",
        level="INFO",
        retention=10,
    )
    logger.add(
        str(resolved / "stderr-{time:YYYYMMDDTHHmmss}.log"),
        level="ERROR",
        retention=10,
    )

    return resolved

@logger.catch
def main():
    """
    Entry point of the application.
    """

    parser = build_args_parser()
    args = parser.parse_args()

    try:
        log_dir = configure_file_logging(resolve_log_dir(args))
        logger.info(f"Using log directory: {log_dir}")
    except OSError as e:
        logger.warning(
            f"Failed to initialize log directory from configuration: {e}. "
            "Continuing with stderr logging only."
        )

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
    configure_bootstrap_logging()

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
