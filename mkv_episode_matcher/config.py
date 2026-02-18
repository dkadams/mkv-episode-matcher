# config.py
import configparser
from argparse import Namespace
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.prompt import Confirm, Prompt

# Check if the configuration directory exists, if not create it
CONFIG_DIR = Path.home() / ".mkv-episode-matcher"
if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Define the paths for the configuration file and cache directory
CONFIG_FILE = CONFIG_DIR / "config.ini"

console = Console()

API_CONFIG_KEYS = [
    "tmdb_api_key",
    "open_subtitles_api_key",
    "open_subtitles_user_agent",
    "open_subtitles_username",
    "open_subtitles_password"
]
DEFAULT_LOG_DIR = CONFIG_DIR / "logs"

@dataclass
class Configuration:
    args: Namespace
    stored: ConfigParser

    def has_required_settings(self):
        required = [self.stored.get("api", k) for k in API_CONFIG_KEYS]
        return all(required)

def get_config(args):
    config_file = get_config_file(args)
    return _get_config(config_file, args)

def _get_config(file, args):
    """
    Read and return the configuration from the specified file.

    Args:
        file (str): The path to the configuration file.
        args (Namespace): command line arguments produced by argsparse

    Returns:
        Configuration: the unified configuration data
    """
    config = read_config(file) or ConfigParser(interpolation=None)

    # Union the stored config with CLI parameters
    args_dict = vars(args)
    # Only override expected keys
    api_args_override = {k: v for k in API_CONFIG_KEYS
                         if (v := args_dict.get(k, None))}
    args_override = {
        "api": api_args_override
    }
    config.read_dict(args_override, source="command line args")

    return Configuration(args, config)

def edit_config(config):
    """Prompt user for all required config values, showing existing as defaults."""
    config_file = get_config_file(config.args)
    config = _get_config(config_file, config.args)

    def ask_with_default(prompt_text, key, description, secret=False):
        current = config.stored.get("api", key, fallback=None)
        if current:
            console.print(f"[cyan]{prompt_text}:[/cyan] {description}")
            console.print(f"Current value: [green]{mask_api_key(current) if secret else current}[/green]")
            if Confirm.ask("Use existing value?", default=True):
                return current
        return Prompt.ask(f"Enter your {key}", default=current or "")

    tmdb_api_key = ask_with_default("TMDb API key", "tmdb_api_key", "Used to lookup show and episode information. To get your API key, create an account at https://www.themoviedb.org/ and follow the instructions at https://developer.themoviedb.org/docs/getting-started", secret=True)
    open_subtitles_username = ask_with_default("OpenSubtitles Username", "open_subtitles_username", "Account username for OpenSubtitles. To create an account, visit https://www.opensubtitles.com/ then click 'Register'")
    open_subtitles_password = ask_with_default("OpenSubtitles Password", "open_subtitles_password", "Account password for OpenSubtitles", secret=True)
    open_subtitles_user_agent = ask_with_default("OpenSubtitles Consumer Name", "open_subtitles_user_agent", "Required for subtitle downloads. Go to https://www.opensubtitles.com/en/consumers, click 'New Consumer', give it a name, then click 'Save'")
    open_subtitles_api_key = ask_with_default("OpenSubtitles API key", "open_subtitles_api_key", "Required for subtitle downloads. Enter the API key linked with the OpenSubtitles Consumer that you created in the previous step.", secret=True)

    store_api_config(
        tmdb_api_key,
        open_subtitles_api_key,
        open_subtitles_user_agent,
        open_subtitles_username,
        open_subtitles_password,
        config_file,
    )
    console.print("[bold green]Configuration saved.[/bold green]")

def get_config_file(args):
    if args.config_file:
        return args.config_file
    else:
        return CONFIG_FILE


def resolve_log_dir(args):
    cli_log_dir = getattr(args, "log_dir", None)
    if cli_log_dir:
        return Path(cli_log_dir).expanduser()

    config_file = get_config_file(args)
    parser = read_config(config_file)
    if parser and parser.has_option("logging", "log_dir"):
        stored_log_dir = parser.get("logging", "log_dir").strip()
        if stored_log_dir:
            return Path(stored_log_dir).expanduser()

    return DEFAULT_LOG_DIR

def mask_api_key(key: str) -> str:
    """Mask the API key for display purposes."""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]

def store_api_config(
    tmdb_api_key,
    open_subtitles_api_key,
    open_subtitles_user_agent,
    open_subtitles_username,
    open_subtitles_password,
    file,
):
    """
    Sets the configuration values and writes them to a file.

    Args:
        tmdb_api_key (str): The API key for TMDB (The Movie Database).
        open_subtitles_api_key (str): The API key for OpenSubtitles.
        open_subtitles_user_agent (str): The user agent for OpenSubtitles.
        open_subtitles_username (str): The username for OpenSubtitles.
        open_subtitles_password (str): The password for OpenSubtitles.
        file (str): The path to the configuration file.

    Returns:
        None
    """
    config = read_config(file) or configparser.ConfigParser(interpolation=None)
    config["api"] = {
        "tmdb_api_key": str(tmdb_api_key),
        "open_subtitles_api_key": str(open_subtitles_api_key),
        "open_subtitles_user_agent": str(open_subtitles_user_agent),
        "open_subtitles_username": str(open_subtitles_username),
        "open_subtitles_password": str(open_subtitles_password),
    }
    logger.info(
        f"Setting config with to {config}"
    )
    with open(file, "w", encoding="utf-8") as configfile:
        config.write(configfile)

def read_config(file):
    """
    Read and return the configuration from the specified file.

    Args:
        file (str): The path to the configuration file.

    Returns:
        ConfigParser: The configuration parser.

    """
    logger.info(f"Loading config from {file}")
    parser = configparser.ConfigParser(interpolation=None)
    if not Path(file).exists():
        return None

    parser.read(file, encoding="utf-8")
    return parser
