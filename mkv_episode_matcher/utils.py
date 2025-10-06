# utils.py
import os
import re
import shutil
from pathlib import Path

import requests
import torch
from loguru import logger
from opensubtitlescom import OpenSubtitles
from opensubtitlescom.exceptions import OpenSubtitlesException
from rich.console import Console
from rich.panel import Panel

from mkv_episode_matcher.__main__ import CACHE_DIR, CONFIG_FILE
from mkv_episode_matcher.config import get_config
from mkv_episode_matcher.subtitle_utils import find_existing_subtitle, sanitize_filename
from mkv_episode_matcher.tmdb_client import fetch_season_details

console = Console()


def normalize_path(path_str):
    """
    Normalize a path string to handle cross-platform path issues.
    Properly handles trailing slashes and backslashes in both Windows and Unix paths.

    Args:
        path_str (str): The path string to normalize

    Returns:
        pathlib.Path: A normalized Path object
    """
    # Convert to string if it's a Path object
    if isinstance(path_str, Path):
        path_str = str(path_str)

    # Remove trailing slashes or backslashes
    path_str = path_str.rstrip("/").rstrip("\\")

    # Handle Windows paths on non-Windows platforms
    if os.name != "nt" and "\\" in path_str and ":" in path_str[:2]:
        # This looks like a Windows path on a non-Windows system
        # Extract the last component which should be the directory/file name
        components = path_str.split("\\")
        return Path(components[-1])

    return Path(path_str)

def rename_episode_file(original_file_path, new_filename):
    """
    Rename an episode file with a standardized naming convention.

    Args:
        original_file_path (str or Path): The original file path of the episode.
        new_filename (str or Path): The new filename including season/episode info.

    Returns:
        Path: Path to the renamed file, or None if rename failed.
    """
    original_dir = Path(original_file_path).parent
    new_file_path = original_dir / new_filename

    # Check if new filepath already exists
    if new_file_path.exists():
        logger.warning(f"File already exists: {new_filename}")

        # Add numeric suffix if file exists
        base, ext = Path(new_filename).stem, Path(new_filename).suffix
        suffix = 2
        while True:
            new_filename = f"{base}_{suffix}{ext}"
            new_file_path = original_dir / new_filename
            if not new_file_path.exists():
                break
            suffix += 1

    try:
        Path(original_file_path).rename(new_file_path)
        logger.info(f"Renamed {Path(original_file_path).name} -> {new_filename}")
        return new_file_path
    except OSError as e:
        logger.error(f"Failed to rename file: {e}")
        return None
    except FileExistsError as e:
        logger.error(f"Failed to rename file: {e}")
        return None


def get_subtitles(show_id, seasons: set[int], config=None, max_retries=3):
    """
    Retrieves and saves subtitles for a given TV show and seasons.

    Args:
        show_id (int): The ID of the TV show.
        seasons (Set[int]): A set of season numbers for which subtitles should be retrieved.
        config (Config object, optional): Preloaded configuration.
        max_retries (int, optional): Number of times to retry subtitle download on OpenSubtitlesException. Defaults to 3.
    """
    if config is None:
        config = get_config(CONFIG_FILE)
    show_dir = config.get("show_dir")
    series_name = sanitize_filename(normalize_path(show_dir).name)
    tmdb_api_key = config.get("tmdb_api_key")
    open_subtitles_api_key = config.get("open_subtitles_api_key")
    open_subtitles_user_agent = config.get("open_subtitles_user_agent")
    open_subtitles_username = config.get("open_subtitles_username")
    open_subtitles_password = config.get("open_subtitles_password")

    if not all([
        show_dir,
        tmdb_api_key,
        open_subtitles_api_key,
        open_subtitles_user_agent,
        open_subtitles_username,
        open_subtitles_password,
    ]):
        logger.error("Missing configuration settings. Please run the setup script.")
        return

    try:
        subtitles = OpenSubtitles(open_subtitles_user_agent, open_subtitles_api_key)
        subtitles.login(open_subtitles_username, open_subtitles_password)
    except Exception as e:
        logger.error(f"Failed to log in to OpenSubtitles: {e}")
        return

    for season in seasons:
        episodes = fetch_season_details(show_id, season)
        logger.info(f"Found {episodes} episodes in Season {season}")

        for episode in range(1, episodes + 1):
            logger.info(f"Processing Season {season}, Episode {episode}...")

            series_cache_dir = get_series_cache_path(series_name)

            # Check for existing subtitle in any supported format
            existing_subtitle = find_existing_subtitle(
                series_cache_dir, series_name, season, episode
            )

            if existing_subtitle:
                logger.info(f"Subtitle already exists: {Path(existing_subtitle).name}")
                continue

            # Default to standard format for new downloads
            srt_filepath = str(
                series_cache_dir / f"{series_name} - S{season:02d}E{episode:02d}.srt"
            )

            # get the episode info from TMDB
            url = f"https://api.themoviedb.org/3/tv/{show_id}/season/{season}/episode/{episode}?api_key={tmdb_api_key}"
            response = requests.get(url)
            response.raise_for_status()
            episode_data = response.json()
            episode_id = episode_data["id"]

            # search for the subtitle
            response = subtitles.search(tmdb_id=episode_id, languages="en")
            if len(response.data) == 0:
                logger.warning(
                    f"No subtitles found for {series_name} - S{season:02d}E{episode:02d}"
                )
                continue

            for subtitle in response.data:
                subtitle_dict = subtitle.to_dict()
                # Remove special characters and convert to uppercase
                filename_clean = re.sub(
                    r"\\W+", " ", subtitle_dict["file_name"]
                ).upper()
                if f"E{episode:02d}" in filename_clean:
                    logger.info(f"Original filename: {subtitle_dict['file_name']}")
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            srt_file = subtitles.download_and_save(subtitle)
                            shutil.move(srt_file, srt_filepath)
                            logger.info(f"Subtitle saved to {srt_filepath}")
                            opensubs_filepath = srt_filepath + ".opensubtitles"
                            opensubs_json = subtitle.to_json()
                            with open(opensubs_filepath, "w") as json_out:
                                json_out.write(opensubs_json)
                            logger.info(f"Subtitle metadata saved to {opensubs_filepath}")
                            break
                        except OpenSubtitlesException as e:
                            retry_count += 1
                            logger.error(
                                f"OpenSubtitlesException (attempt {retry_count}): {e}"
                            )
                            console.print(
                                f"[red]OpenSubtitlesException (attempt {retry_count}): {e}[/red]"
                            )
                            if retry_count >= max_retries:
                                user_input = input(
                                    "Would you like to continue matching? (y/n): "
                                )
                                if user_input.strip().lower() != "y":
                                    logger.info(
                                        "User chose to stop matching due to the error."
                                    )
                                    return
                                else:
                                    logger.info(
                                        "User chose to continue matching despite the error."
                                    )
                                    break
                        except Exception as e:
                            logger.error(f"Failed to download and save subtitle: {e}")
                            console.print(
                                f"[red]Failed to download and save subtitle: {e}[/red]"
                            )
                            user_input = input(
                                "Would you like to continue matching despite the error? (y/n): "
                            )
                            if user_input.strip().lower() != "y":
                                logger.info(
                                    "User chose to stop matching due to the error."
                                )
                                return
                            else:
                                logger.info(
                                    "User chose to continue matching despite the error."
                                )
                                break
                    else:
                        continue
                    break


def get_series_cache_path(series_name: str) -> Path:
    series_cache_dir = Path(CACHE_DIR) / "data" / series_name
    os.makedirs(series_cache_dir, exist_ok=True)
    return series_cache_dir


def clean_text(text):
    # Remove brackets, parentheses, and their content
    cleaned_text = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", text)
    # Strip leading/trailing whitespace
    return cleaned_text.strip()

def extract_season_episode(filename):
    """
    Extract season and episode numbers from filename with support for multiple formats.

    Args:
        filename (str): Filename to parse

    Returns:
        tuple: (season_number, episode_number)
    """
    # List of patterns to try
    patterns = [
        r"S(\d+)E(\d+)",  # S01E01
        r"(\d+)x(\d+)",  # 1x01 or 01x01
        r"Season\s*(\d+).*?(\d+)",  # Season 1 - 01
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))

    return None, None

def compare_and_rename_files(srt_files, reference_files, dry_run=False):
    """
    Compare the srt files with the reference files and rename the matching mkv files.

    Args:
        srt_files (dict): A dictionary containing the srt files as keys and their contents as values.
        reference_files (dict): A dictionary containing the reference files as keys and their contents as values.
        dry_run (bool, optional): If True, the function will only log the renaming actions without actually renaming the files. Defaults to False.
    """
    logger.info(
        f"Comparing {len(srt_files)} srt files with {len(reference_files)} reference files"
    )
    for srt_text in srt_files.keys():
        parent_dir = Path(srt_text).parent.parent
        for reference in reference_files.keys():
            _season, _episode = extract_season_episode(reference)
            mkv_file = str(parent_dir / Path(srt_text).name.replace(".srt", ".mkv"))
            matching_lines = compare_text(
                reference_files[reference], srt_files[srt_text]
            )
            if matching_lines >= int(len(reference_files[reference]) * 0.1):
                logger.info(f"Matching lines: {matching_lines}")
                logger.info(f"Found matching file: {mkv_file} ->{reference}")
                new_filename = parent_dir / reference
                if not dry_run:
                    logger.info(f"Renaming {mkv_file} to {str(new_filename)}")
                    rename_episode_file(mkv_file, reference)


def compare_text(text1, text2):
    """
    Compare two lists of text lines and return the number of matching lines.

    Args:
        text1 (list): List of text lines from the first source.
        text2 (list): List of text lines from the second source.

    Returns:
        int: Number of matching lines between the two sources.
    """
    # Flatten the list of text lines
    flat_text1 = [line for lines in text1 for line in lines]
    flat_text2 = [line for lines in text2 for line in lines]

    # Compare the two lists of text lines
    matching_lines = set(flat_text1).intersection(flat_text2)
    return len(matching_lines)
