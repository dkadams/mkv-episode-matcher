# utils.py
import os
import re
from pathlib import Path

from loguru import logger
from rich.console import Console

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
