import subprocess
from pathlib import Path


def get_video_duration_seconds(file: Path) -> float:
    duration = float(
        subprocess.check_output([
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file,
        ]).decode()
    )
    return duration
