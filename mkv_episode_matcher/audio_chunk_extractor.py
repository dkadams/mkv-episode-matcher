import os
import subprocess
import tempfile
from pathlib import Path
from typing import ContextManager

import numpy as np
from loguru import logger


class AudioChunkExtractor(ContextManager):
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "mkv-episode-matcher-audio-chunks"
        self.temp_dir.mkdir(exist_ok=True)

        self.audio_chunks = set()

    @staticmethod
    def get_video_duration(file: Path):
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"

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
            ], env=env).decode()
        )
        return int(np.ceil(duration))

    def extract(self, file: Path, start_time: int, duration: int) -> Path:

        chunk_name = file.with_suffix(f".{duration}S.AT{start_time}s.wav").name
        chunk_path = self.temp_dir / chunk_name

        if not chunk_path.exists():
            cmd = [
                "ffmpeg",
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                file,
                "-vn",  # Disable video
                "-sn",  # Disable subtitles
                "-dn",  # Disable data streams
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",  # Overwrite output files without asking
                str(chunk_path),
            ]
            subprocess.run(cmd, capture_output=True)
            self.audio_chunks.add(chunk_path)

        return chunk_path

    def __exit__(self, exc_type, exc_value, traceback, /):
        for chunk in self.audio_chunks:
            try:
                chunk.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {chunk}: {e}")
