import subprocess
import tempfile
from pathlib import Path
from typing import ContextManager

from loguru import logger

from mkv_episode_matcher.utils import unique_filename


class AudioChunkExtractor(ContextManager):
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "mkv-episode-matcher-audio-chunks"
        self.temp_dir.mkdir(exist_ok=True)

        self.audio_chunks = set()

    @staticmethod
    def _effective_start_seconds(start_time: float) -> float:
        return max(0.0, float(start_time))

    def extract(self, file: Path, start_time: float, duration: int) -> Path:
        effective_start = self._effective_start_seconds(start_time)
        start_ms = int(round(effective_start * 1000))
        chunk_name = unique_filename(file, f".{duration}S.AT{start_ms}ms.wav")
        chunk_path = self.temp_dir / chunk_name

        if not chunk_path.exists():
            cmd = [
                "ffmpeg",
                "-ss",
                f"{effective_start:.3f}",
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
