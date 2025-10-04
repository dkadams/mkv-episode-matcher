import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache

from loguru import logger
import numpy as np

class AudioChunkExtractor:
    def __init__(self, chunk_duration):
        self.chunk_duration = chunk_duration

        self.temp_dir = Path(tempfile.gettempdir()) / "whisper_chunks"
        self.temp_dir.mkdir(exist_ok=True)

        # Cache for extracted audio chunks
        self.audio_chunks = {}

    @lru_cache(maxsize=100)
    def get_video_duration(self, video_file):
        """Get video duration with caching."""
        duration = float(
            subprocess.check_output([
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_file,
            ]).decode()
        )
        return int(np.ceil(duration))

    def extract_audio_chunk(self, mkv_file, start_time):
        """Extract a chunk of audio from MKV file with caching."""
        cache_key = (str(mkv_file), start_time)

        if cache_key in self.audio_chunks:
            return self.audio_chunks[cache_key]

        chunk_path = self.temp_dir / f"chunk_{start_time}.wav"
        if not chunk_path.exists():
            cmd = [
                "ffmpeg",
                "-ss",
                str(start_time),
                "-t",
                str(self.chunk_duration),
                "-i",
                mkv_file,
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

        chunk_path_str = str(chunk_path)
        self.audio_chunks[cache_key] = chunk_path_str
        return chunk_path_str

    def cleanup(self):
        # Cleanup temp files - keep this limited to only files we know we created
        for chunk_info in self.audio_chunks.values():
            try:
                Path(chunk_info).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {chunk_info}: {e}")
