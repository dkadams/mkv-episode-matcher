import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import ContextManager

from loguru import logger

from mkv_episode_matcher.utils import unique_filename


class AudioChunkExtractor(ContextManager):
    def __init__(self, manage_lifecycle: bool = True):
        self.temp_dir = Path(tempfile.gettempdir()) / "mkv-episode-matcher-audio-chunks"
        self.temp_dir.mkdir(exist_ok=True)

        self.manage_lifecycle = manage_lifecycle
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
            temp_chunk_path = self.temp_dir / (
                f"{chunk_name}.tmp.{os.getpid()}.{uuid.uuid4().hex}.wav"
            )
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
                str(temp_chunk_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                temp_chunk_path.unlink(missing_ok=True)
                stderr = (result.stderr or "").strip()
                stderr = stderr[:1000] + ("...[truncated]" if len(stderr) > 1000 else "")
                raise RuntimeError(
                    f"ffmpeg failed extracting chunk from {file} at {effective_start:.3f}s "
                    f"for {duration}s: {stderr}"
                )
            if not temp_chunk_path.exists() or temp_chunk_path.stat().st_size == 0:
                temp_chunk_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"ffmpeg reported success but produced no usable output: {temp_chunk_path}"
                )
            try:
                os.replace(temp_chunk_path, chunk_path)
            except Exception as exc:  # noqa: BLE001
                temp_chunk_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to commit extracted chunk atomically: {chunk_path}: {exc}"
                ) from exc
            if self.manage_lifecycle:
                self.audio_chunks.add(chunk_path)

        return chunk_path

    def __exit__(self, exc_type, exc_value, traceback, /):
        if not self.manage_lifecycle:
            return
        for chunk in self.audio_chunks:
            try:
                chunk.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {chunk}: {e}")
