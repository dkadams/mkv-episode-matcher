from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import torch
import whisper
from faster_whisper import WhisperModel
from loguru import logger


class WhisperTranscriber:
    def __init__(self, model_name):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        fp16 = self.model.device != torch.device("cpu")
        result = whisper.transcribe(self.model, str(audio_path), fp16=fp16)
        return result["text"] if result else None

class FasterWhisperTranscriber:
    def __init__(self, model_name):
        self.model = WhisperModel(model_name)

    def transcribe(self, audio_path):
        logger.info(f"Transcribing {audio_path}")
        segments, info = self.model.transcribe(str(audio_path))
        text_segments = [segment.text for segment in segments]
        return " ".join(text_segments)

class WhispercppCliTranscriber:
    """Thin wrapper around whisper.cpp's whispercpp-cli binary."""

    def __init__(self, model_name, executable="whisper-cli"):
        self.executable = executable
        self.model_path = self._resolve_model_path(Path(model_name).expanduser())
        if not Path(self.model_path).exists():
            logger.warning(
                "whispercpp-cli model file '%s' does not exist; transcription will likely fail",
                self.model_path,
            )

    @staticmethod
    def _resolve_model_path(model_path: Path) -> str:
        # If caller provided an explicit path, prefer it.
        if model_path.is_absolute() or model_path.parent != Path("."):
            return str(model_path)

        candidates = [model_path]
        if model_path.suffix not in (".bin", ".gguf"):
            candidates.extend([
                model_path.with_suffix(".bin"),
                Path(f"ggml-{model_path.name}.bin"),
                Path(f"ggml-{model_path.name}.gguf"),
            ])

        search_dirs = []
        env_dir = os.environ.get("WHISPER_CPP_MODELS_DIR")
        if env_dir:
            search_dirs.append(Path(env_dir).expanduser())
        search_dirs.extend([
            Path.cwd(),
            Path.home(),
            Path.home() / ".cache/whisper.cpp",
            Path.home() / ".cache/ggml",
            ])

        for candidate in candidates:
            if candidate.is_absolute():
                if candidate.exists():
                    return str(candidate)
                continue

            for directory in search_dirs:
                resolved = (directory / candidate).expanduser()
                if resolved.exists():
                    return str(resolved)

        # Fall back to the ggml naming convention in the first search directory.
        fallback = Path(f"ggml-{model_path.name}.bin")
        return str(fallback)

    def transcribe(self, audio_path: Path):
        logger.info(f"Transcribing {audio_path} with whispercpp-cli")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "whispercpp_transcription"
            cmd = [
                self.executable,
                "-m",
                self.model_path,
                "-f",
                str(audio_path),
                "-otxt",
                "-of",
                str(output_base),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        check=False)
            except FileNotFoundError:
                logger.error(f"whispercpp-cli executable '{self.executable}' not found")
                return None

            if result.returncode != 0:
                logger.error(f"whispercpp-cli failed for {audio_path} "
                             f"(exit {result.returncode}): {result.stderr.strip()}")
                return None

            transcript_file = Path(f"{output_base}.txt")
            if not transcript_file.exists():
                logger.error(f"whispercpp-cli did not produce expected transcript file {transcript_file}")
                return None

            text = transcript_file.read_text(encoding="utf-8").strip()
            return text or None

class WhisperKitCliTranscriber:
    """Adapter for the Swift whisperkit-cli binary."""

    def __init__(self, model_name: str, executable: str = "whisperkit-cli"):
        self.executable = executable
        self.model_arg = None
        if model_name:
            expanded = Path(model_name).expanduser()
            if expanded.exists():
                self.model_arg = ("--model-path", str(expanded))
            else:
                self.model_arg = ("--model", model_name)

    def transcribe(self, audio_path: Path):
        logger.info(f"Transcribing {audio_path} with whisperkit-cli")
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            logger.info(f"Writing report to {report_dir}")
            cmd = [
                self.executable,
                "transcribe",
                "--audio-path",
                str(audio_path),
                "--report",
                "--report-path",
                str(report_dir),
                "--without-timestamps",
            ]
            if self.model_arg:
                cmd.extend(self.model_arg)

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError:
                logger.error(f"whisperkit-cli executable '{self.executable}' not found")
                return None

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.error(f"whisperkit-cli failed for {audio_path} (exit {result.returncode}): {stderr}")
                return None

            return result.stdout.strip()
