from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from loguru import logger


class SubprocessTranscriber:
    """Marker base class for backends that run external CLI executables."""


class WhispercppTranscriber(SubprocessTranscriber):
    """Thin wrapper around whisper.cpp's whisper-cli binary."""

    DEFAULT_THREADS = 2

    def __init__(self, model_name: str, executable: str = "whisper-cli"):
        self.executable = executable
        self.model_path = self._resolve_model_path(Path(model_name).expanduser())
        self.runtime_args = self._runtime_args()
        if not Path(self.model_path).exists():
            logger.warning(
                "whispercpp model file '%s' does not exist; transcription will likely fail",
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

    @staticmethod
    def _positive_int_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        value = str(raw).strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError:
            logger.warning(f"Ignoring {name}: expected integer, got '{raw}'")
            return None
        if parsed < 1:
            logger.warning(f"Ignoring {name}: expected value >= 1, got '{parsed}'")
            return None
        return parsed

    @classmethod
    def _runtime_args(cls) -> list[str]:
        args: list[str] = []
        threads = cls._positive_int_env("WHISPERCPP_THREADS")
        processors = cls._positive_int_env("WHISPERCPP_PROCESSORS")
        args.extend(["-t", str(threads if threads is not None else cls.DEFAULT_THREADS)])
        if processors is not None:
            args.extend(["-p", str(processors)])

        extra = str(os.environ.get("WHISPERCPP_EXTRA_ARGS", "")).strip()
        if extra:
            try:
                args.extend(shlex.split(extra))
            except ValueError as exc:
                logger.warning(f"Ignoring WHISPERCPP_EXTRA_ARGS: {exc}")
        return args

    def transcribe(self, audio_path: Path):
        logger.info(f"Transcribing {audio_path} with whispercpp")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "whispercpp_transcription"
            cmd = [
                self.executable,
                "-m",
                self.model_path,
                "-f",
                str(audio_path),
                *self.runtime_args,
                "-otxt",
                "-of",
                str(output_base),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError:
                logger.error(f"whispercpp executable '{self.executable}' not found")
                return None

            if result.returncode != 0:
                logger.error(
                    f"whispercpp failed for {audio_path} "
                    f"(exit {result.returncode}): {result.stderr.strip()}"
                )
                return None

            transcript_file = Path(f"{output_base}.txt")
            if not transcript_file.exists():
                logger.error(
                    f"whispercpp did not produce expected transcript file {transcript_file}"
                )
                logger.error(f"whispercpp output: {result.stdout.strip()}")
                logger.error(f"whispercpp error: {result.stderr.strip()}")
                return None

            text = transcript_file.read_text(encoding="utf-8").strip()
            return text or None


PARAKEET_MLX_UNSUPPORTED_MESSAGE = (
    "parakeet-mlx is only supported on macOS and requires optional dependencies "
    "(mlx, parakeet-mlx)."
)


def is_parakeet_mlx_supported() -> tuple[bool, str | None]:
    if sys.platform != "darwin":
        return False, PARAKEET_MLX_UNSUPPORTED_MESSAGE

    if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("parakeet_mlx") is None:
        return False, PARAKEET_MLX_UNSUPPORTED_MESSAGE

    return True, None


class ParakeetMlxTranscriber:
    """Batch adapter for parakeet-mlx using model.generate()."""

    DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

    def __init__(self, model_name: str | None):
        supported, reason = is_parakeet_mlx_supported()
        if not supported:
            raise RuntimeError(reason)

        self.model_name = self._resolve_model_name(model_name)
        self.cache_dir = os.environ.get("PARAKEET_CACHE_DIR")
        self.fp32 = self._is_truthy(os.environ.get("PARAKEET_FP32"))
        self.local_attention = self._is_truthy(os.environ.get("PARAKEET_LOCAL_ATTENTION"))
        self.local_attention_context_size = int(
            os.environ.get("PARAKEET_LOCAL_ATTENTION_CTX", "256")
        )
        self.model = self._load_model(
            self.model_name,
            fp32=self.fp32,
            cache_dir=self.cache_dir,
            local_attention=self.local_attention,
            local_attention_context_size=self.local_attention_context_size,
        )
        self.batch_size = max(1, int(os.environ.get("PARAKEET_MLX_BATCH_SIZE", "8")))
        self.batch_debug = self._is_truthy(os.environ.get("PARAKEET_MLX_BATCH_DEBUG"))
        self._batch_counter = 0

    @classmethod
    def _resolve_model_name(cls, model_name: str | None) -> str:
        if not model_name:
            return cls.DEFAULT_MODEL

        expanded = Path(model_name).expanduser()
        if expanded.exists():
            return str(expanded)

        normalized = str(model_name).strip()
        # The app default ("small.en") is for Whisper and should not be used here.
        if "/" in normalized or "parakeet" in normalized.lower():
            return normalized
        return cls.DEFAULT_MODEL

    @staticmethod
    def _is_truthy(value: str | None) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _load_model(
        model_name: str,
        *,
        fp32: bool,
        cache_dir: str | None,
        local_attention: bool,
        local_attention_context_size: int,
    ):
        from mlx.core import bfloat16, float32
        from parakeet_mlx import from_pretrained

        loaded = from_pretrained(
            model_name,
            dtype=float32 if fp32 else bfloat16,
            cache_dir=cache_dir,
        )
        if local_attention:
            loaded.encoder.set_attention_model(
                "rel_pos_local_attn",
                (local_attention_context_size, local_attention_context_size),
            )
        return loaded

    def transcribe(self, audio_path: Path):
        result = self.transcribe_many([audio_path])
        return result[0] if result else None

    def transcribe_many(self, audio_paths: list[Path]) -> list[str | None]:
        if not audio_paths:
            return []

        results: list[str | None] = []
        for batch in self._chunked(audio_paths, self.batch_size):
            batch_id = self._batch_counter
            self._batch_counter += 1
            try:
                results.extend(self._transcribe_batch(batch, batch_id=batch_id))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "parakeet-mlx batch transcription failed (batch_id={}, size={}): {}",
                    batch_id,
                    len(batch),
                    exc,
                )
                self._debug_batch(
                    "batch_failure",
                    {
                        "batch_id": batch_id,
                        "size": len(batch),
                        "paths": [str(path) for path in batch],
                        "error": str(exc),
                    },
                )
                raise RuntimeError(
                    "parakeet-mlx batch transcription failed "
                    f"(batch_id={batch_id}, size={len(batch)})"
                ) from exc
        return results

    @staticmethod
    def _chunked(paths: list[Path], size: int) -> Iterable[list[Path]]:
        for i in range(0, len(paths), size):
            yield paths[i:i + size]

    def _transcribe_batch(self, audio_paths: list[Path], *, batch_id: int) -> list[str | None]:
        from mlx import core as mx
        from parakeet_mlx.audio import get_logmel, load_audio

        mels = []
        mel_shapes = []
        pad_lengths = []
        for audio_path in audio_paths:
            audio = load_audio(Path(audio_path), self.model.preprocessor_config.sample_rate)
            mel = get_logmel(audio, self.model.preprocessor_config)[0]
            mels.append(mel)
            mel_shapes.append(tuple(int(dim) for dim in mel.shape))

        max_length = max(int(mel.shape[0]) for mel in mels)
        padded = []
        for mel in mels:
            pad = max_length - int(mel.shape[0])
            pad_lengths.append(pad)
            if pad > 0:
                mel = mx.pad(mel, ((0, pad), (0, 0)))
            padded.append(mel)

        batch_mel = mx.stack(padded, axis=0)
        self._debug_batch(
            "pre_generate",
            {
                "batch_id": batch_id,
                "size": len(audio_paths),
                "paths": [str(path) for path in audio_paths],
                "mel_shapes": mel_shapes,
                "pad_lengths": pad_lengths,
                "batch_shape": tuple(int(dim) for dim in batch_mel.shape),
            },
        )
        generated = self.model.generate(batch_mel)

        if len(generated) != len(audio_paths):
            raise RuntimeError(
                f"Expected {len(audio_paths)} results but received {len(generated)}"
            )

        text_results: list[str | None] = []
        for result in generated:
            text = getattr(result, "text", None)
            text_results.append(str(text).strip() if text else None)
        self._debug_batch(
            "post_generate",
            {
                "batch_id": batch_id,
                "size": len(audio_paths),
                "result_count": len(text_results),
                "non_empty_text_count": sum(1 for text in text_results if text),
            },
        )
        return text_results

    def _debug_batch(self, phase: str, payload: dict) -> None:
        if not self.batch_debug:
            return
        logger.info("parakeet-mlx debug [{}] {}", phase, payload)


def get_default_transcriber_type() -> type:
    if sys.platform == "darwin":
        return ParakeetMlxTranscriber
    return WhispercppTranscriber


def get_default_transcriber_name() -> str:
    return (
        "parakeet-mlx"
        if get_default_transcriber_type() is ParakeetMlxTranscriber
        else "whispercpp"
    )
