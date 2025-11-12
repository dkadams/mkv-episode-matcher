from pathlib import Path
from typing import Optional

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.embedding_model import SentenceTransformerModel
from mkv_episode_matcher.series import Series
from mkv_episode_matcher.transcription_embeddings_extractor import \
    TranscriptionEmbeddingsExtractor

_EXTRACTOR: Optional[TranscriptionEmbeddingsExtractor] = None


def _init_embeddings_extractor_worker(config: Configuration,
    series: Series, embedding_model_name: str):
    """Initializer for the process pool so Whisper loads only in child processes."""
    global  _EXTRACTOR

    model = SentenceTransformerModel(embedding_model_name)
    _EXTRACTOR = TranscriptionEmbeddingsExtractor(config, series, model)

def _extract_embeddings_from_transcription(inputs: list[Path]) -> dict[Path, Path]:
    """Extract text segments for a single file inside a worker process."""
    return {input: _EXTRACTOR.execute(input) for input in inputs}

