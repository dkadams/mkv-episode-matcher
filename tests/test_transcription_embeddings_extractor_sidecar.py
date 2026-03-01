import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np

from mkv_episode_matcher.transcription_embeddings_extractor import (
    TranscriptionEmbeddingsExtractor,
)


class DummyModel:
    def encode_document(self, _text: str) -> np.ndarray:
        return np.array([0.1, 0.2], dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return 2


def test_execute_writes_low_info_sidecar(tmp_path):
    series = SimpleNamespace(
        low_info_min_words=8,
        low_info_cue_ratio=0.25,
        ensure_transcription_embeddings_dir=lambda: tmp_path,
    )
    config = SimpleNamespace(args=Namespace())
    extractor = TranscriptionEmbeddingsExtractor(config, series, DummyModel())

    transcription = tmp_path / "sample.json"
    transcription.write_text(
        json.dumps(
            {
                "0": "(mysterious music)",
                "1": "This is a full sentence with enough tokens to avoid filtering.",
            }
        ),
        encoding="utf-8",
    )

    embeddings_path = extractor.execute(transcription)
    sidecar_path = embeddings_path.with_suffix(".meta.json")

    assert embeddings_path.exists()
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert "0" in payload["low_info_intervals"]
    assert "1" not in payload["low_info_intervals"]
