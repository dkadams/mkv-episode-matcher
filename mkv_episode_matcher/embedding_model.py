from __future__ import annotations

from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel(Protocol):
    """Protocol that embedding implementations must satisfy."""

    def encode_document(self, text: str) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...

    def get_sentence_embedding_dimension(self) -> int: ...

    def dir_name(self) -> str: ...


class SentenceTransformerModel:
    """EmbeddingModel implementation backed by SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self._model = SentenceTransformer(model_name)

    def encode_document(self, text: str) -> np.ndarray:
        return self._encode(text, method="encode_document")

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text, method="encode_query")

    def get_sentence_embedding_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def dir_name(self) -> str:
        return self.model_name.replace("/", "-")

    def __str__(self) -> str:
        return f"SentenceTransformerModel({self.model_name})"

    def _encode(self, text: str, method: str) -> np.ndarray:
        encoder = getattr(self._model, method, None)
        if callable(encoder):
            vector = encoder(text)
        else:
            vector = self._model.encode(text, convert_to_numpy=True)
        return np.asarray(vector, dtype=np.float32)
