from typing import Protocol
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel(Protocol):
    """Protocol that embedding implementations must satisfy."""

    def encode_document(self, text: str) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...

    def get_sentence_embedding_dimension(self) -> int: ...


class SentenceTransformerModel:
    """EmbeddingModel implementation backed by SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    def encode_document(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode_document(text))

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode_query(text))

    def get_sentence_embedding_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
