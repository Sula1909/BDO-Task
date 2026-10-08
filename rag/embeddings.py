"""OpenAI embeddings with an on-disk cache.

Chunk embeddings are cached by a hash of (model, text), so the documents are
embedded once and only changed chunks are re-sent. Queries are not cached.
"""

import hashlib
from pathlib import Path

import numpy as np
from openai import OpenAI

from rag.llm import CACHE_DIR, EMBEDDING_MODEL, make_client

BATCH_SIZE = 100


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class OpenAIEmbedder:
    def __init__(self, model: str = EMBEDDING_MODEL, cache_dir: Path = CACHE_DIR, client: OpenAI | None = None):
        self.model = model
        # One file per model: vectors from different models have different sizes.
        self.cache_path = cache_dir / f"embeddings-{model}.npz"
        self.client = client or make_client()
        self._cache = self._load_cache()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(text) for text in texts]
        missing = list(dict.fromkeys(text for text, key in zip(texts, keys) if key not in self._cache))
        if missing:
            for vector, text in zip(self._request(missing), missing):
                self._cache[self._key(text)] = vector
            self._save_cache()
        return np.array([self._cache[key] for key in keys])

    def embed_query(self, text: str) -> np.ndarray:
        return self._request([text])[0]

    def _request(self, texts: list[str]) -> list[np.ndarray]:
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = self.client.embeddings.create(model=self.model, input=texts[start : start + BATCH_SIZE])
            vectors.extend(np.array(item.embedding, dtype=np.float32) for item in response.data)
        return vectors

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        with np.load(self.cache_path) as data:
            return dict(zip(data["keys"].tolist(), data["vectors"]))

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        np.savez(self.cache_path, keys=np.array(keys), vectors=np.stack([self._cache[k] for k in keys]))
