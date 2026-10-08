"""Hybrid retrieval: English embeddings (meaning) + Georgian BM25 (exact words), merged with Reciprocal Rank Fusion.

Vector side: OpenAI embeddings cannot tell Georgian texts apart, so chunks are
embedded through their cached English translation and the question is
translated too (see rag/translation.py).
Keyword side: BM25 over the original Georgian text finds exact terms: codes
(UNPAID, NS-AI), numbers, names and words the employee used.
RRF combines the two rankings using only positions, so the very different score
scales of cosine similarity and BM25 never have to be compared. Plain RRF
favours chunks found moderately well by both methods over a chunk found first
by only one, so each method's top GUARANTEED_PER_METHOD results are always kept.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag.bm25 import BM25, tokenize
from rag.chunking import Chunk, build_chunks
from rag.embeddings import normalize

DEFAULT_TOP_K = 6
CANDIDATES_PER_METHOD = 20
GUARANTEED_PER_METHOD = 2
RRF_K = 60


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class Translator(Protocol):
    def translate_documents(self, texts: list[str]) -> list[str]: ...
    def translate_query(self, text: str) -> str: ...


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    vector_rank: int | None
    keyword_rank: int | None


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Each list holds item indices, best first. An item gets 1 / (k + rank) from every list it appears in."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, translator: Translator):
        self.chunks = chunks
        self.embedder = embedder
        self.translator = translator
        english = translator.translate_documents([c.text for c in chunks])
        self.vectors = normalize(embedder.embed_documents(english))
        self.bm25 = BM25([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = DEFAULT_TOP_K, english_query: str | None = None) -> list[SearchResult]:
        """`query` is the employee's wording; pass `english_query` if a translation is already available."""
        vector_ranking = self.vector_ranking(english_query or self.translator.translate_query(query))
        keyword_ranking = self.keyword_ranking(query)
        fused = reciprocal_rank_fusion([vector_ranking, keyword_ranking])

        guaranteed = set(vector_ranking[:GUARANTEED_PER_METHOD] + keyword_ranking[:GUARANTEED_PER_METHOD])
        best = sorted(fused, key=lambda item: (item in guaranteed, fused[item]), reverse=True)[:k]

        vector_pos = {item: rank for rank, item in enumerate(vector_ranking, start=1)}
        keyword_pos = {item: rank for rank, item in enumerate(keyword_ranking, start=1)}
        return [SearchResult(self.chunks[i], fused[i], vector_pos.get(i), keyword_pos.get(i)) for i in best]

    def vector_ranking(self, english_query: str) -> list[int]:
        query_vector = normalize(np.asarray(self.embedder.embed_query(english_query), dtype=np.float32))
        similarities = self.vectors @ query_vector
        return np.argsort(-similarities)[:CANDIDATES_PER_METHOD].tolist()

    def keyword_ranking(self, query: str) -> list[int]:
        scores = self.bm25.scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:CANDIDATES_PER_METHOD] if scores[i] > 0]


def build_retriever() -> HybridRetriever:
    from rag.embeddings import OpenAIEmbedder
    from rag.translation import Translator as OpenAITranslator

    return HybridRetriever(build_chunks(), OpenAIEmbedder(), OpenAITranslator())
