"""Retrieval tests. They never call OpenAI: embeddings and translations are replaced by fakes."""

from types import SimpleNamespace

import numpy as np
import pytest

from rag.bm25 import BM25, tokenize
from rag.chunking import Chunk, build_chunks
from rag.embeddings import OpenAIEmbedder
from rag.retriever import HybridRetriever, reciprocal_rank_fusion
from rag.translation import Translator


class IdentityTranslator:
    def translate_documents(self, texts):
        return list(texts)

    def translate_query(self, text):
        return text


class FixedEmbedder:
    """Returns prepared vectors: one per document in order, and one for any query."""

    def __init__(self, document_vectors, query_vector):
        self.document_vectors = np.array(document_vectors, dtype=np.float32)
        self.query_vector = np.array(query_vector, dtype=np.float32)

    def embed_documents(self, texts):
        return self.document_vectors[: len(texts)]

    def embed_query(self, text):
        return self.query_vector


def make_chunk(n, body):
    return Chunk(chunk_id=f"test:{n}", doc_id="leave_policy", kind="article", section=str(n),
                 title="t", chapter="c", body=body)


# --- tokenizer and BM25 ---------------------------------------------------


@pytest.mark.parametrize(
    "forms",
    [
        ["შვებულება", "შვებულების", "შვებულებას", "შვებულებაზე"],
        ["ბებია", "ბებიის"],
        ["ავადმყოფობა", "ავადმყოფობის"],
    ],
)
def test_word_forms_share_one_token(forms):
    assert len({tokenize(form)[0] for form in forms}) == 1


def test_tokenize_keeps_numbers_and_lowercases_latin():
    assert tokenize("მუხლი 4.4, NS-AI 600 ლარი") == [tokenize("მუხლი")[0], "4.4", "ns", "ai", "600", tokenize("ლარი")[0]]


def test_bm25_prefers_rare_terms_and_ignores_unknown_words():
    bm25 = BM25([["a", "common"], ["b", "common"], ["c", "common", "rare"]])
    assert bm25.idf["rare"] > bm25.idf["common"]
    first, second, third = bm25.scores(["rare"])
    assert first == second == 0.0 and third > 0
    assert bm25.scores(["missing"]) == [0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def policy_bm25():
    chunks = build_chunks()
    return [c.chunk_id for c in chunks], BM25([tokenize(c.text) for c in chunks])


@pytest.mark.parametrize(
    ("query", "expected", "within"),
    [
        ("როდის მჭირდება სამედიცინო ცნობა?", "leave_policy:6.3", 1),
        ("პაროლი რამდენი სიმბოლო უნდა იყოს?", "security_policy:4.1", 1),
        ("ბებია გარდამეცვალა, რამდენი დღე მეკუთვნის?", "leave_policy:8.2", 5),
    ],
)
def test_bm25_finds_georgian_articles(policy_bm25, query, expected, within):
    ids, bm25 = policy_bm25
    scores = bm25.scores(tokenize(query))
    top = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:within]
    assert expected in [ids[i] for i in top]


# --- fusion ---------------------------------------------------------------


def test_rrf_rewards_items_found_by_both_methods():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 5]])
    assert scores[3] > scores[1] > scores[4]


def test_top_result_of_one_method_is_never_dropped():
    # Chunk 0 is the best vector match but has no query words; chunks 1-7 match both methods moderately.
    chunks = [make_chunk(0, "ლეპტოპის დაკარგვა")] + [make_chunk(n, "შეტყობინება " * n) for n in range(1, 8)]
    vectors = [[1.0, 0.0]] + [[0.5, 1.0 - n / 10] for n in range(1, 8)]
    retriever = HybridRetriever(chunks, FixedEmbedder(vectors, [1.0, 0.0]), IdentityTranslator())

    plain_rrf = reciprocal_rank_fusion([retriever.vector_ranking("q"), retriever.keyword_ranking("შეტყობინება")])
    assert sorted(plain_rrf, key=plain_rrf.get, reverse=True).index(0) >= 3

    results = {r.chunk.chunk_id: r for r in retriever.search("შეტყობინება", k=3)}
    assert "test:0" in results
    assert (results["test:0"].vector_rank, results["test:0"].keyword_rank) == (1, None)


def test_search_uses_english_query_when_given():
    calls = []

    class RecordingTranslator(IdentityTranslator):
        def translate_query(self, text):
            calls.append(text)
            return text

    chunks = [make_chunk(n, f"ტექსტი {n}") for n in range(3)]
    retriever = HybridRetriever(chunks, FixedEmbedder([[1, 0], [0, 1], [1, 1]], [1, 0]), RecordingTranslator())
    retriever.search("ტექსტი", english_query="text")
    assert calls == []
    retriever.search("ტექსტი")
    assert calls == ["ტექსტი"]


# --- caches ---------------------------------------------------------------


def fake_openai(calls):
    def create_embeddings(model, input):
        calls.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input])

    def create_chat(model, temperature, messages):
        calls.append(messages[-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"EN {messages[-1]['content']}"))])

    return SimpleNamespace(
        embeddings=SimpleNamespace(create=create_embeddings),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_chat)),
    )


def test_embedding_cache_avoids_repeat_requests(tmp_path):
    calls = []
    first = OpenAIEmbedder(model="m", cache_dir=tmp_path, client=fake_openai(calls))
    vectors = first.embed_documents(["ა", "ბბ", "ა"])
    assert calls == [["ა", "ბბ"]]
    assert vectors.shape == (3, 2)

    second = OpenAIEmbedder(model="m", cache_dir=tmp_path, client=fake_openai(calls))
    second.embed_documents(["ბბ", "ა"])
    second.embed_documents(["გგგ"])
    assert calls == [["ა", "ბბ"], ["გგგ"]]
    assert (tmp_path / "embeddings-m.npz").exists()


def test_translation_cache_avoids_repeat_requests(tmp_path):
    calls = []
    first = Translator(model="m", cache_dir=tmp_path, client=fake_openai(calls))
    assert first.translate_documents(["ა", "ბ"]) == ["EN ა", "EN ბ"]

    second = Translator(model="m", cache_dir=tmp_path, client=fake_openai(calls))
    assert second.translate_documents(["ბ", "ა"]) == ["EN ბ", "EN ა"]
    assert sorted(calls) == ["ა", "ბ"]
