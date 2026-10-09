"""Offline checks for the retrieval evaluation set and its metrics."""

import json

import pytest

from rag.chunking import build_chunks
from rag.evaluation import (
    EvalCase,
    expected_ranks,
    hit_at,
    load_cases,
    recall_at,
    reciprocal_rank,
    summarize,
)

CATEGORIES = {"direct", "conflict", "cross_document", "not_in_documents", "english"}


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_every_expected_chunk_exists(cases):
    known = {c.chunk_id for c in build_chunks()}
    missing = [(case.case_id, cid) for case in cases for cid in case.expected if cid not in known]
    assert missing == []


def test_cases_are_well_formed(cases):
    assert len(cases) >= 30
    for case in cases:
        assert case.category in CATEGORIES, case.case_id
        assert case.question.strip(), case.case_id
        assert len(set(case.expected)) == len(case.expected), case.case_id
        if case.category != "not_in_documents":
            assert case.scored, case.case_id


def test_cross_document_cases_need_two_documents(cases):
    for case in cases:
        if case.category == "cross_document" and len(case.expected) > 1:
            assert len({cid.split(":")[0] for cid in case.expected}) > 1, case.case_id


def test_duplicate_case_ids_are_rejected(tmp_path):
    item = {"id": "same", "category": "direct", "question": "?", "expected": ["a:1"]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([item, item]), encoding="utf-8")
    with pytest.raises(ValueError, match="same"):
        load_cases(path)


def test_metrics_for_one_case():
    case = EvalCase("c", "cross_document", "?", ("a:1", "b:2"), "")
    ranks = expected_ranks(case, ["x", "b:2", "y", "z", "a:1"])
    assert ranks == {"a:1": 5, "b:2": 2}
    assert not hit_at(ranks, 3)
    assert hit_at(ranks, 5)
    assert recall_at(ranks, 3) == 0.5
    assert reciprocal_rank(ranks) == 0.5


def test_missing_chunk_counts_as_miss():
    case = EvalCase("c", "direct", "?", ("a:1",), "")
    ranks = expected_ranks(case, ["x", "y"])
    assert ranks == {"a:1": None}
    assert not hit_at(ranks, 10)
    assert reciprocal_rank(ranks) == 0.0


def test_summarize_averages_over_cases():
    summary = summarize([{"a": 1}, {"b": None}])
    assert summary["hit@3"] == 0.5
    assert summary["mrr"] == 0.5
