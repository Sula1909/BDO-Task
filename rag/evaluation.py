"""Retrieval evaluation: a fixed set of questions with the chunks a correct search must return.

A case passes at k when ALL its expected chunks are in the top k (strict: a
cross-document answer needs every source). Cases with no expected chunks
(questions the documents do not answer) are not scored.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag.registry import BASE_DIR

CASES_PATH = BASE_DIR / "eval" / "retrieval_cases.json"
CUTOFFS = (3, 6, 10)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    question: str
    expected: tuple[str, ...]
    note: str

    @property
    def scored(self) -> bool:
        return bool(self.expected)


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        EvalCase(item["id"], item["category"], item["question"], tuple(item["expected"]), item.get("note", ""))
        for item in raw
    ]
    ids = [c.case_id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate case ids: {sorted(duplicates)}")
    return cases


def expected_ranks(case: EvalCase, ranking: list[str]) -> dict[str, int | None]:
    """1-based position of every expected chunk id in `ranking`, None if absent."""
    position = {chunk_id: rank for rank, chunk_id in enumerate(ranking, start=1)}
    return {chunk_id: position.get(chunk_id) for chunk_id in case.expected}


def hit_at(ranks: dict[str, int | None], k: int) -> bool:
    return all(rank is not None and rank <= k for rank in ranks.values())


def recall_at(ranks: dict[str, int | None], k: int) -> float:
    return sum(rank is not None and rank <= k for rank in ranks.values()) / len(ranks)


def reciprocal_rank(ranks: dict[str, int | None]) -> float:
    """1 / position of the first expected chunk found; 0 if none was found."""
    found = [rank for rank in ranks.values() if rank is not None]
    return 1.0 / min(found) if found else 0.0


def summarize(all_ranks: list[dict[str, int | None]]) -> dict[str, float]:
    """Averages over scored cases: hit@k and recall@k for every cutoff, plus MRR."""
    n = len(all_ranks)
    summary: dict[str, float] = {}
    for k in CUTOFFS:
        summary[f"hit@{k}"] = sum(hit_at(r, k) for r in all_ranks) / n
        summary[f"recall@{k}"] = sum(recall_at(r, k) for r in all_ranks) / n
    summary["mrr"] = sum(reciprocal_rank(r) for r in all_ranks) / n
    return summary
