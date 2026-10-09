"""Score vector-only, keyword-only and hybrid retrieval on eval/retrieval_cases.json.

Needs OPENAI_API_KEY in .env. Chunk translations/embeddings come from the cache;
each run translates and embeds the questions again (a fraction of a cent).

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py -v     # also print the English translation of each question
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rag.evaluation import CUTOFFS, EvalCase, expected_ranks, hit_at, load_cases, summarize  # noqa: E402
from rag.retriever import CANDIDATES_PER_METHOD, DEFAULT_TOP_K, build_retriever, fuse_rankings  # noqa: E402

METHODS = ("vector", "keyword", "hybrid")


def format_ranks(ranks: dict[str, int | None]) -> str:
    return ",".join(str(r) if r is not None else "-" for r in ranks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cases = load_cases()
    retriever = build_retriever()
    chunk_ids = [c.chunk_id for c in retriever.chunks]
    with ThreadPoolExecutor(max_workers=8) as pool:
        english = list(pool.map(retriever.translator.translate_query, [c.question for c in cases]))

    ranks: dict[str, list[dict[str, int | None]]] = {m: [] for m in METHODS}
    scored_cases: list[EvalCase] = []
    print(f"{'case':<26}{'category':<18}{'vector':>8}{'keyword':>9}{'hybrid':>8}  hybrid@{DEFAULT_TOP_K}")
    for case, english_query in zip(cases, english):
        vector = retriever.vector_ranking(english_query)
        keyword = retriever.keyword_ranking(case.question)
        hybrid = fuse_rankings(vector, keyword, CANDIDATES_PER_METHOD)
        rankings = {name: [chunk_ids[i] for i in r] for name, r in zip(METHODS, (vector, keyword, hybrid))}

        if not case.scored:
            top = ", ".join(rankings["hybrid"][:3])
            print(f"{case.case_id:<26}{case.category:<18}{'(not scored) top 3: ' + top}")
        else:
            scored_cases.append(case)
            case_ranks = {m: expected_ranks(case, rankings[m]) for m in METHODS}
            for m in METHODS:
                ranks[m].append(case_ranks[m])
            status = "PASS" if hit_at(case_ranks["hybrid"], DEFAULT_TOP_K) else "FAIL"
            cells = "".join(f"{format_ranks(case_ranks[m]):>{w}}" for m, w in zip(METHODS, (8, 9, 8)))
            print(f"{case.case_id:<26}{case.category:<18}{cells}  {status}")
        if args.verbose:
            print(f"{'':<26}EN: {english_query}")

    print(f"\nCells show the position of each expected chunk (- = not in the top {CANDIDATES_PER_METHOD}).")
    print(f"Scored cases: {len(scored_cases)} of {len(cases)}\n")

    metric_names = [f"hit@{k}" for k in CUTOFFS] + [f"recall@{k}" for k in CUTOFFS] + ["mrr"]
    print(f"{'method':<10}" + "".join(f"{name:>11}" for name in metric_names))
    for m in METHODS:
        summary = summarize(ranks[m])
        print(f"{m:<10}" + "".join(f"{summary[name]:>11.2f}" for name in metric_names))

    print(f"\nHybrid hit@{DEFAULT_TOP_K} by category:")
    for category in dict.fromkeys(c.category for c in scored_cases):
        indices = [i for i, c in enumerate(scored_cases) if c.category == category]
        hits = sum(hit_at(ranks["hybrid"][i], DEFAULT_TOP_K) for i in indices)
        print(f"  {category:<18}{hits}/{len(indices)}")


if __name__ == "__main__":
    main()
