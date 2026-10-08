"""Try the hybrid retriever from the terminal. Needs OPENAI_API_KEY in .env.

Usage:
    python scripts/search.py "რამდენი დღით ადრე უნდა მოვითხოვო შვებულება?"
    python scripts/search.py            # interactive: type questions, empty line to quit
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rag.retriever import DEFAULT_TOP_K, HybridRetriever, build_retriever  # noqa: E402


def show(retriever: HybridRetriever, query: str, k: int) -> None:
    print(f"\n? {query}")
    for n, r in enumerate(retriever.search(query, k), start=1):
        vec = r.vector_rank or "-"
        kw = r.keyword_rank or "-"
        print(f"{n}. [{r.score:.4f} | vector #{vec} | keyword #{kw}] {r.chunk.citation} | {r.chunk.title}")
        print(f"   {r.chunk.body[:140].replace(chr(10), ' ')}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*")
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    retriever = build_retriever()

    if args.query:
        show(retriever, " ".join(args.query), args.k)
        return
    while query := input("\nკითხვა: ").strip():
        show(retriever, query, args.k)


if __name__ == "__main__":
    main()
