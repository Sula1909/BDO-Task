"""Ask a policy question and get a cited answer. Needs OPENAI_API_KEY in .env.

Usage:
    python scripts/ask.py "რამდენი დღით ადრე უნდა მოვითხოვო შვებულება?"
    python scripts/ask.py            # interactive: type questions, empty line to quit
    python scripts/ask.py -v "..."   # also list the retrieved chunks
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rag.answering import ANSWER_TOP_K, Answerer  # noqa: E402
from rag.retriever import build_retriever  # noqa: E402


def show(answerer: Answerer, question: str, k: int, verbose: bool) -> None:
    answer = answerer.answer(question, k)
    print(f"\n? {question}\n")
    print(answer.format())
    if verbose:
        print("\nმოძიებული ფრაგმენტები:")
        for n, chunk in enumerate(answer.retrieved, start=1):
            print(f"  [{n}] {chunk.citation}: {chunk.title}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*")
    parser.add_argument("-k", type=int, default=ANSWER_TOP_K)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    answerer = Answerer(build_retriever())

    if args.question:
        show(answerer, " ".join(args.question), args.k, args.verbose)
        return
    while question := input("\nკითხვა: ").strip():
        show(answerer, question, args.k, args.verbose)


if __name__ == "__main__":
    main()
