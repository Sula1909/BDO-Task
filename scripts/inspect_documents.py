"""Show what the loaders and the chunker produce from each document.

Usage:
    python scripts/inspect_documents.py          # summary table
    python scripts/inspect_documents.py --dump   # also write extracted/<doc_id>.txt and extracted/chunks.txt
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rag.chunking import Chunk, build_chunks  # noqa: E402
from rag.loaders import LoadedDocument, load_all  # noqa: E402

DUMP_DIR = PROJECT_DIR / "extracted"


def summarize(doc: LoadedDocument) -> str:
    counts = Counter((b.kind, b.level) for b in doc.blocks)
    meta_ok = (
        doc.metadata.get("დოკუმენტის კოდი") == doc.info.code
        and doc.metadata.get("ვერსია") == doc.info.version
    )
    return (
        f"{doc.info.code:<11} {doc.info.format:<5} "
        f"chapters={counts[('heading', 1)]:<3} articles={counts[('heading', 2)]:<3} "
        f"paragraphs={counts[('paragraph', 0)]:<4} tables={counts[('table', 0)]:<2} "
        f"metadata={'OK' if meta_ok else 'MISMATCH'}"
    )


def render(doc: LoadedDocument) -> str:
    lines = [f"# {doc.info.title} ({doc.info.code}, v{doc.info.version})", ""]
    lines += [f"{key}: {value}" for key, value in doc.metadata.items()]
    for block in doc.blocks:
        page = f" p.{block.page}" if block.page else ""
        if block.kind == "heading":
            lines += ["", f"{'#' * (block.level + 1)} {block.text}{page}"]
        elif block.kind == "table":
            lines += [f"[TABLE{page}]", block.to_text(), "[/TABLE]"]
        else:
            lines.append(block.text)
    return "\n".join(lines) + "\n"


def summarize_chunks(chunks: list[Chunk]) -> str:
    kinds = Counter(c.kind for c in chunks)
    sizes = sorted(len(c.text) for c in chunks)
    longest = max(chunks, key=lambda c: len(c.text))
    return (
        f"chunks={len(chunks)} ({', '.join(f'{k}={n}' for k, n in sorted(kinds.items()))})\n"
        f"characters: min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]} ({longest.chunk_id})"
    )


def render_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        pages = f" | pages {c.page_start}-{c.page_end}" if c.page_start else ""
        parts.append(f"===== {c.chunk_id} | {c.kind} | {c.citation}{pages}\n{c.text}\n")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", action="store_true", help="write extracted/<doc_id>.txt files")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    documents = load_all()
    for doc in documents:
        print(summarize(doc))

    chunks = build_chunks(documents)
    print()
    print(summarize_chunks(chunks))

    if args.dump:
        DUMP_DIR.mkdir(exist_ok=True)
        for doc in documents:
            (DUMP_DIR / f"{doc.info.doc_id}.txt").write_text(render(doc), encoding="utf-8")
        (DUMP_DIR / "chunks.txt").write_text(render_chunks(chunks), encoding="utf-8")
        print(f"\nWrote {len(documents) + 1} files to {DUMP_DIR}")


if __name__ == "__main__":
    main()
