"""Group document blocks into retrieval chunks.

One chunk per article (N.M in policies and the Handbook, ა.1 in the FAQ). Text
that sits directly under a chapter heading (a chapter without articles, or a
chapter introduction) becomes a chapter chunk. The version history chapter is
marked separately because it quotes old values. Each document also gets one
chunk built from its metadata table (code, version, effective date, status).
"""

import re
from dataclasses import dataclass
from typing import Literal

from rag.loaders import Block, LoadedDocument, load_all
from rag.registry import DocumentInfo, get_document

ChunkKind = Literal["document", "chapter", "article", "history"]

SECTION_NUMBER = re.compile(r"^(?P<number>\d+(?:\.\d+)*|[ა-ჰ](?:\.\d+)?)\.?\s+(?P<title>.+)$")
HISTORY_TITLE = "დოკუმენტის ისტორია"
UNNUMBERED_SECTION = "intro"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    kind: ChunkKind
    section: str
    title: str
    chapter: str
    body: str
    page_start: int | None = None
    page_end: int | None = None

    @property
    def info(self) -> DocumentInfo:
        return get_document(self.doc_id)

    @property
    def citation(self) -> str:
        """Short source reference shown to the employee, e.g. "HR-POL-02 v4.0, მუხლი 4.4"."""
        source = f"{self.info.code} v{self.info.version}"
        if self.kind == "document":
            return source
        if self.kind == "history" or self.section == UNNUMBERED_SECTION:
            return f"{source}, {self.title}"
        label = "მუხლი" if self.section[0].isdigit() else "კითხვა"
        return f"{source}, {label} {self.section}"

    @property
    def text(self) -> str:
        """Header plus body: what gets embedded, keyword-searched and shown to the model."""
        lines = [f"დოკუმენტი: {self.info.title} ({self.info.code}, ვერსია {self.info.version})"]
        if self.kind == "article":
            lines.append(f"თავი: {self.chapter}")
            lines.append(f"მუხლი: {self.section} {self.title}")
        elif self.kind != "document":
            lines.append(f"თავი: {self.chapter}")
        return "\n".join(lines) + "\n\n" + self.body


def chunk_document(doc: LoadedDocument) -> list[Chunk]:
    chunks = [_document_chunk(doc)]
    chapter = ""
    heading: Block | None = None
    body: list[Block] = []

    # Blocks before the first heading (the "this document is fictional" note) have no heading and are dropped.
    for block in [*doc.blocks, None]:
        if block is None or block.kind == "heading":
            if heading is not None and body:
                chunks.append(_section_chunk(doc.info, heading, chapter, body))
            if block is None:
                break
            heading, body = block, []
            if block.level == 1:
                chapter = block.text
        else:
            body.append(block)
    return chunks


def build_chunks(documents: list[LoadedDocument] | None = None) -> list[Chunk]:
    documents = load_all() if documents is None else documents
    chunks = [chunk for doc in documents for chunk in chunk_document(doc)]
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate chunk id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
    return chunks


def split_heading(text: str) -> tuple[str, str]:
    """'4.4 მოთხოვნის წარდგენა' -> ('4.4', 'მოთხოვნის წარდგენა'); unnumbered -> ('', text)."""
    match = SECTION_NUMBER.match(text)
    if match is None:
        return "", text
    return match["number"], match["title"]


def _document_chunk(doc: LoadedDocument) -> Chunk:
    body = "\n".join(f"{key}: {value}" for key, value in doc.metadata.items())
    return Chunk(
        chunk_id=f"{doc.info.doc_id}:document",
        doc_id=doc.info.doc_id,
        kind="document",
        section="",
        title=doc.info.title,
        chapter="",
        body=body,
    )


def _section_chunk(info: DocumentInfo, heading: Block, chapter: str, body: list[Block]) -> Chunk:
    section, title = split_heading(heading.text)
    _, chapter_title = split_heading(chapter)
    if chapter_title == HISTORY_TITLE:
        kind: ChunkKind = "history"
    elif heading.level == 1:
        kind = "chapter"
    else:
        kind = "article"

    pages = [b.page for b in (heading, *body) if b.page is not None]
    return Chunk(
        chunk_id=f"{info.doc_id}:{section or UNNUMBERED_SECTION}",
        doc_id=info.doc_id,
        kind=kind,
        section=section or UNNUMBERED_SECTION,
        title=title,
        chapter=chapter,
        body="\n".join(block.to_text() for block in body),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
    )
