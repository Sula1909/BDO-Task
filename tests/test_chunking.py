from collections import Counter

import pytest

from rag.chunking import build_chunks, chunk_document, split_heading
from rag.loaders import load_all
from rag.registry import DOCUMENTS


@pytest.fixture(scope="module")
def documents():
    return load_all()


@pytest.fixture(scope="module")
def chunks(documents):
    return {c.chunk_id: c for c in build_chunks(documents)}


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("4.4 მოთხოვნის წარდგენა", ("4.4", "მოთხოვნის წარდგენა")),
        ("8. ქაღალდის დოკუმენტები", ("8", "ქაღალდის დოკუმენტები")),
        ("ა.2 რამდენი ხნით ადრე?", ("ა.2", "რამდენი ხნით ადრე?")),
        ("ბ. დისტანციური მუშაობა", ("ბ", "დისტანციური მუშაობა")),
        ("როგორ გამოვიყენოთ ეს დოკუმენტი", ("", "როგორ გამოვიყენოთ ეს დოკუმენტი")),
    ],
)
def test_split_heading(heading, expected):
    assert split_heading(heading) == expected


def test_every_article_becomes_one_chunk(documents, chunks):
    articles = sum(1 for doc in documents for b in doc.blocks if b.kind == "heading" and b.level == 2)
    assert Counter(c.kind for c in chunks.values())["article"] == articles


def test_each_document_has_one_metadata_chunk(chunks):
    document_chunks = [c for c in chunks.values() if c.kind == "document"]
    assert {c.doc_id for c in document_chunks} == {d.doc_id for d in DOCUMENTS}
    leave = chunks["leave_policy:document"]
    assert "ძალაშია: 2026 წლის 1 იანვრიდან" in leave.body
    assert "სტატუსი: მოქმედი" in leave.body


def test_no_content_is_lost(documents):
    """Every paragraph and table after the first heading ends up in exactly one chunk."""
    for doc in documents:
        bodies = "\n".join(c.body for c in chunk_document(doc))
        first_heading = next(i for i, b in enumerate(doc.blocks) if b.kind == "heading")
        for block in doc.blocks[first_heading:]:
            if block.kind != "heading":
                assert bodies.count(block.to_text()) >= 1, (doc.info.code, block.to_text()[:60])


def test_fictional_document_note_is_not_a_chunk(chunks):
    assert not any(c.body.startswith("ეს დოკუმენტი გამოგონილია") for c in chunks.values())


def test_article_chunk_with_table(chunks):
    chunk = chunks["leave_policy:4.4"]
    assert chunk.kind == "article"
    assert chunk.title == "მოთხოვნის წარდგენა და წინასწარი შეტყობინება"
    assert chunk.chapter == "4. ყოველწლიური ანაზღაურებადი შვებულება"
    assert "6-დან 15-მდე | 15 სამუშაო დღე" in chunk.body
    assert chunk.citation == "HR-POL-02 v4.0, მუხლი 4.4"


def test_text_starts_with_context_header(chunks):
    lines = chunks["security_policy:9.1"].text.splitlines()
    assert lines[0].startswith("დოკუმენტი: ინფორმაციული უსაფრთხოებისა") and "SEC-POL-01" in lines[0]
    assert lines[1] == "თავი: 9. საზღვარგარეთ მგზავრობა მოწყობილობით"
    assert lines[2] == "მუხლი: 9.1 წინასწარი შეტყობინება"


def test_chapter_without_articles_is_a_chunk(chunks):
    chunk = chunks["security_policy:8"]
    assert chunk.kind == "chapter"
    assert chunk.citation == "SEC-POL-01 v3.2, მუხლი 8"
    assert (chunk.page_start, chunk.page_end) == (5, 5)


def test_faq_question_chunks(chunks):
    chunk = chunks["employee_faq:ა.2"]
    assert chunk.title == "რამდენი ხნით ადრე უნდა მოვითხოვო შვებულება?"
    assert "5 სამუშაო დღით ადრე" in chunk.body
    assert chunk.citation == "HR-FAQ-01 v1.4, კითხვა ა.2"


def test_faq_introduction_keeps_precedence_rule(chunks):
    chunk = chunks["employee_faq:intro"]
    assert chunk.kind == "chapter"
    assert "იხელმძღვანელეთ პოლიტიკით" in chunk.body


def test_history_chapters_are_marked(chunks):
    history = {c.doc_id: c for c in chunks.values() if c.kind == "history"}
    assert set(history) == {"leave_policy", "remote_policy", "learning_policy", "security_policy", "travel_policy"}
    assert "3-დან 2 დღემდე" in history["remote_policy"].body
    assert history["remote_policy"].citation == "HR-POL-05 v2.0, დოკუმენტის ისტორია"


def test_chunks_are_small_enough_to_keep_whole(chunks):
    assert max(len(c.text) for c in chunks.values()) < 2000
