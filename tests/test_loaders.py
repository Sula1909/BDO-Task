import re

import pytest

from rag.loaders import Block, _append_across_pages, load_all
from rag.registry import AUTHORITY_FAQ, AUTHORITY_HANDBOOK, AUTHORITY_POLICY, DOCUMENTS, get_document

PAGE_FURNITURE = re.compile(r"შიდა გამოყენებისთვის გვერდი|^შპს „ნორთსტარ სერვისეზი“ \|")


@pytest.fixture(scope="module")
def documents():
    return {doc.info.doc_id: doc for doc in load_all()}


def blocks_of(doc, kind, level=None):
    return [b for b in doc.blocks if b.kind == kind and (level is None or b.level == level)]


def article(doc, number):
    """Blocks between the article heading `number` and the next heading."""
    start = next(i for i, b in enumerate(doc.blocks) if b.kind == "heading" and b.text.startswith(f"{number} "))
    body = []
    for block in doc.blocks[start + 1 :]:
        if block.kind == "heading":
            break
        body.append(block)
    return body


# --- registry -------------------------------------------------------------


def test_registry_files_exist_and_ids_are_unique():
    assert len({d.doc_id for d in DOCUMENTS}) == len(DOCUMENTS) == 7
    for info in DOCUMENTS:
        assert info.path.is_file(), info.file_name


def test_registry_authority_levels():
    assert get_document("employee_faq").authority == AUTHORITY_FAQ
    assert get_document("employee_handbook").authority == AUTHORITY_HANDBOOK
    policies = [d for d in DOCUMENTS if d.authority == AUTHORITY_POLICY]
    assert {d.code for d in policies} == {"HR-POL-02", "HR-POL-05", "HR-POL-07", "SEC-POL-01", "FIN-POL-03"}


def test_registry_matches_document_metadata(documents):
    for doc in documents.values():
        assert doc.metadata["დოკუმენტის კოდი"] == doc.info.code
        assert doc.metadata["ვერსია"] == doc.info.version


# --- structure ------------------------------------------------------------


@pytest.mark.parametrize(
    ("doc_id", "chapters", "articles"),
    [("leave_policy", 15, 56), ("employee_handbook", 18, 50), ("employee_faq", 6, 21)],
)
def test_docx_headings_follow_word_styles(documents, doc_id, chapters, articles):
    doc = documents[doc_id]
    assert len(blocks_of(doc, "heading", 1)) == chapters
    assert len(blocks_of(doc, "heading", 2)) == articles


@pytest.mark.parametrize("doc_id", ["remote_policy", "learning_policy", "security_policy", "travel_policy"])
def test_pdf_headings_are_numbered(documents, doc_id):
    doc = documents[doc_id]
    for block in blocks_of(doc, "heading", 1):
        assert re.match(r"^\d+\. \S", block.text), block.text
    for block in blocks_of(doc, "heading", 2):
        assert re.match(r"^\d+\.\d+ \S", block.text), block.text


def test_faq_uses_georgian_letter_numbering(documents):
    headings = [b.text for b in blocks_of(documents["employee_faq"], "heading", 2)]
    assert headings[0].startswith("ა.1 ")
    assert any(h.startswith("ა.3 ") for h in headings)


def test_no_page_headers_or_footers(documents):
    for doc in documents.values():
        for block in doc.blocks:
            assert not PAGE_FURNITURE.search(block.to_text()), (doc.info.code, block.to_text())


def test_metadata_table_is_not_a_content_block(documents):
    for doc in documents.values():
        for block in blocks_of(doc, "table"):
            assert block.rows[0][0] != "კომპანია", doc.info.code


# --- PDF text and tables --------------------------------------------------


def test_pdf_wrapped_lines_are_joined_into_one_paragraph(documents):
    (paragraph,) = article(documents["remote_policy"], "4.1")
    assert paragraph.text == (
        "უშუალო ხელმძღვანელის წინასწარი თანხმობით თანამშრომელს შეუძლია კვირაში "
        "არაუმეტეს 2 სამუშაო დღე იმუშაოს დისტანციურად. ეს დღეები წინასწარ უნდა შეიტანოს "
        "გუნდის საერთო კალენდარში."
    )


def test_pdf_list_items_stay_separate(documents):
    texts = [b.text for b in article(documents["security_policy"], "6.1")]
    assert "Microsoft 365 Copilot, კომპანიის ანგარიშით;" in texts
    assert "HR ასისტენტი, მხოლოდ HR საკითხებისთვის." in texts


def test_pdf_table_keeps_rows_and_wrapped_cells(documents):
    (table,) = [b for b in article(documents["security_policy"], "6.2") if b.kind == "table"]
    assert table.rows[0] == ["სერვისი", "საჯარო", "შიდა", "კონფიდენციალური", "შეზღუდული"]
    assert ["NS-AI", "დიახ", "დიახ", "დიახ, პროექტის პარტნიორის თანხმობით", "არა"] in table.rows


def test_table_split_across_pages_is_merged(documents):
    tables = [b for b in article(documents["security_policy"], "2.1") if b.kind == "table"]
    assert len(tables) == 1
    assert [row[0] for row in tables[0].rows] == ["კლასი", "საჯარო", "შიდა", "კონფიდენციალური", "შეზღუდული"]


def test_table_stays_inside_its_article(documents):
    body = article(documents["travel_policy"], "6.2")
    assert any(b.kind == "table" and ["გერმანია", "60 ევრო"] in b.rows for b in body)


def test_docx_table_extracted_in_reading_order(documents):
    body = article(documents["leave_policy"], "4.1")
    assert [b.kind for b in body] == ["paragraph", "table", "paragraph"]
    assert body[1].rows[1:] == [["5 წლამდე", "24"], ["5 წლიდან 10 წლამდე", "25"], ["10 წელი და მეტი", "26"]]


def test_paragraph_cut_by_page_break_is_joined():
    blocks = [Block("paragraph", "პირველი გვერდის ტექსტი გრძელდება", page=1)]
    _append_across_pages(blocks, Block("paragraph", "მეორე გვერდზე.", page=2))
    assert [b.text for b in blocks] == ["პირველი გვერდის ტექსტი გრძელდება მეორე გვერდზე."]

    _append_across_pages(blocks, Block("paragraph", "ახალი აბზაცი.", page=3))
    assert len(blocks) == 2
