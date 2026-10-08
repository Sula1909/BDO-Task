"""Document registry: what we know about each policy document.

The values come from each document's own metadata block (code, version,
effective date, status). Authority reflects the precedence rules the documents
state themselves: Leave policy 1.4, Handbook 1.2, the FAQ introduction and
Remote policy 1.3.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from hr_mcp.config import BASE_DIR

DOCUMENTS_DIR = BASE_DIR / "documents"

Status = Literal["current", "overview", "reference"]

# Lower number wins when sources disagree.
AUTHORITY_POLICY = 1
AUTHORITY_HANDBOOK = 2
AUTHORITY_FAQ = 3


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    code: str
    file_name: str
    title: str
    version: str
    effective_date: date
    status: Status
    authority: int

    @property
    def path(self) -> Path:
        return DOCUMENTS_DIR / self.file_name

    @property
    def format(self) -> str:
        return self.path.suffix.lstrip(".").lower()


DOCUMENTS: tuple[DocumentInfo, ...] = (
    DocumentInfo(
        doc_id="leave_policy",
        code="HR-POL-02",
        file_name="Leave_and_Absence_Policy_v4.0.docx",
        title="შვებულებისა და გაცდენის პოლიტიკა",
        version="4.0",
        effective_date=date(2026, 1, 1),
        status="current",
        authority=AUTHORITY_POLICY,
    ),
    DocumentInfo(
        doc_id="remote_policy",
        code="HR-POL-05",
        file_name="Remote_and_Hybrid_Work_Policy_v2.0.pdf",
        title="დისტანციური და ჰიბრიდული მუშაობის პოლიტიკა",
        version="2.0",
        effective_date=date(2026, 9, 1),
        status="current",
        authority=AUTHORITY_POLICY,
    ),
    DocumentInfo(
        doc_id="learning_policy",
        code="HR-POL-07",
        file_name="Learning_and_Development_Policy_v1.2.pdf",
        title="სწავლისა და პროფესიული განვითარების პოლიტიკა",
        version="1.2",
        effective_date=date(2026, 2, 1),
        status="current",
        authority=AUTHORITY_POLICY,
    ),
    DocumentInfo(
        doc_id="security_policy",
        code="SEC-POL-01",
        file_name="Information_Security_Policy_v3.2.pdf",
        title="ინფორმაციული უსაფრთხოებისა და IT რესურსების გამოყენების პოლიტიკა",
        version="3.2",
        effective_date=date(2026, 3, 1),
        status="current",
        authority=AUTHORITY_POLICY,
    ),
    DocumentInfo(
        doc_id="travel_policy",
        code="FIN-POL-03",
        file_name="Travel_and_Expense_Policy_v2.3.pdf",
        title="მივლინებისა და ხარჯების ანაზღაურების პოლიტიკა",
        version="2.3",
        effective_date=date(2026, 4, 1),
        status="current",
        authority=AUTHORITY_POLICY,
    ),
    DocumentInfo(
        doc_id="employee_handbook",
        code="HR-HB-01",
        file_name="Employee_Handbook_v3.1.docx",
        title="თანამშრომლის სახელმძღვანელო",
        version="3.1",
        effective_date=date(2025, 2, 1),
        status="overview",
        authority=AUTHORITY_HANDBOOK,
    ),
    DocumentInfo(
        doc_id="employee_faq",
        code="HR-FAQ-01",
        file_name="Employee_FAQ_2025.docx",
        title="ხშირად დასმული კითხვები თანამშრომლებისთვის",
        version="1.4",
        effective_date=date(2025, 11, 1),
        status="reference",
        authority=AUTHORITY_FAQ,
    ),
)

_BY_ID = {doc.doc_id: doc for doc in DOCUMENTS}


def get_document(doc_id: str) -> DocumentInfo:
    return _BY_ID[doc_id]
