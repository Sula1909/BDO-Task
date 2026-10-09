"""Answer policy questions from retrieved chunks, with citations and precedence between sources.

The model sees numbered sources, each labelled with its weight (current policy,
Handbook overview, FAQ, or version history) and replies with JSON. It first
fills "source_check": what each relevant source states and which other sources
state something different. Deciding which source wins is NOT left to the
model (it reliably spots differences but often fails to apply precedence):
the code marks a source outdated when it differs from a source with higher
weight. If the answer cited an outdated source, the model gets one correction
and rewrites it. The code also turns [n] markers into citations, so a citation
can only point at a chunk that was actually retrieved.
"""

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from hr_mcp.config import app_today
from rag.chunking import Chunk
from rag.llm import CHAT_MODEL, make_client
from rag.registry import AUTHORITY_FAQ, AUTHORITY_HANDBOOK
from rag.retriever import HybridRetriever

# More than the search default: two-part questions often have their second source at rank 7-10.
ANSWER_TOP_K = 8
HISTORY_WEIGHT = 9
NOT_FOUND_MESSAGE = "ამ კითხვაზე პასუხი კომპანიის პოლიტიკის დოკუმენტებში ვერ მოიძებნა."
CONTACT_HR = "დამატებითი ინფორმაციისთვის მიმართეთ HR დეპარტამენტს."
ERROR_MESSAGE = "პასუხის მომზადება ვერ მოხერხდა. გთხოვთ, სცადოთ ხელახლა."
SOURCE_MARKER = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")

SYSTEM_PROMPT = """You are the HR policy assistant of Northstar Services. Today is {today}.
Answer the employee's question using ONLY the numbered sources below. Never use outside knowledge.

Source weight (each source is labelled):
1. A current policy (HR-POL-*, SEC-POL-*, FIN-POL-*) always wins.
2. The Employee Handbook is an overview. If it disagrees with a policy, the policy applies.
3. The FAQ is reference material. If it disagrees with a policy or the Handbook, they apply.
4. A "version history" source describes earlier versions. Old values mentioned there are NOT in force;
   use it only to explain what changed.
A source that states something different from a source with higher weight is OUTDATED: never use its
numbers or conditions in the answer, even if they seem to fit the question.

Rules:
- Always reply in Georgian, even if the question is in English. Be concise: 1-5 sentences or a short list.
- Put source markers like [1] or [2, 3] right after the statement they support. Every number, amount,
  deadline or condition must have a marker. Never put a marker of an outdated source in the answer.
- Copy numbers, amounts and dates exactly as the source states them.
- If the sources do not answer the question, set "found" to false. Then the answer must begin with
  "ამ კითხვაზე პასუხი კომპანიის პოლიტიკის დოკუმენტებში ვერ მოიძებნა." If a source says where the
  information is published or that it is not covered, mention that with its marker.
- If the sources answer only part of the question, answer that part and say which part is not covered.
- Never invent rules, numbers or procedures.

JSON fields, filled in this order:
- "source_check": every source that is about the question, BEFORE writing the answer. For each:
  "source" (its number), "states" (what it literally says on the question, with its exact numbers, in
  English; only what is written there) and "differs_from" (numbers of other listed sources that state a
  different number, deadline, limit or condition on the same point; empty if it agrees with all).
  Leave out sources that are not about the question.
- "found": whether the sources answer the question.
- "answer": the answer, following the source with the highest weight.
- "conflict_note": if a lower-weight source differs from a higher-weight one, one Georgian sentence per
  outdated source saying what it states and that it is outdated, with its marker, e.g.
  "FAQ-ში მითითებული 5 სამუშაო დღე [3] მოძველებულია; მოქმედებს პოლიტიკის წესი." Otherwise ""."""

CORRECTION_PROMPT = (
    "Correction: source(s) {outdated} differ from source(s) with higher weight, so they are outdated. "
    "Rewrite the JSON: the answer must use only the higher-weight sources and their exact numbers, and "
    "conflict_note must say that {outdated} are outdated."
)

ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "policy_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "source_check": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "integer"},
                            "states": {"type": "string"},
                            "differs_from": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["source", "states", "differs_from"],
                        "additionalProperties": False,
                    },
                },
                "found": {"type": "boolean"},
                "answer": {"type": "string"},
                "conflict_note": {"type": "string"},
            },
            "required": ["source_check", "found", "answer", "conflict_note"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    found: bool
    sources: tuple[Chunk, ...]
    outdated: tuple[Chunk, ...]
    retrieved: tuple[Chunk, ...]
    conflict_note: str = ""

    def format(self) -> str:
        """Answer plus the source list, as shown to the employee."""
        lines = [self.text]
        if self.conflict_note:
            lines += ["", f"შენიშვნა: {self.conflict_note}"]
        if not self.found:
            lines += ["", CONTACT_HR]
        if self.sources:
            lines += ["", "წყაროები:"] + [f"- {c.citation}: {c.title}" for c in self.sources]
        if self.outdated:
            lines += ["", "მოძველებული ინფორმაცია (არ გამოიყენება):"]
            lines += [f"- {c.citation}: {c.title}" for c in self.outdated]
        return "\n".join(lines)


@dataclass(frozen=True)
class Reply:
    """The model's JSON with precedence already applied; source numbers are 1-based and valid."""

    found: bool
    text: str
    note: str
    relevant: list[int]
    outdated: list[int]
    cited: list[int]

    @property
    def cites_outdated(self) -> list[int]:
        return [n for n in self.cited if n in self.outdated]


def weight(chunk: Chunk) -> int:
    """Lower wins: policy 1, Handbook 2, FAQ 3, version history last."""
    return HISTORY_WEIGHT if chunk.kind == "history" else chunk.info.authority


def source_label(chunk: Chunk) -> str:
    if chunk.kind == "history":
        return "version history: describes earlier versions, old values are not in force"
    if chunk.info.authority == AUTHORITY_HANDBOOK:
        return "Handbook overview: a policy wins if they disagree"
    if chunk.info.authority == AUTHORITY_FAQ:
        return "FAQ reference: a policy or the Handbook wins if they disagree"
    return "current policy: highest weight"


def order_for_context(chunks: list[Chunk]) -> list[Chunk]:
    """Policies first, then Handbook, then FAQ, history chunks last; retrieval order within each group."""
    position = {c.chunk_id: i for i, c in enumerate(chunks)}
    return sorted(chunks, key=lambda c: (weight(c), position[c.chunk_id]))


def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for n, chunk in enumerate(chunks, start=1):
        header = f"[{n}] {chunk.citation} | {source_label(chunk)} | effective {chunk.info.effective_date.isoformat()}"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


def outdated_sources(chunks: list[Chunk], differs: dict[int, list[int]]) -> list[int]:
    """A source is outdated if it differs from a source with higher weight (in either direction of the report)."""

    def heavier(a: int, b: int) -> bool:
        return weight(chunks[a - 1]) < weight(chunks[b - 1])

    pairs = {(a, b) for a, others in differs.items() for b in others if a != b}
    pairs |= {(b, a) for a, b in pairs}
    return sorted({a for a, b in pairs if heavier(b, a)})


def read_reply(chunks: list[Chunk], content: str) -> Reply | None:
    """Parse and validate the model's JSON; None if it is malformed."""

    def valid(n: int) -> bool:
        return 1 <= n <= len(chunks)

    try:
        data = json.loads(content)
        differs = {
            int(c["source"]): [int(m) for m in c["differs_from"] if valid(int(m))]
            for c in data["source_check"]
            if valid(int(c["source"]))
        }
        found = bool(data["found"])
        text = str(data["answer"]).strip()
        note = str(data["conflict_note"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    cited = [int(n) for group in SOURCE_MARKER.findall(text) for n in group.split(",")]
    outdated = outdated_sources(chunks, differs) if found else []
    return Reply(
        found=found,
        text=text,
        note=note,
        relevant=list(differs),
        outdated=outdated,
        cited=[n for n in dict.fromkeys(cited) if valid(n)],
    )


def build_answer(question: str, chunks: list[Chunk], reply: Reply | None) -> Answer:
    if reply is None:
        return Answer(question, ERROR_MESSAGE, False, (), (), tuple(chunks))

    def citations_for(numbers: list[int], exclude: list[int]) -> str:
        allowed = [n for n in numbers if 1 <= n <= len(chunks) and n not in exclude]
        return "; ".join(dict.fromkeys(chunks[n - 1].citation for n in allowed))

    def markers_to_citations(text: str, exclude: list[int]) -> str:
        def replace(match: re.Match) -> str:
            citations = citations_for([int(n) for n in match.group(1).split(",")], exclude)
            return f" ({citations})" if citations else ""

        return SOURCE_MARKER.sub(replace, text).strip()

    used = [n for n in reply.cited if n not in reply.outdated]
    text = markers_to_citations(reply.text, reply.outdated)
    if not text:
        text = NOT_FOUND_MESSAGE if not reply.found else ERROR_MESSAGE
    elif reply.found and not used:
        used = [n for n in reply.relevant if n not in reply.outdated]
        if used:
            text = f"{text} ({citations_for(used, [])})"

    return Answer(
        question=question,
        text=text,
        found=reply.found,
        sources=tuple(chunks[n - 1] for n in used),
        outdated=tuple(chunks[n - 1] for n in reply.outdated),
        retrieved=tuple(chunks),
        conflict_note=markers_to_citations(reply.note, []) if reply.outdated else "",
    )


class Answerer:
    def __init__(self, retriever: HybridRetriever, model: str = CHAT_MODEL, client: OpenAI | None = None):
        self.retriever = retriever
        self.model = model
        self.client = client or make_client()

    def answer(self, question: str, k: int = ANSWER_TOP_K, english_query: str | None = None) -> Answer:
        results = self.retriever.search(question, k, english_query)
        chunks = order_for_context([r.chunk for r in results])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(today=app_today().isoformat())},
            {"role": "user", "content": f"Sources:\n\n{build_context(chunks)}\n\nQuestion: {question}"},
        ]
        content = self._complete(messages)
        reply = read_reply(chunks, content)

        if reply is not None and reply.cites_outdated:
            outdated = ", ".join(f"[{n}]" for n in reply.outdated)
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": CORRECTION_PROMPT.format(outdated=outdated)},
            ]
            corrected = read_reply(chunks, self._complete(messages))
            if corrected is not None:
                # Keep the first reply's precedence: the rewrite may drop the outdated sources from source_check.
                reply = Reply(
                    found=corrected.found,
                    text=corrected.text,
                    note=corrected.note or reply.note,
                    relevant=sorted(set(corrected.relevant) | set(reply.relevant)),
                    outdated=reply.outdated,
                    cited=corrected.cited,
                )
        return build_answer(question, chunks, reply)

    def _complete(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, response_format=ANSWER_SCHEMA, messages=messages
        )
        return response.choices[0].message.content or ""
