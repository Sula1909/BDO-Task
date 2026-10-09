"""Answer generation tests. OpenAI is replaced by a fake client that returns scripted JSON replies."""

import json
from types import SimpleNamespace

from rag.answering import (
    CONTACT_HR,
    ERROR_MESSAGE,
    NOT_FOUND_MESSAGE,
    Answerer,
    build_answer,
    build_context,
    order_for_context,
    outdated_sources,
    read_reply,
)
from rag.chunking import Chunk
from rag.retriever import SearchResult

POLICY = Chunk("leave_policy:6.3", "leave_policy", "article", "6.3", "სამედიცინო ცნობა", "6. ავადმყოფობა",
               "2 სამუშაო დღეზე მეტი")
HANDBOOK = Chunk("employee_handbook:5.3", "employee_handbook", "article", "5.3", "ავადმყოფობა", "5. შვებულება",
                 "3 სამუშაო დღეზე მეტი")
FAQ = Chunk("employee_faq:ა.6", "employee_faq", "article", "ა.6", "ავად ვარ", "ა. შვებულება", "3 დღე")
HISTORY = Chunk("leave_policy:15", "leave_policy", "history", "15", "დოკუმენტის ისტორია",
                "15. დოკუმენტის ისტორია", "ვერსია 3.0: 3 დღე")
REMOTE = Chunk("remote_policy:4.1", "remote_policy", "article", "4.1", "ლიმიტი", "4. კვირის ლიმიტი", "2 დღე")


def reply(answer, checks=(), found=True, note=""):
    return json.dumps({
        "source_check": [{"source": s, "states": "...", "differs_from": list(d)} for s, d in checks],
        "found": found,
        "answer": answer,
        "conflict_note": note,
    })


class FakeClient:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, *contents):
        self.contents = list(contents)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k, english_query=None):
        return [SearchResult(c, 1.0, None, None) for c in self.chunks[:k]]


# --- context ----------------------------------------------------------------


def test_context_orders_policy_handbook_faq_history():
    assert order_for_context([HISTORY, FAQ, HANDBOOK, POLICY]) == [POLICY, HANDBOOK, FAQ, HISTORY]


def test_context_numbers_and_labels_sources():
    context = build_context([POLICY, FAQ, HISTORY])
    assert context.startswith("[1] HR-POL-02 v4.0, მუხლი 6.3 | current policy")
    assert "[2] HR-FAQ-01 v1.4, კითხვა ა.6 | FAQ reference" in context
    assert "[3] HR-POL-02 v4.0, დოკუმენტის ისტორია | version history" in context
    assert "effective 2026-01-01" in context


# --- precedence -------------------------------------------------------------


def test_lower_weight_source_that_differs_is_outdated():
    chunks = [POLICY, HANDBOOK, FAQ]
    assert outdated_sources(chunks, {2: [1], 3: [1]}) == [2, 3]


def test_difference_reported_only_by_the_policy_side_still_counts():
    assert outdated_sources([POLICY, HANDBOOK], {1: [2]}) == [2]


def test_handbook_wins_over_faq():
    assert outdated_sources([HANDBOOK, FAQ], {2: [1]}) == [2]


def test_two_policies_that_differ_are_both_kept():
    assert outdated_sources([POLICY, REMOTE], {1: [2]}) == []


def test_history_is_outdated_against_a_policy():
    assert outdated_sources([POLICY, HISTORY], {2: [1]}) == [2]


# --- reply parsing ----------------------------------------------------------


def test_markers_become_citations_and_invalid_numbers_are_dropped():
    chunks = [POLICY, HANDBOOK]
    answer = build_answer("q", chunks, read_reply(chunks, reply("ცნობა საჭიროა [1, 7].", [(1, [])])))
    assert answer.text == "ცნობა საჭიროა (HR-POL-02 v4.0, მუხლი 6.3)."
    assert answer.sources == (POLICY,)


def test_outdated_marker_is_removed_from_text_and_listed_separately():
    chunks = [POLICY, HANDBOOK]
    content = reply("2 დღეზე მეტი [1, 2].", [(1, []), (2, [1])], note="Handbook-ის 3 დღე [2] მოძველებულია.")
    answer = build_answer("q", chunks, read_reply(chunks, content))
    assert answer.text == "2 დღეზე მეტი (HR-POL-02 v4.0, მუხლი 6.3)."
    assert answer.sources == (POLICY,)
    assert answer.outdated == (HANDBOOK,)
    assert answer.conflict_note == "Handbook-ის 3 დღე (HR-HB-01 v3.1, მუხლი 5.3) მოძველებულია."


def test_citation_is_appended_when_the_model_forgets_markers():
    chunks = [POLICY]
    answer = build_answer("q", chunks, read_reply(chunks, reply("ცნობა საჭიროა.", [(1, [])])))
    assert answer.text == "ცნობა საჭიროა. (HR-POL-02 v4.0, მუხლი 6.3)"


def test_not_found_drops_conflicts_and_suggests_hr():
    chunks = [POLICY, HANDBOOK]
    content = reply(NOT_FOUND_MESSAGE, [(2, [1])], found=False, note="invented conflict")
    answer = build_answer("q", chunks, read_reply(chunks, content))
    assert not answer.found
    assert answer.outdated == ()
    assert answer.conflict_note == ""
    assert CONTACT_HR in answer.format()


def test_malformed_reply_gives_error_message():
    chunks = [POLICY]
    assert read_reply(chunks, "not json") is None
    assert read_reply(chunks, json.dumps({"answer": "x"})) is None
    assert build_answer("q", chunks, None).text == ERROR_MESSAGE


def test_format_lists_sources_and_outdated():
    chunks = [POLICY, HANDBOOK]
    content = reply("2 დღე [1].", [(1, []), (2, [1])], note="3 დღე [2] მოძველებულია.")
    text = build_answer("q", chunks, read_reply(chunks, content)).format()
    assert "წყაროები:\n- HR-POL-02 v4.0, მუხლი 6.3: სამედიცინო ცნობა" in text
    assert "მოძველებული ინფორმაცია (არ გამოიყენება):\n- HR-HB-01 v3.1, მუხლი 5.3: ავადმყოფობა" in text


# --- the full answer call ---------------------------------------------------


def test_answer_makes_one_call_when_no_outdated_source_is_cited():
    client = FakeClient(reply("2 დღე [1].", [(1, []), (2, [1])]))
    answer = Answerer(FakeRetriever([HANDBOOK, POLICY]), client=client).answer("კითხვა")
    assert len(client.requests) == 1
    assert answer.sources == (POLICY,)
    assert answer.outdated == (HANDBOOK,)
    user_message = client.requests[0]["messages"][1]["content"]
    assert user_message.index("HR-POL-02") < user_message.index("HR-HB-01")
    assert user_message.endswith("Question: კითხვა")


def test_answer_asks_for_a_correction_when_it_cites_an_outdated_source():
    client = FakeClient(
        reply("3 დღეზე მეტი [1, 2].", [(1, []), (2, [1])]),
        reply("2 დღეზე მეტი [1].", [(1, [])], note="3 დღე [2] მოძველებულია."),
    )
    answer = Answerer(FakeRetriever([POLICY, HANDBOOK]), client=client).answer("კითხვა")
    assert len(client.requests) == 2
    assert "[2]" in client.requests[1]["messages"][-1]["content"]
    assert answer.text == "2 დღეზე მეტი (HR-POL-02 v4.0, მუხლი 6.3)."
    assert answer.outdated == (HANDBOOK,)
    assert answer.conflict_note == "3 დღე (HR-HB-01 v3.1, მუხლი 5.3) მოძველებულია."


def test_answer_uses_structured_output_and_app_date(monkeypatch):
    monkeypatch.setenv("APP_TODAY", "2026-10-19")
    client = FakeClient(reply("x [1].", [(1, [])]))
    Answerer(FakeRetriever([POLICY]), client=client).answer("q")
    request = client.requests[0]
    assert request["temperature"] == 0
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "Today is 2026-10-19" in request["messages"][0]["content"]
