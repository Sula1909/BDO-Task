"""Agent routing tests. OpenAI, the policy answerer and the HR session are fakes; nothing leaves the process."""

import asyncio
import json
from datetime import date
from types import SimpleNamespace

import pytest

from assistant.agent import FALLBACK_REPLY, MAX_TOOL_ROUNDS, TOOLS, Agent, calendar_lines


def tool_call(name, arguments=None, call_id="call-1"):
    function = SimpleNamespace(name=name, arguments=json.dumps(arguments or {}))
    return SimpleNamespace(id=call_id, function=function)


def message(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


class FakeOpenAI:
    """Returns the scripted messages in order (an Exception is raised instead) and records what it was sent."""

    def __init__(self, *messages):
        self.messages = list(messages)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        reply = self.messages.pop(0) if self.messages else message(calls=[tool_call("list_leave_types")])
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])


class FakeSession:
    employee_id = "E1001"

    def __init__(self):
        self.calls = []
        self.turn = 0
        self.pending = None

    def new_turn(self):
        self.turn += 1

    async def balance(self, leave_type=None, year=None):
        self.calls.append(("balance", leave_type, year))
        return {"ok": True, "balances": [{"leave_type": "ANNUAL", "available_days": 12}]}

    async def my_requests(self, status=None, date_from=None, date_to=None):
        self.calls.append(("my_requests", status, date_from, date_to))
        return {"ok": True, "count": 0, "requests": []}

    async def leave_types(self):
        self.calls.append(("leave_types",))
        return {"ok": True, "leave_types": []}

    async def propose(self, leave_type, start_date, end_date, comment=None):
        self.calls.append(("propose", leave_type, start_date, end_date, comment))
        self.pending = "p1"
        return {"ok": True, "proposal_id": "p1"}

    async def confirm(self):
        self.calls.append(("confirm",))
        self.pending = None
        return {"ok": True, "request": {"request_id": 99, "status": "pending"}}


class FakeAnswerer:
    def __init__(self):
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        return SimpleNamespace(found=True, format=lambda: "15 სამუშაო დღე (HR-POL-02 v4.0, მუხლი 4.4)")


def make_agent(*messages):
    session, answerer, client = FakeSession(), FakeAnswerer(), FakeOpenAI(*messages)
    return Agent(session, answerer, client=client), session, answerer, client


def reply(agent, text):
    return asyncio.run(agent.reply(text))


def test_tools_never_expose_hr_actions_or_employee_id():
    names = {t["function"]["name"] for t in TOOLS}
    assert not any(name.startswith("hr_") for name in names)
    for tool in TOOLS:
        assert "employee_id" not in tool["function"]["parameters"]["properties"]
    confirm = next(t for t in TOOLS if t["function"]["name"] == "confirm_leave_request")
    assert confirm["function"]["parameters"]["properties"] == {}


def test_system_prompt_has_employee_date_and_calendar(monkeypatch):
    monkeypatch.setenv("APP_TODAY", "2026-10-19")
    agent, *_ = make_agent()
    assert "employee E1001" in agent.system_prompt
    assert "Today is 2026-10-19 (ორშაბათი)" in agent.system_prompt
    assert "Tomorrow (ხვალ) is 2026-10-20 სამშაბათი" in agent.system_prompt
    assert "from 2026-10-26 ორშაბათი to 2026-11-01 კვირა" in agent.system_prompt
    assert "2026-10-26 ორშაბათი | 2026-10-27 სამშაბათი" in agent.system_prompt


def test_calendar_runs_from_this_monday_to_year_end():
    lines = calendar_lines(date(2026, 10, 21)).splitlines()
    assert lines[0].startswith("2026-10-19 ორშაბათი")
    assert lines[-1].endswith("2026-12-31 ხუთშაბათი")


def test_plain_reply_without_tools():
    agent, session, _, client = make_agent(message("გამარჯობა!"))
    result = reply(agent, "გამარჯობა")
    assert result.text == "გამარჯობა!" and result.events == ()
    assert session.turn == 1 and session.calls == []
    assert client.requests[0][0]["role"] == "system"


def test_policy_question_goes_to_the_answerer_and_result_returns_to_the_model():
    agent, _, answerer, client = make_agent(
        message(calls=[tool_call("search_policies", {"question": "რამდენი დღით ადრე?"})]),
        message("15 სამუშაო დღე (HR-POL-02 v4.0, მუხლი 4.4)"),
    )
    assert reply(agent, "რამდენი დღით ადრე?").text == "15 სამუშაო დღე (HR-POL-02 v4.0, მუხლი 4.4)"
    assert answerer.questions == ["რამდენი დღით ადრე?"]
    tool_message = client.requests[1][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call-1"
    assert "მუხლი 4.4" in json.loads(tool_message["content"])["answer"]


def test_two_tools_in_one_round():
    agent, session, answerer, client = make_agent(
        message(calls=[tool_call("get_leave_balance", {"leave_type": "ANNUAL"}, "a"),
                       tool_call("search_policies", {"question": "q"}, "b")]),
        message("done"),
    )
    result = reply(agent, "რამდენი დღე მაქვს და როდის უნდა მოვითხოვო?")
    assert session.calls == [("balance", "ANNUAL", None)]
    assert answerer.questions == ["q"]
    assert [m["tool_call_id"] for m in client.requests[1] if m["role"] == "tool"] == ["a", "b"]
    assert [e.name for e in result.events] == ["get_leave_balance", "search_policies"]


def test_leave_request_flow_routes_to_session_and_reports_results():
    agent, session, *_ = make_agent(
        message(calls=[tool_call("propose_leave_request",
                                 {"leave_type": "ANNUAL", "start_date": "2026-11-02", "end_date": "2026-11-04"})]),
        message("დაადასტურებთ?"),
        message(calls=[tool_call("confirm_leave_request")]),
        message("მოთხოვნა გაგზავნილია."),
    )
    first = reply(agent, "შვებულება მინდა 2-დან 4 ნოემბრამდე")
    assert first.last_result("propose_leave_request")["proposal_id"] == "p1"
    assert first.last_result("confirm_leave_request") is None
    second = reply(agent, "კი")
    assert second.last_result("confirm_leave_request")["request"]["request_id"] == 99
    assert session.calls == [("propose", "ANNUAL", "2026-11-02", "2026-11-04", None), ("confirm",)]
    assert session.turn == 2


def test_failed_api_call_leaves_no_trace_in_history():
    agent, *_ = make_agent(
        message("პირველი"),
        message(calls=[tool_call("get_leave_balance")]),
        ConnectionError("network down"),
        message("მეორე"),
    )
    reply(agent, "ერთი")
    with pytest.raises(ConnectionError):
        reply(agent, "ორი")
    assert [m["content"] for m in agent.history] == ["ერთი", "პირველი"]
    assert reply(agent, "სამი").text == "მეორე"
    assert [m["content"] for m in agent.history] == ["ერთი", "პირველი", "სამი", "მეორე"]


def test_reset_forgets_history_and_pending_proposal():
    agent, session, *_ = make_agent(message("ok"))
    reply(agent, "გამარჯობა")
    session.pending = "p1"
    agent.reset()
    assert agent.history == [] and session.pending is None


def test_future_sick_leave_needs_the_procedure_question(monkeypatch):
    monkeypatch.setenv("APP_TODAY", "2026-10-19")
    agent, session, *_ = make_agent()
    future = json.dumps({"leave_type": "SICK", "start_date": "2026-10-26", "end_date": "2026-10-27"})
    result = asyncio.run(agent.run_tool("propose_leave_request", future))
    assert result["errors"][0]["code"] == "ASK_EMPLOYEE_FIRST"
    assert session.calls == []

    confirmed = json.dumps({"leave_type": "SICK", "start_date": "2026-10-26", "end_date": "2026-10-27",
                            "planned_procedure": True})
    assert asyncio.run(agent.run_tool("propose_leave_request", confirmed))["ok"]
    today = json.dumps({"leave_type": "SICK", "start_date": "2026-10-19", "end_date": "2026-10-19"})
    assert asyncio.run(agent.run_tool("propose_leave_request", today))["ok"]
    assert [c[0] for c in session.calls] == ["propose", "propose"]


def test_unknown_tool_and_bad_arguments_return_errors_to_the_model():
    agent, *_ = make_agent()
    assert asyncio.run(agent.run_tool("hr_decide_leave_request", "{}"))["errors"][0]["code"] == "UNKNOWN_TOOL"
    assert asyncio.run(agent.run_tool("get_leave_balance", "not json"))["errors"][0]["code"] == "BAD_ARGUMENTS"
    assert asyncio.run(agent.run_tool("get_leave_balance", "[1]"))["errors"][0]["code"] == "BAD_ARGUMENTS"


def test_endless_tool_calls_stop_with_fallback():
    agent, session, *_ = make_agent()
    assert reply(agent, "?").text == FALLBACK_REPLY
    assert len(session.calls) == MAX_TOOL_ROUNDS


def test_history_is_trimmed_at_a_user_message():
    agent, *_ = make_agent()
    for i in range(30):
        agent.history += [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "x", "content": "{}"},
        ]
    messages = agent._messages()
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert len(messages) <= 41
