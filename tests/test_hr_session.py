"""HRSession against the real MCP server (in memory, fresh test database): the assistant's security boundary."""

import asyncio

import pytest
from mcp import Client

from assistant.hr_session import EMPLOYEE_TOOLS, HRSession, login
from hr_mcp.server import mcp

EMPLOYEE = "E1001"
ANNUAL = {"leave_type": "ANNUAL", "start_date": "2026-11-02", "end_date": "2026-11-04"}


@pytest.fixture(autouse=True)
def server_uses_test_db(monkeypatch, db_path):
    monkeypatch.setattr("hr_mcp.db.DB_PATH", db_path)


def run(scenario):
    async def wrapper():
        async with Client(mcp) as client:
            return await scenario(client)

    return asyncio.run(wrapper())


def test_login_accepts_known_employee_and_normalizes_id():
    async def scenario(client):
        session = await login(client, "  e1001 ")
        assert session is not None and session.employee_id == EMPLOYEE

    run(scenario)


@pytest.mark.parametrize("employee_id", ["E9999", "", "   "])
def test_login_rejects_unknown_employee(employee_id):
    async def scenario(client):
        assert await login(client, employee_id) is None

    run(scenario)


def test_hr_tools_are_refused():
    assert not any(tool.startswith("hr_") for tool in EMPLOYEE_TOOLS)

    async def scenario(client):
        session = HRSession(client, EMPLOYEE)
        with pytest.raises(PermissionError):
            await session.call("hr_decide_leave_request", request_id=1, decision="approved")

    run(scenario)


def test_requests_are_always_the_employees_own():
    async def scenario(client):
        result = await HRSession(client, EMPLOYEE).my_requests()
        assert result["ok"] and result["count"] > 0
        assert {r["employee_id"] for r in result["requests"]} == {EMPLOYEE}

    run(scenario)


def test_balance_uses_the_session_employee():
    async def scenario(client):
        result = await HRSession(client, EMPLOYEE).balance("ANNUAL")
        assert result["ok"] and result["employee_id"] == EMPLOYEE
        assert [b["leave_type"] for b in result["balances"]] == ["ANNUAL"]

    run(scenario)


def test_confirm_without_proposal_is_refused():
    async def scenario(client):
        session = HRSession(client, EMPLOYEE)
        session.new_turn()
        result = await session.confirm()
        assert result["errors"][0]["code"] == "NO_PROPOSAL"

    run(scenario)


def test_confirm_in_the_same_turn_as_the_proposal_is_refused():
    async def scenario(client):
        session = HRSession(client, EMPLOYEE)
        session.new_turn()
        assert (await session.propose(**ANNUAL))["ok"]
        result = await session.confirm()
        assert result["errors"][0]["code"] == "CONFIRMATION_REQUIRED"
        assert (await session.my_requests(status="pending", date_from="2026-11-02", date_to="2026-11-04"))["count"] == 0

    run(scenario)


def test_confirm_after_the_employee_replied_creates_a_pending_request():
    async def scenario(client):
        session = HRSession(client, EMPLOYEE)
        session.new_turn()
        await session.propose(**ANNUAL)
        session.new_turn()
        result = await session.confirm()
        assert result["ok"] and result["request"]["status"] == "pending"
        assert result["request"]["employee_id"] == EMPLOYEE
        assert session.pending is None
        assert (await session.confirm())["errors"][0]["code"] == "NO_PROPOSAL"

    run(scenario)


def test_proposal_expires_if_the_next_message_does_not_confirm_it():
    async def scenario(client):
        session = HRSession(client, EMPLOYEE)
        session.new_turn()
        await session.propose(**ANNUAL)
        assert session.awaiting_confirmation
        session.new_turn()
        assert session.pending is not None and not session.awaiting_confirmation
        session.new_turn()
        assert session.pending is None
        assert (await session.confirm())["errors"][0]["code"] == "NO_PROPOSAL"

    run(scenario)


def test_failed_proposal_leaves_nothing_to_confirm():
    async def scenario(client):
        session = HRSession(client, EMPLOYEE)
        session.new_turn()
        result = await session.propose("BEREAVEMENT", "2026-11-02", "2026-11-03")
        assert not result["ok"] and result["errors"][0]["code"] == "NOT_ASSISTANT_SUPPORTED"
        assert session.pending is None

    run(scenario)
