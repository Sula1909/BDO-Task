"""End-to-end tests through the MCP protocol: a real MCP client talks to the server in memory."""

import asyncio
import json

import pytest
from mcp import Client

from hr_mcp.server import mcp

EXPECTED_TOOLS = {
    "list_leave_types",
    "get_leave_balance",
    "list_leave_requests",
    "propose_leave_request",
    "confirm_leave_request",
    "hr_decide_leave_request",
    "hr_cancel_leave_request",
}


@pytest.fixture(autouse=True)
def server_uses_test_db(monkeypatch, db_path):
    monkeypatch.setattr("hr_mcp.db.DB_PATH", db_path)


async def call(client, tool, **arguments):
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


def run(coro):
    return asyncio.run(coro)


def test_server_lists_all_tools():
    async def scenario():
        async with Client(mcp) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == EXPECTED_TOOLS
            assert tools["get_leave_balance"].annotations.read_only_hint is True
            assert "employee_id" in tools["get_leave_balance"].input_schema["required"]

    run(scenario())


def test_balance_through_mcp():
    async def scenario():
        async with Client(mcp) as client:
            result = await call(client, "get_leave_balance", employee_id="E1001", leave_type="ANNUAL")
            assert result["balances"][0]["available_days"] == 10

    run(scenario())


def test_propose_and_confirm_through_mcp():
    async def scenario():
        async with Client(mcp) as client:
            ids = {"employee_id": "E1001", "conversation_id": "conv-1"}
            proposal = await call(
                client, "propose_leave_request", **ids, leave_type="ANNUAL", start_date="2026-11-02", end_date="2026-11-04"
            )
            first = await call(client, "confirm_leave_request", **ids, proposal_id=proposal["proposal_id"])
            second = await call(client, "confirm_leave_request", **ids, proposal_id=proposal["proposal_id"])
            assert first["request"]["request_id"] == second["request"]["request_id"] == 28
            assert second["already_confirmed"] is True

    run(scenario())


def test_missing_argument_is_rejected_by_mcp():
    async def scenario():
        async with Client(mcp) as client:
            result = await client.call_tool("get_leave_balance", {})
            assert result.is_error

    run(scenario())
