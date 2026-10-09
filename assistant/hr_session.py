"""The logged-in employee's connection to the Task 1 HR MCP server.

This is the security boundary of the assistant (Leave policy, Article 12.3):
- only the employee tools are reachable; hr_* tools are refused here, whatever the model asks;
- the employee ID is fixed at login and added to every call, so the model can never act for someone else;
- confirm_leave_request takes no proposal ID from the model: it confirms the session's latest proposal,
  and only in the employee's next message after that proposal; any other reply discards it.
"""

import json
import os
import sys
import uuid
from dataclasses import dataclass

from mcp import Client, StdioServerParameters

from hr_mcp.config import BASE_DIR

EMPLOYEE_TOOLS = frozenset(
    {"list_leave_types", "get_leave_balance", "list_leave_requests", "propose_leave_request", "confirm_leave_request"}
)
PASSED_ENV = ("APP_TODAY", "HR_DB_PATH")


def server_parameters() -> StdioServerParameters:
    """Launch the MCP server as a subprocess (stdio), with the same database and app date as this process."""
    env = {name: os.environ[name] for name in PASSED_ENV if name in os.environ}
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "hr_mcp.server"],
        cwd=str(BASE_DIR),
        env={**env, "PYTHONIOENCODING": "utf-8"},
    )


def error(code: str, message: str) -> dict:
    """Same shape as the server's own failures, so the model handles both the same way."""
    return {"ok": False, "errors": [{"code": code, "article": "", "message": message, "redirect_to": None}]}


@dataclass(frozen=True)
class PendingProposal:
    proposal_id: str
    turn: int


class HRSession:
    def __init__(self, client: Client, employee_id: str, conversation_id: str | None = None):
        self.client = client
        self.employee_id = employee_id
        self.conversation_id = conversation_id or uuid.uuid4().hex
        self.turn = 0
        self.pending: PendingProposal | None = None

    def new_turn(self) -> None:
        """Call once for every message the employee sends. A proposal can only be confirmed in the next one."""
        self.turn += 1
        if self.pending is not None and self.pending.turn < self.turn - 1:
            self.pending = None

    @property
    def awaiting_confirmation(self) -> bool:
        """True right after a proposal: the employee's next message is the only one that can confirm it."""
        return self.pending is not None and self.pending.turn == self.turn

    async def call(self, tool: str, **arguments) -> dict:
        if tool not in EMPLOYEE_TOOLS:
            raise PermissionError(f"{tool} is not available to the employee assistant")
        result = await self.client.call_tool(tool, arguments)
        text = result.content[0].text if result.content else ""
        if result.is_error:
            return error("TOOL_ERROR", text or f"{tool} failed")
        return json.loads(text)

    async def leave_types(self) -> dict:
        return await self.call("list_leave_types")

    async def balance(self, leave_type: str | None = None, year: int | None = None) -> dict:
        arguments = {"employee_id": self.employee_id}
        if leave_type:
            arguments["leave_type"] = leave_type
        if year:
            arguments["year"] = year
        return await self.call("get_leave_balance", **arguments)

    async def my_requests(
        self, status: str | None = None, date_from: str | None = None, date_to: str | None = None
    ) -> dict:
        filters = {"status": status, "date_from": date_from, "date_to": date_to}
        arguments = {name: value for name, value in filters.items() if value}
        return await self.call("list_leave_requests", employee_id=self.employee_id, **arguments)

    async def propose(self, leave_type: str, start_date: str, end_date: str, comment: str | None = None) -> dict:
        arguments = {
            "employee_id": self.employee_id,
            "conversation_id": self.conversation_id,
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date,
        }
        if comment:
            arguments["comment"] = comment
        result = await self.call("propose_leave_request", **arguments)
        self.pending = PendingProposal(result["proposal_id"], self.turn) if result.get("ok") else None
        return result

    async def confirm(self) -> dict:
        if self.pending is None:
            return error("NO_PROPOSAL", "There is no proposal to confirm. Call propose_leave_request first.")
        if self.pending.turn >= self.turn:
            return error(
                "CONFIRMATION_REQUIRED",
                "Show the proposal to the employee and wait for their reply before confirming.",
            )
        result = await self.call(
            "confirm_leave_request",
            employee_id=self.employee_id,
            conversation_id=self.conversation_id,
            proposal_id=self.pending.proposal_id,
        )
        if result.get("ok"):
            self.pending = None
        return result


async def login(client: Client, employee_id: str) -> HRSession | None:
    """Open a session for an existing employee; None if the ID is unknown."""
    employee_id = employee_id.strip().upper()
    if not employee_id:
        return None
    session = HRSession(client, employee_id)
    result = await session.balance()
    return session if result.get("ok") else None
