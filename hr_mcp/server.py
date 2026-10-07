"""Northstar Services HR MCP server.

Exposes the leave actions in leave_service.py as MCP tools. Each tool opens its own database
connection, calls the service function and returns its dict, which MCP sends to the client as JSON.

Run (stdio transport, launched by an MCP client): python -m hr_mcp.server
"""

from contextlib import closing
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from hr_mcp import leave_service as service
from hr_mcp.config import app_today
from hr_mcp.db import get_connection

INSTRUCTIONS = f"""\
HR leave management for Northstar Services (Leave and Absence Policy v4.0).
Today's date for all leave rules is {app_today().isoformat()}. Use it instead of your own clock.

Tools for the employee assistant: list_leave_types, get_leave_balance, list_leave_requests,
propose_leave_request, confirm_leave_request. An employee may only see and act on their own data.

Creating a request is two steps: propose_leave_request validates it against the policy and returns a
proposal; show it to the employee and call confirm_leave_request only after their explicit confirmation.
The assistant can create ANNUAL, SICK and UNPAID requests only. For other types, or when a tool returns
errors, explain the reason and the policy article and redirect the employee as indicated in redirect_to.

Tools prefixed hr_ are for HR staff only. The employee assistant must never call them (Article 12.3).
"""

mcp = MCPServer("northstar-hr", instructions=INSTRUCTIONS)

EmployeeId = Annotated[str, Field(description="Employee ID, e.g. E1001.")]
IsoDate = Annotated[str, Field(description="Date in YYYY-MM-DD format.")]
OptionalIsoDate = Annotated[str | None, Field(description="Date in YYYY-MM-DD format.")]
LeaveTypeCode = Annotated[str, Field(description="Leave type code: ANNUAL, SICK, UNPAID, BEREAVEMENT, STUDY or PARENTAL.")]
ConversationId = Annotated[str, Field(description="ID of the current conversation with the employee.")]
RequestId = Annotated[int, Field(description="Leave request ID.")]

READ_ONLY = ToolAnnotations(read_only_hint=True)


# ---------------------------------------------------------------------------
# Employee assistant and HR
# ---------------------------------------------------------------------------

@mcp.tool(annotations=READ_ONLY)
def list_leave_types() -> dict:
    """List all leave types with their day unit, yearly limit, whether the assistant can create them,
    the channel for submitting them and the policy article."""
    with closing(get_connection()) as conn:
        return service.list_leave_types(conn)


@mcp.tool(annotations=READ_ONLY)
def get_leave_balance(
    employee_id: EmployeeId,
    leave_type: Annotated[
        str | None,
        Field(description="Leave type code (ANNUAL, SICK, UNPAID, BEREAVEMENT, STUDY, PARENTAL); omit to get every type."),
    ] = None,
    year: Annotated[int | None, Field(description="Leave year; defaults to the current year.")] = None,
) -> dict:
    """Remaining leave days per Article 5.1: entitled + carried over - approved - pending.
    Shows approved and pending days separately. SICK shows remaining paid days (never below 0)."""
    with closing(get_connection()) as conn:
        return service.get_leave_balance(conn, employee_id, leave_type, year)


@mcp.tool(annotations=READ_ONLY)
def list_leave_requests(
    employee_id: Annotated[str | None, Field(description="Only this employee's requests.")] = None,
    status: Annotated[
        Literal["pending", "approved", "rejected", "cancelled"] | None, Field(description="Only this status.")
    ] = None,
    leave_type: Annotated[str | None, Field(description="Only this leave type code.")] = None,
    date_from: OptionalIsoDate = None,
    date_to: OptionalIsoDate = None,
) -> dict:
    """List leave requests matching all given filters. date_from/date_to keep requests that overlap the range."""
    with closing(get_connection()) as conn:
        return service.list_leave_requests(conn, employee_id, status, leave_type, date_from, date_to)


# ---------------------------------------------------------------------------
# Employee assistant: create a request in two steps
# ---------------------------------------------------------------------------

@mcp.tool()
def propose_leave_request(
    employee_id: EmployeeId,
    conversation_id: ConversationId,
    leave_type: LeaveTypeCode,
    start_date: IsoDate,
    end_date: IsoDate,
    comment: Annotated[
        str | None, Field(description="Brief reason. Required for UNPAID. Never include health details.")
    ] = None,
) -> dict:
    """Step 1 of creating a leave request. Checks the request against the leave policy and saves it as a
    proposal. Does NOT create the request. On success, show the returned details to the employee and ask
    for explicit confirmation. On failure, explain each error with its article and redirect_to."""
    with closing(get_connection()) as conn:
        return service.propose_leave_request(
            conn, employee_id, conversation_id, leave_type, start_date, end_date, comment
        )


@mcp.tool(annotations=ToolAnnotations(idempotent_hint=True))
def confirm_leave_request(
    employee_id: EmployeeId,
    conversation_id: ConversationId,
    proposal_id: Annotated[str, Field(description="proposal_id returned by propose_leave_request.")],
) -> dict:
    """Step 2: create the pending leave request from a proposal the employee explicitly confirmed.
    Safe to repeat: confirming the same proposal again returns the same request and creates nothing new."""
    with closing(get_connection()) as conn:
        return service.confirm_leave_request(conn, employee_id, conversation_id, proposal_id)


# ---------------------------------------------------------------------------
# HR only
# ---------------------------------------------------------------------------

@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
def hr_decide_leave_request(
    request_id: RequestId,
    decision: Literal["approved", "rejected"],
    comment: Annotated[str | None, Field(description="Decision note. Required when rejecting (Article 13.2).")] = None,
) -> dict:
    """HR/manager only: approve or reject a pending leave request. Never call from the employee assistant."""
    with closing(get_connection()) as conn:
        return service.decide_leave_request(conn, request_id, decision, comment)


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
def hr_cancel_leave_request(
    request_id: RequestId,
    reason: Annotated[str | None, Field(description="Why the request is cancelled.")] = None,
    manager_consent: Annotated[
        bool, Field(description="Set true only if the manager agreed to a late cancellation (Article 4.8).")
    ] = False,
) -> dict:
    """HR/portal only: cancel a pending or approved leave request. Approved leave can be cancelled until
    2 working days before it starts; later needs manager consent. Never call from the employee assistant."""
    with closing(get_connection()) as conn:
        return service.cancel_leave_request(conn, request_id, reason, manager_consent)


if __name__ == "__main__":
    mcp.run()  # stdio: messages travel over stdin/stdout, so this process must never print() to stdout
