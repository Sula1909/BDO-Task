"""Leave actions behind the MCP tools: listing, balances, proposals, confirmation and HR decisions.

Every function takes an open connection and returns a plain dict:
    success: {"ok": True, ...}
    failure: {"ok": False, "errors": [{"code", "article", "message", "redirect_to"}, ...]}
Expected business outcomes (not enough balance, wrong status...) are returned as data, never raised,
so the AI assistant always receives an explanation it can pass on to the employee.
"""

import sqlite3
import uuid
from dataclasses import asdict
from datetime import date

from hr_mcp import leave_rules as rules
from hr_mcp.config import app_now, app_today

REQUEST_STATUSES = ("pending", "approved", "rejected", "cancelled")
DECISIONS = ("approved", "rejected")


def _error(code: str, message: str, article: str = "", redirect_to: str | None = None) -> dict:
    return {"ok": False, "errors": [asdict(rules.Violation(code, article, message, redirect_to))]}


def _validation_failure(check: rules.ValidationResult) -> dict:
    return {"ok": False, "days": check.days, "errors": [asdict(v) for v in check.violations]}


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value.strip()) if value else None


def _get_request(conn: sqlite3.Connection, request_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM leave_requests WHERE request_id = ?", (request_id,)).fetchone()
    return dict(row) if row else None


def _submission_channel(leave_type: sqlite3.Row) -> str:
    if leave_type["assistant_supported"]:
        return "ASSISTANT_OR_HR_PORTAL"
    return rules.REDIRECT_HR_PORTAL_OR_HR if leave_type["self_service"] else rules.REDIRECT_HR


# ---------------------------------------------------------------------------
# Read-only actions (employee assistant and HR)
# ---------------------------------------------------------------------------

def list_leave_types(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT * FROM leave_types ORDER BY rowid").fetchall()
    return {
        "ok": True,
        "leave_types": [
            {
                "code": r["code"],
                "name": r["name"],
                "day_unit": r["day_unit"],
                "annual_limit_days": r["annual_limit_days"],
                "assistant_can_create": bool(r["assistant_supported"]),
                "submission_channel": _submission_channel(r),
                "policy_reference": r["policy_reference"],
            }
            for r in rows
        ],
    }


def get_leave_balance(
    conn: sqlite3.Connection, employee_id: str, leave_type: str | None = None, year: int | None = None
) -> dict:
    if rules.get_employee(conn, employee_id) is None:
        return _error("UNKNOWN_EMPLOYEE", f"Employee {employee_id} does not exist.")
    year = year or app_today().year

    if leave_type:
        lt = rules.get_leave_type(conn, leave_type.strip().upper())
        if lt is None:
            return _error("UNKNOWN_LEAVE_TYPE", f"Unknown leave type {leave_type!r}.", "3.1")
        types = [lt]
    else:
        types = conn.execute("SELECT * FROM leave_types ORDER BY rowid").fetchall()

    balances = []
    for lt in types:
        entry = rules.get_balance(conn, employee_id, lt["code"], year).to_dict()
        entry["policy_reference"] = lt["policy_reference"]
        if not entry["has_annual_balance"]:
            entry["note"] = "This leave type has no annual balance (Article 5.1); entitlement is decided per case."
        balances.append(entry)
    return {"ok": True, "employee_id": employee_id, "year": year, "balances": balances}


def list_leave_requests(
    conn: sqlite3.Connection,
    employee_id: str | None = None,
    status: str | None = None,
    leave_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """Requests matching all given filters. The date range keeps requests that overlap it."""
    if status and status not in REQUEST_STATUSES:
        return _error("INVALID_STATUS", f"Status must be one of: {', '.join(REQUEST_STATUSES)}.")
    try:
        start, end = _parse_date(date_from), _parse_date(date_to)
    except ValueError:
        return _error("INVALID_DATE", "Dates must be in YYYY-MM-DD format.")

    sql, params = "SELECT * FROM leave_requests WHERE 1 = 1", []
    if employee_id:
        sql += " AND employee_id = ?"
        params.append(employee_id)
    if status:
        sql += " AND status = ?"
        params.append(status)
    if leave_type:
        sql += " AND leave_type = ?"
        params.append(leave_type.strip().upper())
    if start:
        sql += " AND end_date >= ?"
        params.append(start.isoformat())
    if end:
        sql += " AND start_date <= ?"
        params.append(end.isoformat())

    rows = [dict(r) for r in conn.execute(sql + " ORDER BY start_date, request_id", params)]
    return {"ok": True, "count": len(rows), "requests": rows}


# ---------------------------------------------------------------------------
# Employee assistant: two-step creation (propose, then confirm)
# ---------------------------------------------------------------------------

def propose_leave_request(
    conn: sqlite3.Connection,
    employee_id: str,
    conversation_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    comment: str | None = None,
) -> dict:
    """Validate a request and store it as a proposal. Does NOT create a leave request."""
    if not conversation_id or not conversation_id.strip():
        return _error("CONVERSATION_REQUIRED", "conversation_id is required.")

    check = rules.validate_new_request(conn, employee_id, leave_type, start_date, end_date, comment)
    if not check.ok:
        return _validation_failure(check)

    leave_type = leave_type.strip().upper()
    start, end = _parse_date(start_date), _parse_date(end_date)
    comment = comment.strip() if comment and comment.strip() else None
    proposal_id = uuid.uuid4().hex
    with conn:
        conn.execute(
            """
            INSERT INTO leave_proposals
                (proposal_id, conversation_id, employee_id, leave_type, start_date, end_date, days, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (proposal_id, conversation_id, employee_id, leave_type, start.isoformat(), end.isoformat(),
             check.days, comment, app_now()),
        )

    lt = rules.get_leave_type(conn, leave_type)
    balance = rules.get_balance(conn, employee_id, leave_type, start.year)
    return {
        "ok": True,
        "proposal_id": proposal_id,
        "proposal": {
            "employee_id": employee_id,
            "leave_type": leave_type,
            "leave_type_name": lt["name"],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "days": check.days,
            "day_unit": lt["day_unit"],
            "comment": comment,
        },
        "available_before": balance.available_days,
        "available_after": balance.available_days - check.days,
        "next_step": (
            "Show the type, dates and number of days to the employee and ask for explicit confirmation. "
            "Call confirm_leave_request only after they confirm. The created request will be pending, not approved."
        ),
    }


def confirm_leave_request(conn: sqlite3.Connection, employee_id: str, conversation_id: str, proposal_id: str) -> dict:
    """Turn a confirmed proposal into a pending leave request. Calling it again returns the same request."""
    # BEGIN IMMEDIATE takes the database write lock now, so two simultaneous confirmations
    # of the same proposal run one after the other and the second sees the first one's result.
    conn.execute("BEGIN IMMEDIATE")
    try:
        p = conn.execute("SELECT * FROM leave_proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
        # Same answer for "missing" and "belongs to someone else", so proposal IDs reveal nothing.
        if p is None or p["employee_id"] != employee_id or p["conversation_id"] != conversation_id:
            conn.rollback()
            return _error("PROPOSAL_NOT_FOUND", "No such proposal for this employee and conversation.")

        if p["request_id"] is not None:
            conn.rollback()
            return {"ok": True, "already_confirmed": True, "request": _get_request(conn, p["request_id"])}

        # Re-check: other requests may have been created or approved since the proposal was made.
        check = rules.validate_new_request(
            conn, employee_id, p["leave_type"], p["start_date"], p["end_date"], p["comment"]
        )
        if not check.ok:
            conn.rollback()
            return _validation_failure(check)

        cursor = conn.execute(
            """
            INSERT INTO leave_requests
                (employee_id, leave_type, start_date, end_date, days, status, created_at, created_via, comment)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, 'assistant', ?)
            """,
            (employee_id, p["leave_type"], p["start_date"], p["end_date"], check.days, app_now(), p["comment"]),
        )
        request_id = cursor.lastrowid
        conn.execute("UPDATE leave_proposals SET request_id = ? WHERE proposal_id = ?", (request_id, proposal_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"ok": True, "already_confirmed": False, "request": _get_request(conn, request_id)}


# ---------------------------------------------------------------------------
# HR only: decisions and cancellation (the employee assistant may not do these, Article 12.3)
# ---------------------------------------------------------------------------

def decide_leave_request(conn: sqlite3.Connection, request_id: int, decision: str, comment: str | None = None) -> dict:
    if decision not in DECISIONS:
        return _error("INVALID_DECISION", "decision must be 'approved' or 'rejected'.")
    if decision == "rejected" and not (comment and comment.strip()):
        return _error("REASON_REQUIRED", "A rejection must state the reason in writing.", "13.2")

    request = _get_request(conn, request_id)
    if request is None:
        return _error("REQUEST_NOT_FOUND", f"Leave request #{request_id} does not exist.")
    if request["status"] != "pending":
        return _error("INVALID_STATUS", f"Only pending requests can be decided; #{request_id} is {request['status']}.")

    with conn:
        conn.execute(
            "UPDATE leave_requests SET status = ?, decided_at = ?, decision_comment = ? "
            "WHERE request_id = ? AND status = 'pending'",
            (decision, app_now(), comment.strip() if comment else None, request_id),
        )
    return {"ok": True, "request": _get_request(conn, request_id)}


def cancel_leave_request(
    conn: sqlite3.Connection, request_id: int, reason: str | None = None, manager_consent: bool = False
) -> dict:
    """Pending requests can always be cancelled; approved ones only up to 2 working days before the start (Article 4.8)."""
    request = _get_request(conn, request_id)
    if request is None:
        return _error("REQUEST_NOT_FOUND", f"Leave request #{request_id} does not exist.")
    if request["status"] not in ("pending", "approved"):
        return _error("INVALID_STATUS", f"Request #{request_id} is already {request['status']}.")

    if request["status"] == "approved" and not manager_consent:
        holidays = rules.load_holidays(conn)
        start = date.fromisoformat(request["start_date"])
        deadline = rules.subtract_working_days(start, rules.APPROVED_CANCELLATION_WORKING_DAYS, holidays)
        if app_today() > deadline:
            return _error(
                "LATE_CANCELLATION",
                f"Approved leave starting {start.isoformat()} could be cancelled until {deadline.isoformat()}. "
                "Later cancellation needs the manager's consent (call again with manager_consent=true once given).",
                "4.8",
                rules.REDIRECT_MANAGER,
            )

    with conn:
        conn.execute(
            "UPDATE leave_requests SET status = 'cancelled', decided_at = ?, decision_comment = ? "
            "WHERE request_id = ? AND status IN ('pending', 'approved')",
            (app_now(), reason.strip() if reason else None, request_id),
        )
    return {"ok": True, "request": _get_request(conn, request_id)}
