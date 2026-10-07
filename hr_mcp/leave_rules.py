"""Leave policy rules from Leave_and_Absence_Policy_v4.0 as plain, read-only functions.

Nothing in this module writes to the database or knows about MCP. The MCP tools call
these functions, which keeps the policy logic testable on its own.
"""

import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from hr_mcp.config import app_today

ACTIVE_STATUSES = ("approved", "pending")

ANNUAL_MAX_CONTINUOUS_DAYS = 15  # Article 4.5
UNPAID_NOTICE_WORKING_DAYS = 10  # Article 7.2
SICK_LATE_FILING_WORKING_DAYS = 2  # Article 6.2

AUDIT_DEPARTMENT = "AUD"
# Inclusive (month, day) ranges, Article 4.6.
AUDIT_RESTRICTED_PERIODS = (((1, 15), (3, 15)), ((12, 1), (12, 20)))

# Where the assistant sends the employee when it cannot create the request itself.
REDIRECT_HR = "HR"
REDIRECT_HR_PORTAL = "HR_PORTAL"
REDIRECT_HR_PORTAL_OR_HR = "HR_PORTAL_OR_HR"
REDIRECT_MANAGER = "MANAGER"


@dataclass(frozen=True)
class Violation:
    code: str
    article: str
    message: str
    redirect_to: str | None = None


@dataclass
class ValidationResult:
    days: int | None = None
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, code: str, article: str, message: str, redirect_to: str | None = None) -> None:
        self.violations.append(Violation(code, article, message, redirect_to))

    def to_dict(self) -> dict:
        return {"ok": self.ok, "days": self.days, "violations": [asdict(v) for v in self.violations]}


@dataclass(frozen=True)
class Balance:
    employee_id: str
    leave_type: str
    year: int
    has_annual_balance: bool
    entitled_days: int | None
    carried_over_days: int | None
    approved_days: int
    pending_days: int
    available_days: int | None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Calendar: working days and holidays (Articles 1.3, 2.2, 2.3, 4.4)
# ---------------------------------------------------------------------------

def load_holidays(conn: sqlite3.Connection) -> frozenset[date]:
    return frozenset(date.fromisoformat(row[0]) for row in conn.execute("SELECT date FROM public_holidays"))


def date_range(start: date, end: date) -> Iterator[date]:
    """Every date from start to end inclusive; empty if end < start."""
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def is_working_day(d: date, holidays: frozenset[date]) -> bool:
    return d.weekday() < 5 and d not in holidays


def count_leave_days(start: date, end: date, day_unit: str | None, holidays: frozenset[date]) -> int:
    if day_unit == "calendar":
        return max((end - start).days + 1, 0)
    if day_unit == "working":
        return sum(1 for d in date_range(start, end) if is_working_day(d, holidays))
    raise ValueError(f"Leave type has no countable day unit: {day_unit!r}")


def working_days_between(after: date, before: date, holidays: frozenset[date]) -> int:
    """Working days strictly between two dates, both excluded (Article 4.4 notice counting)."""
    return sum(1 for d in date_range(after + timedelta(days=1), before - timedelta(days=1)) if is_working_day(d, holidays))


def required_annual_notice(days: int) -> int | None:
    """Minimum notice in working days for an ANNUAL request; None if the request is too long to file (Article 4.4)."""
    if days <= 5:
        return 5
    if days <= ANNUAL_MAX_CONTINUOUS_DAYS:
        return 15
    return None


# ---------------------------------------------------------------------------
# Database lookups (read-only)
# ---------------------------------------------------------------------------

def get_employee(conn: sqlite3.Connection, employee_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()


def get_leave_type(conn: sqlite3.Connection, code: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM leave_types WHERE code = ?", (code,)).fetchone()


def get_active_requests(conn: sqlite3.Connection, employee_id: str, leave_type: str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM leave_requests WHERE employee_id = ? AND status IN ('approved', 'pending')"
    params: list = [employee_id]
    if leave_type:
        sql += " AND leave_type = ?"
        params.append(leave_type)
    return conn.execute(sql + " ORDER BY start_date", params).fetchall()


def get_balance(conn: sqlite3.Connection, employee_id: str, leave_type: str, year: int) -> Balance:
    """Article 5.1: available = entitled + carried_over - approved - pending, for requests starting in `year`."""
    approved, pending = conn.execute(
        """
        SELECT COALESCE(SUM(CASE WHEN status = 'approved' THEN days END), 0),
               COALESCE(SUM(CASE WHEN status = 'pending'  THEN days END), 0)
        FROM leave_requests
        WHERE employee_id = ? AND leave_type = ? AND start_date BETWEEN ? AND ?
        """,
        (employee_id, leave_type, f"{year}-01-01", f"{year}-12-31"),
    ).fetchone()

    entitlement = conn.execute(
        "SELECT entitled_days, carried_over_days FROM leave_entitlements WHERE employee_id = ? AND year = ? AND leave_type = ?",
        (employee_id, year, leave_type),
    ).fetchone()

    if entitlement is None:
        # BEREAVEMENT and PARENTAL have no annual balance (Article 5.1).
        return Balance(employee_id, leave_type, year, False, None, None, approved, pending, None)

    available = entitlement["entitled_days"] + entitlement["carried_over_days"] - approved - pending
    if leave_type == "SICK":
        available = max(available, 0)  # Article 5.1: the paid SICK balance never shows below zero.
    return Balance(
        employee_id, leave_type, year, True,
        entitlement["entitled_days"], entitlement["carried_over_days"], approved, pending, available,
    )


# ---------------------------------------------------------------------------
# Policy checks for a new assistant-created request (Article 12.3)
# ---------------------------------------------------------------------------

def continuous_annual_days(
    conn: sqlite3.Connection, employee_id: str, start: date, end: date, days: int, holidays: frozenset[date]
) -> int:
    """Total days of the continuous ANNUAL leave the new request would form with existing requests.

    Two requests are one continuous leave if only weekends or holidays lie between them (Article 4.5).
    """
    others = list(get_active_requests(conn, employee_id, "ANNUAL"))
    block_start, block_end, total = start, end, days
    merged = True
    while merged:
        merged = False
        for r in list(others):
            r_start, r_end = date.fromisoformat(r["start_date"]), date.fromisoformat(r["end_date"])
            touches_after = r_start > block_end and working_days_between(block_end, r_start, holidays) == 0
            touches_before = r_end < block_start and working_days_between(r_end, block_start, holidays) == 0
            if touches_after or touches_before:
                block_start, block_end = min(block_start, r_start), max(block_end, r_end)
                total += r["days"]
                others.remove(r)
                merged = True
    return total


def audit_restricted_period(start: date, end: date) -> tuple[date, date] | None:
    """The Article 4.6 restricted period the range touches, if any. Assumes start and end share a year."""
    for (m1, d1), (m2, d2) in AUDIT_RESTRICTED_PERIODS:
        period_start, period_end = date(start.year, m1, d1), date(start.year, m2, d2)
        if start <= period_end and end >= period_start:
            return period_start, period_end
    return None


def _parse_iso_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value.strip())


def validate_new_request(
    conn: sqlite3.Connection,
    employee_id: str,
    leave_type: str,
    start_date: str | date,
    end_date: str | date,
    comment: str | None = None,
    today: date | None = None,
) -> ValidationResult:
    """Check whether the assistant may create this request. Read-only: nothing is written."""
    today = today or app_today()
    result = ValidationResult()
    leave_type = leave_type.strip().upper()

    # Basic input checks: stop at the first failure, later checks depend on them.
    employee = get_employee(conn, employee_id)
    if employee is None:
        result.add("UNKNOWN_EMPLOYEE", "12.3", f"Employee {employee_id} does not exist.")
        return result
    if employee["status"] != "active":
        result.add("EMPLOYEE_INACTIVE", "1.2", f"Employee {employee_id} is not active.", REDIRECT_HR)
        return result

    lt = get_leave_type(conn, leave_type)
    if lt is None:
        result.add("UNKNOWN_LEAVE_TYPE", "3.1", f"Unknown leave type {leave_type!r}.")
        return result
    if not lt["assistant_supported"]:
        channel = REDIRECT_HR_PORTAL_OR_HR if lt["self_service"] else REDIRECT_HR
        result.add(
            "NOT_ASSISTANT_SUPPORTED", "12.3",
            f"The assistant cannot create {leave_type} requests ({lt['policy_reference']}). "
            f"Submit via {'the HR portal or HR' if lt['self_service'] else 'HR directly'}.",
            channel,
        )
        return result

    try:
        start, end = _parse_iso_date(start_date), _parse_iso_date(end_date)
    except ValueError:
        result.add("INVALID_DATE", "2.4", "Dates must be in YYYY-MM-DD format.")
        return result
    if end < start:
        result.add("INVALID_RANGE", "2.4", "The end date is before the start date.")
        return result

    if start.year != end.year:
        result.add(
            "CROSS_YEAR", "2.1",
            "Leave that spans two years needs a separate request for each year.",
        )
        return result
    if start.year > today.year:
        result.add(
            "NEXT_YEAR", "12.3",
            f"The assistant only creates requests for {today.year}. "
            f"Next-year requests can be submitted via the HR portal from {today.year}-12-01.",
            REDIRECT_HR_PORTAL,
        )
        return result
    if start.year < today.year:
        result.add("PAST_YEAR", "12.3", f"The assistant only creates requests for {today.year}.", REDIRECT_HR)
        return result

    holidays = load_holidays(conn)
    days = count_leave_days(start, end, lt["day_unit"], holidays)
    result.days = days
    if days == 0:
        result.add("ZERO_DAYS", "2.2", "The selected period contains no working days.")
        return result

    # Policy checks: collect every violation so the employee gets the full picture at once.
    overlapping = [
        r for r in get_active_requests(conn, employee_id)
        if r["start_date"] <= end.isoformat() and r["end_date"] >= start.isoformat()
    ]
    if overlapping:
        ids = ", ".join(f"#{r['request_id']} ({r['leave_type']}, {r['status']})" for r in overlapping)
        result.add("OVERLAP", "12.3", f"The dates overlap existing request(s): {ids}.")

    if leave_type == "ANNUAL":
        _check_annual(conn, result, employee, start, end, days, today, holidays)
    elif leave_type == "SICK":
        _check_sick(result, start, today, holidays)
    elif leave_type == "UNPAID":
        _check_unpaid(result, start, comment, today, holidays)

    balance = get_balance(conn, employee_id, leave_type, start.year)
    if not balance.has_annual_balance:
        result.add("NO_ENTITLEMENT", "5.3", f"No {leave_type} entitlement is recorded for {start.year}.", REDIRECT_HR)
    elif days > balance.available_days:
        if leave_type == "SICK":
            result.add(
                "SICK_PAID_LIMIT", "6.4",
                f"The request is {days} days but only {balance.available_days} paid sick days remain. "
                "Sick leave can still be recorded: HR registers it and sets the pay conditions individually.",
                REDIRECT_HR,
            )
        elif leave_type == "UNPAID":
            result.add(
                "UNPAID_LIMIT", "7.1",
                f"The request is {days} days but only {balance.available_days} of the 30 yearly unpaid days remain. "
                "Longer unpaid leave needs managing partner approval via HR.",
                REDIRECT_HR,
            )
        else:
            result.add(
                "INSUFFICIENT_BALANCE", "5.1",
                f"The request is {days} days but only {balance.available_days} {leave_type} days are available.",
            )

    return result


def _check_annual(conn, result, employee, start, end, days, today, holidays) -> None:
    probation_end = date.fromisoformat(employee["probation_end_date"])
    if start <= probation_end:
        result.add(
            "PROBATION", "4.3",
            f"Annual leave cannot be used during probation (ends {probation_end.isoformat()}). "
            "Exceptional cases of up to 2 days are handled by HR directly.",
            REDIRECT_HR,
        )

    required = required_annual_notice(days)
    if required is None:
        result.add(
            "MAX_CONTINUOUS", "4.5",
            f"A single request can cover at most {ANNUAL_MAX_CONTINUOUS_DAYS} working days. "
            "Longer leave needs partner or director approval via your manager.",
            REDIRECT_MANAGER,
        )
    else:
        notice = working_days_between(today, start, holidays)
        if notice < required:
            result.add(
                "NOTICE_PERIOD", "4.4",
                f"A {days}-day annual leave needs {required} working days' notice; this gives {notice}. "
                "For urgent cases, ask your manager directly.",
                REDIRECT_MANAGER,
            )
        continuous = continuous_annual_days(conn, employee["employee_id"], start, end, days, holidays)
        if continuous > ANNUAL_MAX_CONTINUOUS_DAYS:
            result.add(
                "MAX_CONTINUOUS", "4.5",
                f"Together with adjacent annual leave this makes {continuous} continuous working days "
                f"(maximum {ANNUAL_MAX_CONTINUOUS_DAYS}).",
                REDIRECT_MANAGER,
            )

    if employee["department_code"] == AUDIT_DEPARTMENT:
        period = audit_restricted_period(start, end)
        if period:
            result.add(
                "RESTRICTED_PERIOD", "4.6",
                f"{period[0].isoformat()} to {period[1].isoformat()} is a restricted period for the audit department. "
                "Annual leave then needs the project partner's written consent, arranged through your manager.",
                REDIRECT_MANAGER,
            )


def _check_sick(result, start, today, holidays) -> None:
    if start < today:
        days_late = sum(1 for d in date_range(start + timedelta(days=1), today) if is_working_day(d, holidays))
        if days_late > SICK_LATE_FILING_WORKING_DAYS:
            result.add(
                "SICK_LATE_FILING", "6.2",
                f"Sick leave must be recorded within {SICK_LATE_FILING_WORKING_DAYS} working days of the first day.",
                REDIRECT_HR,
            )


def _check_unpaid(result, start, comment, today, holidays) -> None:
    if not (comment and comment.strip()):
        result.add("REASON_REQUIRED", "7.2", "Unpaid leave requires a brief reason (no health details).")
    notice = working_days_between(today, start, holidays)
    if notice < UNPAID_NOTICE_WORKING_DAYS:
        result.add(
            "NOTICE_PERIOD", "7.2",
            f"Unpaid leave needs {UNPAID_NOTICE_WORKING_DAYS} working days' notice; this gives {notice}.",
            REDIRECT_MANAGER,
        )
