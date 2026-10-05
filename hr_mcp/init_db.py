"""Build company_hr.db from the CSV files in data/.

The CSVs are never modified. Running this script again drops and recreates every table,
so the database always reflects the original data plus nothing else.

Usage: python -m hr_mcp.init_db
"""

import csv
from datetime import datetime
from pathlib import Path

from hr_mcp.config import DATA_DIR, DB_PATH
from hr_mcp.db import get_connection

SCHEMA = """
DROP TABLE IF EXISTS leave_proposals;
DROP TABLE IF EXISTS leave_requests;
DROP TABLE IF EXISTS leave_entitlements;
DROP TABLE IF EXISTS public_holidays;
DROP TABLE IF EXISTS leave_types;
DROP TABLE IF EXISTS employees;

CREATE TABLE employees (
    employee_id        TEXT PRIMARY KEY,
    full_name          TEXT NOT NULL,
    email              TEXT NOT NULL UNIQUE,
    department_code    TEXT NOT NULL,
    department_name    TEXT NOT NULL,
    job_title          TEXT NOT NULL,
    employment_type    TEXT NOT NULL,
    start_date         TEXT NOT NULL,
    probation_end_date TEXT NOT NULL,
    -- Deferred because a manager row can appear later in the CSV than their reports.
    manager_id         TEXT REFERENCES employees(employee_id) DEFERRABLE INITIALLY DEFERRED,
    status             TEXT NOT NULL
);

CREATE TABLE leave_types (
    code                TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    day_unit            TEXT CHECK (day_unit IN ('working', 'calendar')),
    annual_limit_days   INTEGER,
    self_service        INTEGER NOT NULL CHECK (self_service IN (0, 1)),
    assistant_supported INTEGER NOT NULL CHECK (assistant_supported IN (0, 1)),
    policy_reference    TEXT
);

CREATE TABLE leave_entitlements (
    employee_id       TEXT    NOT NULL REFERENCES employees(employee_id),
    year              INTEGER NOT NULL,
    leave_type        TEXT    NOT NULL REFERENCES leave_types(code),
    entitled_days     INTEGER NOT NULL CHECK (entitled_days >= 0),
    carried_over_days INTEGER NOT NULL DEFAULT 0 CHECK (carried_over_days >= 0),
    PRIMARY KEY (employee_id, year, leave_type)
);

CREATE TABLE leave_requests (
    request_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT    NOT NULL REFERENCES employees(employee_id),
    leave_type  TEXT    NOT NULL REFERENCES leave_types(code),
    start_date  TEXT    NOT NULL,
    end_date    TEXT    NOT NULL,
    days        INTEGER NOT NULL CHECK (days > 0),
    status      TEXT    NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
    created_at  TEXT    NOT NULL,
    created_via TEXT    NOT NULL CHECK (created_via IN ('portal', 'assistant')),
    comment     TEXT,
    CHECK (end_date >= start_date)
);

CREATE TABLE public_holidays (
    date TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

-- Drafts shown to the employee before they confirm (data dictionary: "conversation and confirmation").
-- request_id is filled on confirmation; UNIQUE guarantees one proposal can never create two requests.
CREATE TABLE leave_proposals (
    proposal_id     TEXT    PRIMARY KEY,
    conversation_id TEXT    NOT NULL,
    employee_id     TEXT    NOT NULL REFERENCES employees(employee_id),
    leave_type      TEXT    NOT NULL REFERENCES leave_types(code),
    start_date      TEXT    NOT NULL,
    end_date        TEXT    NOT NULL,
    days            INTEGER NOT NULL,
    comment         TEXT,
    created_at      TEXT    NOT NULL,
    request_id      INTEGER UNIQUE REFERENCES leave_requests(request_id)
);

CREATE INDEX idx_requests_employee_type_start ON leave_requests (employee_id, leave_type, start_date);
CREATE INDEX idx_requests_status ON leave_requests (status);
CREATE INDEX idx_proposals_conversation ON leave_proposals (employee_id, conversation_id);
"""

# The data dictionary says ISO dates, but the delivered CSVs use M/D/YYYY. Accept both.
DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y")


def parse_date(value: str, where: str) -> str:
    """Return the date as ISO YYYY-MM-DD, or raise with the file/row that failed."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"{where}: unrecognised date {value!r} (expected YYYY-MM-DD or M/D/YYYY)")


def empty_to_none(value: str) -> str | None:
    value = value.strip()
    return value or None


def int_or_none(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


def read_csv(name: str) -> list[dict[str, str]]:
    # utf-8-sig transparently strips a BOM if a file was ever saved with one.
    with open(DATA_DIR / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_employees(conn) -> int:
    rows = read_csv("employees.csv")
    conn.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["employee_id"].strip(),
                r["full_name"].strip(),
                r["email"].strip(),
                r["department_code"].strip(),
                r["department_name"].strip(),
                r["job_title"].strip(),
                r["employment_type"].strip(),
                parse_date(r["start_date"], f"employees.csv {r['employee_id']} start_date"),
                parse_date(r["probation_end_date"], f"employees.csv {r['employee_id']} probation_end_date"),
                empty_to_none(r["manager_id"]),
                r["status"].strip(),
            )
            for r in rows
        ],
    )
    return len(rows)


def load_leave_types(conn) -> int:
    rows = read_csv("leave_types.csv")
    conn.executemany(
        "INSERT INTO leave_types VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["code"].strip(),
                r["name"].strip(),
                empty_to_none(r["day_unit"]),
                int_or_none(r["annual_limit_days"]),
                int(r["self_service"]),
                int(r["assistant_supported"]),
                empty_to_none(r["policy_reference"]),
            )
            for r in rows
        ],
    )
    return len(rows)


def load_entitlements(conn) -> int:
    rows = read_csv("leave_entitlements.csv")
    conn.executemany(
        "INSERT INTO leave_entitlements VALUES (?, ?, ?, ?, ?)",
        [
            (
                r["employee_id"].strip(),
                int(r["year"]),
                r["leave_type"].strip(),
                int(r["entitled_days"]),
                int_or_none(r["carried_over_days"]) or 0,
            )
            for r in rows
        ],
    )
    return len(rows)


def load_leave_requests(conn) -> int:
    rows = read_csv("leave_requests.csv")
    conn.executemany(
        "INSERT INTO leave_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                int(r["request_id"]),
                r["employee_id"].strip(),
                r["leave_type"].strip(),
                parse_date(r["start_date"], f"leave_requests.csv #{r['request_id']} start_date"),
                parse_date(r["end_date"], f"leave_requests.csv #{r['request_id']} end_date"),
                int(r["days"]),
                r["status"].strip(),
                r["created_at"].strip(),
                r["created_via"].strip(),
                empty_to_none(r["comment"]),
            )
            for r in rows
        ],
    )
    return len(rows)


def load_public_holidays(conn) -> int:
    rows = read_csv("public_holidays.csv")
    conn.executemany(
        "INSERT INTO public_holidays VALUES (?, ?)",
        [(parse_date(r["date"], "public_holidays.csv"), r["name"].strip()) for r in rows],
    )
    return len(rows)


def build_database(db_path: Path | str = DB_PATH) -> dict[str, int]:
    conn = get_connection(db_path, must_exist=False)
    try:
        conn.executescript(SCHEMA)
        # One transaction: either every table loads, or nothing is committed.
        with conn:
            counts = {
                "employees": load_employees(conn),
                "leave_types": load_leave_types(conn),
                "leave_entitlements": load_entitlements(conn),
                "leave_requests": load_leave_requests(conn),
                "public_holidays": load_public_holidays(conn),
            }
        return counts
    finally:
        conn.close()


if __name__ == "__main__":
    counts = build_database()
    print(f"Built {DB_PATH}")
    for table, n in counts.items():
        print(f"  {table:<20} {n:>3} rows")
