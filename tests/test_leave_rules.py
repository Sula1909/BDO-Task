from datetime import date

import pytest

from hr_mcp.leave_rules import (
    continuous_annual_days,
    count_leave_days,
    get_balance,
    load_holidays,
    validate_new_request,
    working_days_between,
)
from tests.conftest import TODAY


def codes(result):
    return {v.code for v in result.violations}


def validate(conn, employee_id, leave_type, start, end, comment=None):
    return validate_new_request(conn, employee_id, leave_type, start, end, comment, today=TODAY)


# --- Day counting -----------------------------------------------------------

def test_policy_example_holiday_on_tuesday(conn):
    # Article 2.2 example: Monday to the following Monday with a Tuesday holiday = 5 days.
    # 2026-03-03 (Mother's Day) is a Tuesday; 2026-03-08 falls on a Sunday.
    holidays = load_holidays(conn)
    assert count_leave_days(date(2026, 3, 2), date(2026, 3, 9), "working", holidays) == 5


def test_calendar_days_include_weekends(conn):
    holidays = load_holidays(conn)
    assert count_leave_days(date(2026, 11, 6), date(2026, 11, 9), "calendar", holidays) == 4


def test_days_column_of_every_csv_request_matches(conn):
    holidays = load_holidays(conn)
    rows = conn.execute(
        "SELECT r.*, t.day_unit FROM leave_requests r JOIN leave_types t ON t.code = r.leave_type"
    ).fetchall()
    for r in rows:
        counted = count_leave_days(date.fromisoformat(r["start_date"]), date.fromisoformat(r["end_date"]), r["day_unit"], holidays)
        assert counted == r["days"], f"request #{r['request_id']}"


def test_notice_policy_example(conn):
    # Article 4.4 example: submitted on Monday, a 3-day leave can start next week's Tuesday at the earliest.
    holidays = load_holidays(conn)
    assert working_days_between(TODAY, date(2026, 10, 27), holidays) == 5
    assert working_days_between(TODAY, date(2026, 10, 26), holidays) == 4


# --- Balance (Article 5.1) --------------------------------------------------

@pytest.mark.parametrize(
    "employee_id, leave_type, approved, pending, available",
    [
        ("E1001", "ANNUAL", 15, 3, 10),  # 25 + 3 carried over
        ("E1001", "SICK", 2, 0, 8),
        ("E1002", "ANNUAL", 20, 0, 4),  # the rejected request #8 is not counted
        ("E1009", "STUDY", 1, 0, 4),
    ],
)
def test_balance(conn, employee_id, leave_type, approved, pending, available):
    b = get_balance(conn, employee_id, leave_type, 2026)
    assert (b.approved_days, b.pending_days, b.available_days) == (approved, pending, available)


def test_bereavement_has_no_annual_balance(conn):
    b = get_balance(conn, "E1001", "BEREAVEMENT", 2026)
    assert b.has_annual_balance is False
    assert b.available_days is None


# --- Valid requests ---------------------------------------------------------

def test_valid_annual(conn):
    r = validate(conn, "E1001", "ANNUAL", "2026-11-02", "2026-11-04")
    assert r.ok, r.violations
    assert r.days == 3


def test_valid_unpaid_with_reason(conn):
    r = validate(conn, "E1001", "UNPAID", "2026-11-03", "2026-11-05", comment="Family matter")
    assert r.ok, r.violations
    assert r.days == 3


def test_valid_sick_filed_two_working_days_late(conn):
    # Started Thursday 10-15; Friday 10-16 and Monday 10-19 are the 2 allowed working days.
    r = validate(conn, "E1001", "SICK", "2026-10-15", "2026-10-16")
    assert r.ok, r.violations


def test_leave_type_is_case_insensitive(conn):
    assert validate(conn, "E1001", "annual", "2026-11-02", "2026-11-04").ok


def test_probation_over_allows_annual(conn):
    r = validate(conn, "E1004", "ANNUAL", "2026-12-01", "2026-12-02")
    assert r.ok, r.violations


# --- Rejections -------------------------------------------------------------

@pytest.mark.parametrize(
    "leave_type, redirect",
    [("BEREAVEMENT", "HR_PORTAL_OR_HR"), ("STUDY", "HR_PORTAL_OR_HR"), ("PARENTAL", "HR")],
)
def test_types_the_assistant_cannot_create(conn, leave_type, redirect):
    r = validate(conn, "E1001", leave_type, "2026-11-02", "2026-11-03")
    assert codes(r) == {"NOT_ASSISTANT_SUPPORTED"}
    assert r.violations[0].redirect_to == redirect


def test_unknown_employee(conn):
    assert codes(validate(conn, "E9999", "ANNUAL", "2026-11-02", "2026-11-04")) == {"UNKNOWN_EMPLOYEE"}


def test_unknown_leave_type(conn):
    assert codes(validate(conn, "E1001", "HOLIDAY", "2026-11-02", "2026-11-04")) == {"UNKNOWN_LEAVE_TYPE"}


def test_invalid_date_format(conn):
    assert codes(validate(conn, "E1001", "ANNUAL", "11/02/2026", "2026-11-04")) == {"INVALID_DATE"}


def test_end_before_start(conn):
    assert codes(validate(conn, "E1001", "ANNUAL", "2026-11-04", "2026-11-02")) == {"INVALID_RANGE"}


def test_cross_year(conn):
    assert codes(validate(conn, "E1001", "ANNUAL", "2026-12-28", "2027-01-08")) == {"CROSS_YEAR"}


def test_next_year(conn):
    assert codes(validate(conn, "E1001", "ANNUAL", "2027-02-01", "2027-02-03")) == {"NEXT_YEAR"}


def test_weekend_only(conn):
    assert codes(validate(conn, "E1001", "ANNUAL", "2026-10-31", "2026-11-01")) == {"ZERO_DAYS"}


def test_overlap_with_pending_request(conn):
    # E1001 already has pending request #5 for 2026-11-09 to 11-11.
    r = validate(conn, "E1001", "ANNUAL", "2026-11-10", "2026-11-10")
    assert "OVERLAP" in codes(r)


def test_annual_notice_too_short(conn):
    r = validate(conn, "E1001", "ANNUAL", "2026-10-26", "2026-10-27")
    assert codes(r) == {"NOTICE_PERIOD"}


def test_annual_during_probation(conn):
    # E1004's probation ends 2026-11-30.
    r = validate(conn, "E1004", "ANNUAL", "2026-11-16", "2026-11-17")
    assert codes(r) == {"PROBATION"}


def test_audit_restricted_period(conn):
    r = validate(conn, "E1005", "ANNUAL", "2026-12-01", "2026-12-02")
    assert codes(r) == {"RESTRICTED_PERIOD"}


def test_same_dates_fine_outside_audit(conn):
    assert validate(conn, "E1006", "ANNUAL", "2026-12-01", "2026-12-02").ok


def test_continuous_leave_counts_adjacent_requests(conn):
    # New Mon-Fri 11-02..11-06 + weekend + pending #5 Mon-Wed 11-09..11-11 = one 8-day leave.
    holidays = load_holidays(conn)
    assert continuous_annual_days(conn, "E1001", date(2026, 11, 2), date(2026, 11, 6), 5, holidays) == 8


def test_continuous_leave_over_15_days(conn):
    # E1006 has pending #15 for 11-16..11-18 (3 days); 11-19..12-08 adds 13 more = 16.
    r = validate(conn, "E1006", "ANNUAL", "2026-11-19", "2026-12-08")
    assert "MAX_CONTINUOUS" in codes(r)


def test_single_request_over_15_days(conn):
    r = validate(conn, "E1003", "ANNUAL", "2026-11-24", "2026-12-15")
    assert r.days == 16
    assert "MAX_CONTINUOUS" in codes(r)


def test_annual_insufficient_balance(conn):
    # E1002 has 4 ANNUAL days left.
    r = validate(conn, "E1002", "ANNUAL", "2026-11-02", "2026-11-06")
    assert codes(r) == {"INSUFFICIENT_BALANCE"}


def test_sick_over_paid_limit_redirects_to_hr(conn):
    # E1001 has 8 paid sick days left; 10-19..10-30 is 10 working days.
    r = validate(conn, "E1001", "SICK", "2026-10-19", "2026-10-30")
    assert codes(r) == {"SICK_PAID_LIMIT"}
    v = r.violations[0]
    assert v.redirect_to == "HR"
    assert "can still be recorded" in v.message


def test_sick_filed_too_late(conn):
    # Started Tuesday 10-13; 10-14 is a holiday, so 10-15, 10-16, 10-19 = 3 working days late.
    r = validate(conn, "E1001", "SICK", "2026-10-13", "2026-10-13")
    assert codes(r) == {"SICK_LATE_FILING"}


def test_unpaid_requires_reason(conn):
    r = validate(conn, "E1001", "UNPAID", "2026-11-03", "2026-11-05", comment="   ")
    assert codes(r) == {"REASON_REQUIRED"}


def test_unpaid_notice_too_short(conn):
    r = validate(conn, "E1001", "UNPAID", "2026-10-30", "2026-10-31", comment="Moving house")
    assert codes(r) == {"NOTICE_PERIOD"}


def test_validation_does_not_write(conn):
    before = conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0]
    validate(conn, "E1001", "ANNUAL", "2026-11-02", "2026-11-04")
    assert conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] == before
