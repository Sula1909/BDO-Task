from hr_mcp import leave_service as service


def count_requests(conn):
    return conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0]


def error_codes(result):
    return {e["code"] for e in result["errors"]}


def propose(conn, employee_id="E1001", leave_type="ANNUAL", start="2026-11-02", end="2026-11-04",
            comment=None, conversation_id="conv-1"):
    return service.propose_leave_request(conn, employee_id, conversation_id, leave_type, start, end, comment)


# --- Read-only tools --------------------------------------------------------

def test_list_leave_types(conn):
    types = {t["code"]: t for t in service.list_leave_types(conn)["leave_types"]}
    assert len(types) == 6
    assert types["ANNUAL"]["assistant_can_create"] is True
    assert types["STUDY"]["submission_channel"] == "HR_PORTAL_OR_HR"
    assert types["PARENTAL"]["submission_channel"] == "HR"


def test_balance_for_all_types(conn):
    result = service.get_leave_balance(conn, "E1001")
    by_type = {b["leave_type"]: b for b in result["balances"]}
    assert result["year"] == 2026
    assert len(by_type) == 6
    assert by_type["ANNUAL"]["available_days"] == 10
    assert by_type["BEREAVEMENT"]["has_annual_balance"] is False
    assert "note" in by_type["BEREAVEMENT"]


def test_balance_unknown_employee(conn):
    assert error_codes(service.get_leave_balance(conn, "E9999")) == {"UNKNOWN_EMPLOYEE"}


def test_list_requests_by_employee_and_status(conn):
    result = service.list_leave_requests(conn, employee_id="E1001", status="approved")
    assert [r["request_id"] for r in result["requests"]] == [1, 2, 4]


def test_list_requests_by_date_range(conn):
    result = service.list_leave_requests(conn, date_from="2026-11-01", date_to="2026-11-30")
    assert [r["request_id"] for r in result["requests"]] == [5, 15]


def test_list_requests_invalid_status(conn):
    assert error_codes(service.list_leave_requests(conn, status="done")) == {"INVALID_STATUS"}


# --- Propose and confirm ----------------------------------------------------

def test_propose_creates_no_request(fresh_conn):
    before = count_requests(fresh_conn)
    result = propose(fresh_conn)
    assert result["ok"]
    assert result["proposal"]["days"] == 3
    assert (result["available_before"], result["available_after"]) == (10, 7)
    assert count_requests(fresh_conn) == before


def test_propose_refused_type(fresh_conn):
    result = propose(fresh_conn, leave_type="BEREAVEMENT")
    assert not result["ok"]
    assert error_codes(result) == {"NOT_ASSISTANT_SUPPORTED"}


def test_confirm_creates_pending_assistant_request(fresh_conn):
    proposal_id = propose(fresh_conn)["proposal_id"]
    result = service.confirm_leave_request(fresh_conn, "E1001", "conv-1", proposal_id)
    request = result["request"]
    assert result["ok"] and result["already_confirmed"] is False
    assert request["request_id"] == 28
    assert (request["status"], request["created_via"], request["days"]) == ("pending", "assistant", 3)
    assert request["created_at"].startswith("2026-10-19T")


def test_confirm_twice_returns_same_request(fresh_conn):
    proposal_id = propose(fresh_conn)["proposal_id"]
    before = count_requests(fresh_conn)
    first = service.confirm_leave_request(fresh_conn, "E1001", "conv-1", proposal_id)
    second = service.confirm_leave_request(fresh_conn, "E1001", "conv-1", proposal_id)
    assert second["already_confirmed"] is True
    assert second["request"]["request_id"] == first["request"]["request_id"]
    assert count_requests(fresh_conn) == before + 1


def test_confirm_counts_as_pending_in_balance(fresh_conn):
    proposal_id = propose(fresh_conn)["proposal_id"]
    service.confirm_leave_request(fresh_conn, "E1001", "conv-1", proposal_id)
    balance = service.get_leave_balance(fresh_conn, "E1001", "ANNUAL")["balances"][0]
    assert (balance["pending_days"], balance["available_days"]) == (6, 7)


def test_confirm_wrong_employee_or_conversation(fresh_conn):
    proposal_id = propose(fresh_conn)["proposal_id"]
    assert error_codes(service.confirm_leave_request(fresh_conn, "E1002", "conv-1", proposal_id)) == {"PROPOSAL_NOT_FOUND"}
    assert error_codes(service.confirm_leave_request(fresh_conn, "E1001", "conv-2", proposal_id)) == {"PROPOSAL_NOT_FOUND"}


def test_confirm_revalidates(fresh_conn):
    # Two proposals for the same dates are both valid on their own; once the first is confirmed, the second overlaps.
    first = propose(fresh_conn)["proposal_id"]
    second = propose(fresh_conn)["proposal_id"]
    assert service.confirm_leave_request(fresh_conn, "E1001", "conv-1", first)["ok"]
    result = service.confirm_leave_request(fresh_conn, "E1001", "conv-1", second)
    assert "OVERLAP" in error_codes(result)


def test_unpaid_reason_is_stored(fresh_conn):
    proposal_id = propose(fresh_conn, leave_type="UNPAID", start="2026-11-03", end="2026-11-05", comment=" Family matter ")["proposal_id"]
    request = service.confirm_leave_request(fresh_conn, "E1001", "conv-1", proposal_id)["request"]
    assert request["comment"] == "Family matter"


# --- HR decisions -----------------------------------------------------------

def test_approve_pending(fresh_conn):
    request = service.decide_leave_request(fresh_conn, 5, "approved")["request"]
    assert request["status"] == "approved"
    assert request["decided_at"] is not None


def test_reject_requires_reason(fresh_conn):
    assert error_codes(service.decide_leave_request(fresh_conn, 5, "rejected")) == {"REASON_REQUIRED"}
    assert service.decide_leave_request(fresh_conn, 5, "rejected", "Client deadline")["request"]["status"] == "rejected"


def test_only_pending_can_be_decided(fresh_conn):
    assert error_codes(service.decide_leave_request(fresh_conn, 1, "approved")) == {"INVALID_STATUS"}


def test_unknown_request(fresh_conn):
    assert error_codes(service.decide_leave_request(fresh_conn, 999, "approved")) == {"REQUEST_NOT_FOUND"}


# --- Cancellation (Article 4.8) ---------------------------------------------

def test_cancel_pending_returns_days_to_balance(fresh_conn):
    assert service.cancel_leave_request(fresh_conn, 5, "Plans changed")["request"]["status"] == "cancelled"
    balance = service.get_leave_balance(fresh_conn, "E1001", "ANNUAL")["balances"][0]
    assert (balance["pending_days"], balance["available_days"]) == (0, 13)


def test_cancel_approved_future_leave(fresh_conn):
    service.decide_leave_request(fresh_conn, 5, "approved")  # starts 2026-11-09, well after the deadline check
    assert service.cancel_leave_request(fresh_conn, 5)["ok"]


def test_cancel_approved_leave_too_late_needs_manager(fresh_conn):
    # Request #1 (approved) started 2026-02-16, long before today.
    result = service.cancel_leave_request(fresh_conn, 1)
    assert error_codes(result) == {"LATE_CANCELLATION"}
    assert result["errors"][0]["redirect_to"] == "MANAGER"
    assert service.cancel_leave_request(fresh_conn, 1, manager_consent=True)["ok"]


def test_cannot_cancel_twice(fresh_conn):
    assert error_codes(service.cancel_leave_request(fresh_conn, 3)) == {"INVALID_STATUS"}  # #3 is already cancelled
