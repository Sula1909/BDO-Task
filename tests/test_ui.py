"""Terminal rendering and error messages, checked on a recording rich console (no real terminal needed)."""

from datetime import date

import httpx2 as httpx
import openai
import pytest
from rich.console import Console

from assistant import ui

PROPOSAL = {
    "ok": True,
    "proposal_id": "p1",
    "proposal": {
        "employee_id": "E1001",
        "leave_type": "ANNUAL",
        "leave_type_name": "ყოველწლიური ანაზღაურებადი შვებულება",
        "start_date": "2026-11-02",
        "end_date": "2026-11-04",
        "days": 3,
        "day_unit": "working",
        "comment": None,
    },
    "available_before": 10,
    "available_after": 7,
}
CONFIRMED = {
    "ok": True,
    "already_confirmed": False,
    "request": {"request_id": 28, "leave_type": "ANNUAL", "start_date": "2026-11-02", "end_date": "2026-11-04",
                "days": 3, "status": "pending"},
}
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def render(renderable) -> str:
    console = Console(record=True, width=100, color_system=None)
    console.print(renderable)
    return console.export_text()


def response(status: int) -> httpx.Response:
    return httpx.Response(status, request=REQUEST)


def test_proposal_card_shows_the_server_data_and_how_to_answer():
    text = render(ui.proposal_panel(PROPOSAL))
    assert "დასადასტურებელი მოთხოვნა" in text
    assert "ყოველწლიური ანაზღაურებადი შვებულება (ANNUAL)" in text
    assert "2026-11-02 — 2026-11-04" in text
    assert "3 სამუშაო დღე" in text
    assert "10 → 7" in text
    assert "ჯერ არ არის გაგზავნილი" in text and "კი" in text and "არა" in text


def test_proposal_card_shows_comment_and_calendar_days():
    unpaid = {**PROPOSAL, "proposal": {**PROPOSAL["proposal"], "leave_type": "UNPAID", "day_unit": "calendar",
                                       "comment": "ოჯახური საქმე"}}
    text = render(ui.proposal_panel(unpaid))
    assert "3 კალენდარული დღე" in text
    assert "ოჯახური საქმე" in text


def test_request_card_shows_number_and_georgian_status():
    text = render(ui.request_panel(CONFIRMED))
    assert "მოთხოვნა გაგზავნილია" in text
    assert "#28" in text
    assert "ელოდება დამტკიცებას" in text
    assert "უკვე გაგზავნილია" in render(ui.request_panel({**CONFIRMED, "already_confirmed": True}))


def test_welcome_panel_shows_employee_date_and_commands():
    text = render(ui.welcome_panel("E1001", date(2026, 10, 19)))
    assert "E1001" in text and "2026-10-19" in text
    for command in ui.COMMANDS:
        assert command in text


def test_reply_panel_renders_markdown():
    text = render(ui.reply_panel("**სულ:** 10 დღე\n\n- პირველი\n- მეორე"))
    assert "სულ: 10 დღე" in text and "**" not in text
    assert "• პირველი" in text


@pytest.mark.parametrize(
    "exc, expected",
    [
        (openai.AuthenticationError("bad key", response=response(401), body=None), "API გასაღები"),
        (openai.PermissionDeniedError("denied", response=response(403), body=None), "წვდომა აკრძალულია"),
        (openai.RateLimitError("slow down", response=response(429), body=None), "ლიმიტი"),
        (openai.APITimeoutError(request=REQUEST), "დროულად არ უპასუხა"),
        (openai.APIConnectionError(request=REQUEST), "კავშირი ვერ დამყარდა"),
        (openai.InternalServerError("boom", response=response(500), body=None), "კოდი 500"),
        (ValueError("x"), "მოულოდნელი შეცდომა"),
    ],
)
def test_errors_get_georgian_messages(exc, expected):
    message, hint = ui.describe_error(exc)
    assert expected in message + hint


def test_error_panel_shows_message_and_hint():
    text = render(ui.error_panel("შეცდომის ტექსტი", "რჩევა"))
    assert "შეცდომა" in text and "შეცდომის ტექსტი" in text and "რჩევა" in text
