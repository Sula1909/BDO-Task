import os
from datetime import date, datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = Path(os.getenv("HR_DB_PATH", BASE_DIR / "company_hr.db"))

DEFAULT_APP_TODAY = "2026-10-19"


def app_today() -> date:
    """Reference "today" for all policy checks. Read on every call so tests can override APP_TODAY."""
    return date.fromisoformat(os.getenv("APP_TODAY", DEFAULT_APP_TODAY))


def app_now() -> str:
    """Timestamp for created_at/decided_at: the app's date with the real clock time, ISO 8601 like the CSV."""
    return datetime.combine(app_today(), datetime.now().time()).isoformat(timespec="seconds")
