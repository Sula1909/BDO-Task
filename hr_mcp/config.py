import os
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = Path(os.getenv("HR_DB_PATH", BASE_DIR / "company_hr.db"))

DEFAULT_APP_TODAY = "2026-10-19"


def app_today() -> date:
    """Reference "today" for all policy checks. Read on every call so tests can override APP_TODAY."""
    return date.fromisoformat(os.getenv("APP_TODAY", DEFAULT_APP_TODAY))
