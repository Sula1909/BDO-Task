from datetime import date

import pytest

from hr_mcp.db import get_connection
from hr_mcp.init_db import build_database

TODAY = date(2026, 10, 19)  # Monday; the reference date from the data dictionary


@pytest.fixture(autouse=True)
def fixed_app_today(monkeypatch):
    """Code that calls app_today() itself (the service layer) sees the reference date, whatever the shell sets."""
    monkeypatch.setenv("APP_TODAY", TODAY.isoformat())


@pytest.fixture(scope="session")
def conn(tmp_path_factory):
    """A database built from the CSVs, shared by read-only tests. Separate from company_hr.db."""
    db_path = tmp_path_factory.mktemp("db") / "test_hr.db"
    build_database(db_path)
    connection = get_connection(db_path)
    yield connection
    connection.close()


@pytest.fixture
def db_path(tmp_path):
    """A fresh database file per test, for tests that write."""
    path = tmp_path / "test_hr.db"
    build_database(path)
    return path


@pytest.fixture
def fresh_conn(db_path):
    connection = get_connection(db_path)
    yield connection
    connection.close()
