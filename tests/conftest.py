from datetime import date

import pytest

from hr_mcp.db import get_connection
from hr_mcp.init_db import build_database

TODAY = date(2026, 10, 19)  # Monday; the reference date from the data dictionary


@pytest.fixture(scope="session")
def conn(tmp_path_factory):
    """A fresh database built from the CSVs, separate from company_hr.db."""
    db_path = tmp_path_factory.mktemp("db") / "test_hr.db"
    build_database(db_path)
    connection = get_connection(db_path)
    yield connection
    connection.close()
