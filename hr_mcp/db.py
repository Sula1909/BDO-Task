import sqlite3
from pathlib import Path

from hr_mcp.config import DB_PATH


def get_connection(db_path: Path | str | None = None, must_exist: bool = True) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Database not found at {path}. Run: python -m hr_mcp.init_db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # SQLite ignores FOREIGN KEY constraints unless enabled per connection.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
