"""Walk through the MCP server the way a real client does: launch it as a subprocess and call tools over stdio.

Uses a temporary copy of the database, so company_hr.db is not changed.

Usage: python scripts/demo_client.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcp import Client, StdioServerParameters

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from hr_mcp.init_db import build_database  # noqa: E402


def show(title: str, data) -> None:
    print(f"\n=== {title}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def call(client: Client, tool: str, **arguments) -> dict:
    result = await client.call_tool(tool, arguments)
    return json.loads(result.content[0].text)


async def main(db_path: Path) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hr_mcp.server"],
        cwd=str(PROJECT_DIR),
        env={"HR_DB_PATH": str(db_path), "PYTHONIOENCODING": "utf-8"},
    )
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        show("1. Tools the server advertises", [{"name": t.name, "description": t.description.split("\n")[0]} for t in tools])
        show("   Input schema of get_leave_balance", next(t for t in tools if t.name == "get_leave_balance").input_schema)

        show("2. get_leave_balance(E1001, ANNUAL)", await call(client, "get_leave_balance", employee_id="E1001", leave_type="ANNUAL"))

        ids = {"employee_id": "E1001", "conversation_id": "demo-conversation"}
        proposal = await call(client, "propose_leave_request", **ids, leave_type="ANNUAL", start_date="2026-11-02", end_date="2026-11-04")
        show("3. propose_leave_request(ANNUAL 2026-11-02..04)", proposal)

        show("4. confirm_leave_request", await call(client, "confirm_leave_request", **ids, proposal_id=proposal["proposal_id"]))
        show("5. confirm_leave_request again (same proposal)", await call(client, "confirm_leave_request", **ids, proposal_id=proposal["proposal_id"]))

        show("6. propose_leave_request(SICK, more than the paid balance)",
             await call(client, "propose_leave_request", **ids, leave_type="SICK", start_date="2026-10-19", end_date="2026-10-30"))

        show("7. hr_decide_leave_request(#28, approved)", await call(client, "hr_decide_leave_request", request_id=28, decision="approved"))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        demo_db = Path(tmp) / "demo_hr.db"
        build_database(demo_db)
        asyncio.run(main(demo_db))
