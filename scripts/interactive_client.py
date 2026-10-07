"""Call the MCP server's tools by hand: pick a tool, type its arguments, see the JSON result.

Launches the server as a subprocess over stdio, like a real MCP client. By default it works on a
temporary copy of the database, so you can experiment freely; --real-db uses company_hr.db.

Usage: python scripts/interactive_client.py [--real-db]
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcp import Client, StdioServerParameters

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from hr_mcp.config import DB_PATH  # noqa: E402
from hr_mcp.init_db import build_database  # noqa: E402


def allowed_types(prop: dict) -> list[str]:
    """JSON Schema types of a parameter, e.g. ["integer"] for `int | None`."""
    options = prop.get("anyOf", [prop])
    return [o["type"] for o in options if o.get("type") and o["type"] != "null"]


def allowed_values(prop: dict) -> list[str] | None:
    for option in prop.get("anyOf", [prop]):
        if "enum" in option:
            return option["enum"]
    return None


def convert(raw: str, prop: dict):
    types = allowed_types(prop)
    if "integer" in types:
        return int(raw)
    if "boolean" in types:
        return raw.strip().lower() in ("true", "yes", "y", "1")
    return raw


async def ask(prompt: str) -> str:
    # input() blocks; running it in a thread keeps the connection to the server responsive.
    return (await asyncio.to_thread(input, prompt)).strip()


async def ask_arguments(tool) -> dict:
    schema = tool.input_schema
    required = set(schema.get("required", []))
    arguments = {}
    for name, prop in schema.get("properties", {}).items():
        hint = "/".join(allowed_values(prop) or allowed_types(prop))
        label = "required" if name in required else "optional, Enter to skip"
        print(f"    {prop.get('description', '')}")
        while True:
            raw = await ask(f"  {name} ({hint}, {label}): ")
            if not raw:
                if name in required:
                    print("    This argument is required.")
                    continue
                break
            try:
                arguments[name] = convert(raw, prop)
                break
            except ValueError:
                print(f"    Expected {hint}.")
    return arguments


async def main(db_path: Path) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hr_mcp.server"],
        cwd=str(PROJECT_DIR),
        env={"HR_DB_PATH": str(db_path), "PYTHONIOENCODING": "utf-8"},
    )
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        print(f"Connected to the MCP server. Database: {db_path}")
        while True:
            print("\nTools:")
            for i, tool in enumerate(tools, 1):
                print(f"  {i}. {tool.name}")
            choice = await ask("Choose a tool number (q to quit): ")
            if choice.lower() == "q":
                return
            if not choice.isdigit() or not 1 <= int(choice) <= len(tools):
                continue

            tool = tools[int(choice) - 1]
            print(f"\n{tool.name}: {tool.description}\n")
            arguments = await ask_arguments(tool)
            print(f"\n-> call {tool.name} {json.dumps(arguments, ensure_ascii=False)}")
            result = await client.call_tool(tool.name, arguments)
            text = result.content[0].text if result.content else ""
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
            print(("<- ERROR\n" if result.is_error else "<- result\n") + text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-db", action="store_true", help="use company_hr.db instead of a temporary copy")
    args = parser.parse_args()

    if args.real_db:
        asyncio.run(main(DB_PATH))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            temp_db = Path(tmp) / "interactive_hr.db"
            build_database(temp_db)
            asyncio.run(main(temp_db))
