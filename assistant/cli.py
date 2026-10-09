"""Georgian command-line HR assistant. Needs OPENAI_API_KEY in .env and company_hr.db (python -m hr_mcp.init_db).

Run: python -m assistant
"""

import asyncio
import sys

import openai
from mcp import Client
from rich.console import Console

from assistant import ui
from assistant.agent import Agent, AgentReply
from assistant.hr_session import HRSession, login, server_parameters
from hr_mcp.config import DB_PATH, app_today
from rag.answering import Answerer
from rag.retriever import build_retriever

EXIT_COMMANDS = {"/exit", "exit", "quit", "გასვლა"}
MAX_LOGIN_ATTEMPTS = 3

console = Console()


def read(prompt: str) -> str:
    # Read on the main thread: Ctrl+C then interrupts immediately. Nothing is lost while the event loop
    # waits here, because the MCP server only sends data in reply to our requests.
    return console.input(prompt).strip()


def load_answerer() -> Answerer | None:
    try:
        with console.status("იტვირთება პოლიტიკის დოკუმენტები..."):
            return Answerer(build_retriever())
    except RuntimeError:
        console.print(ui.error_panel(
            "OpenAI API გასაღები ვერ მოიძებნა.",
            "დაამატეთ ხაზი OPENAI_API_KEY=... პროექტის .env ფაილში (იხ. README) და გაუშვით თავიდან.",
        ))
    except openai.OpenAIError as exc:
        console.print(ui.error_panel(*ui.describe_error(exc)))
    return None


async def log_in(client: Client) -> HRSession | None:
    for _ in range(MAX_LOGIN_ATTEMPTS):
        try:
            employee_id = read("[bold]თანამშრომლის ID[/] [dim](მაგ. E1001)[/] › ")
        except (KeyboardInterrupt, EOFError):
            return None
        session = await login(client, employee_id)
        if session is not None:
            return session
        console.print("[red]ასეთი თანამშრომელი ვერ მოიძებნა.[/]")
    console.print("[red]მცდელობები ამოიწურა.[/]")
    return None


def show(reply: AgentReply, session: HRSession) -> None:
    console.print(ui.reply_panel(reply.text))
    confirmed = reply.last_result("confirm_leave_request")
    proposed = reply.last_result("propose_leave_request")
    if confirmed and confirmed.get("ok"):
        console.print(ui.request_panel(confirmed))
    elif proposed and proposed.get("ok") and session.pending is not None:
        console.print(ui.proposal_panel(proposed))


async def converse(agent: Agent) -> None:
    while True:
        hint = " [yellow](კი / არა)[/]" if agent.session.pending else ""
        try:
            text = read(f"\n[bold green]თქვენ[/]{hint} › ")
        except (KeyboardInterrupt, EOFError):
            return
        command = text.lower()
        if not text:
            continue
        if command in EXIT_COMMANDS:
            return
        if command == "/help":
            console.print(ui.help_panel())
            continue
        if command == "/new":
            agent.reset()
            console.print("[dim]ახალი საუბარი დაიწყო.[/]")
            continue

        try:
            with console.status("ვფიქრობ..."):
                reply = await agent.reply(text)
        except KeyboardInterrupt:
            console.print("[dim]შეწყვეტილია. შეგიძლიათ დასვათ სხვა კითხვა.[/]")
            continue
        except Exception as exc:
            console.print(ui.error_panel(*ui.describe_error(exc)))
            continue
        show(reply, agent.session)


async def chat() -> int:
    console.rule("[bold cyan]Northstar Services · HR ასისტენტი[/]")
    if not DB_PATH.exists():
        console.print(ui.error_panel("HR მონაცემთა ბაზა ვერ მოიძებნა.", "გაუშვით: python -m hr_mcp.init_db"))
        return 1
    answerer = load_answerer()
    if answerer is None:
        return 1

    try:
        async with Client(server_parameters()) as client:
            session = await log_in(client)
            if session is None:
                return 1
            agent = Agent(session, answerer)
            console.print(ui.welcome_panel(session.employee_id, app_today()))
            await converse(agent)
    except Exception as exc:
        console.print(ui.error_panel(
            "HR სისტემასთან (MCP სერვერთან) კავშირი ვერ დამყარდა ან გაწყდა.",
            f"{type(exc).__name__}. შეამოწმეთ, რომ გარემო სწორადაა დაყენებული, და გაუშვით თავიდან.",
        ))
        return 1
    console.print("\n[cyan]ნახვამდის![/]")
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    try:
        code = asyncio.run(chat())
    except (KeyboardInterrupt, EOFError):
        console.print("\n[cyan]ნახვამდის![/]")
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
