"""Georgian command-line HR assistant. Needs OPENAI_API_KEY in .env and company_hr.db (python -m hr_mcp.init_db).

Run: python -m assistant
"""

import asyncio
import sys

from mcp import Client

from assistant.agent import Agent
from assistant.hr_session import HRSession, login, server_parameters
from rag.answering import Answerer
from rag.retriever import build_retriever

EXIT_WORDS = {"exit", "quit", "გასვლა"}
MAX_LOGIN_ATTEMPTS = 3


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip()


async def log_in(client: Client) -> HRSession | None:
    for _ in range(MAX_LOGIN_ATTEMPTS):
        session = await login(client, await ask("თანამშრომლის ID (მაგ. E1001): "))
        if session is not None:
            return session
        print("ასეთი თანამშრომელი ვერ მოიძებნა.")
    return None


async def chat() -> None:
    print("Northstar Services – HR ასისტენტი")
    print("იტვირთება პოლიტიკის დოკუმენტები...")
    answerer = Answerer(build_retriever())

    async with Client(server_parameters()) as client:
        session = await log_in(client)
        if session is None:
            return
        agent = Agent(session, answerer)
        print(f"\nგამარჯობა, {session.employee_id}! დასვით კითხვა. გასასვლელად აკრიფეთ „გასვლა“.")
        while (text := await ask("\nთქვენ: ")) and text.lower() not in EXIT_WORDS:
            print(f"\nასისტენტი: {await agent.reply(text)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    try:
        asyncio.run(chat())
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
