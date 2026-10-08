"""Shared OpenAI settings. The API key is read from OPENAI_API_KEY (environment or the project's .env file)."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from hr_mcp.config import BASE_DIR

load_dotenv(BASE_DIR / ".env")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CACHE_DIR = BASE_DIR / "cache"


def make_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to the .env file in the project root.")
    return OpenAI()
