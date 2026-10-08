"""Georgian -> English translation used as a bridge for semantic search.

OpenAI embedding models barely distinguish Georgian texts (a Georgian question
scores almost the same against the right article and unrelated ones), but they
work well in English. So each chunk is translated once (cached on disk) and
embedded in English, and the employee's question is translated before the
vector search. The Georgian original is still what BM25 searches and what the
model quotes in answers.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from rag.llm import CACHE_DIR, CHAT_MODEL, make_client

PARALLEL_REQUESTS = 8

DOCUMENT_PROMPT = (
    "Translate this excerpt of a Georgian company policy into English. "
    "Keep every number, date, amount, code (e.g. HR-POL-02, ANNUAL) and article number exactly. "
    "Keep the line structure; table rows use ' | ' between cells. Output only the translation."
)
QUERY_PROMPT = (
    "Translate this employee question from Georgian into English. "
    "If it is already in English, return it unchanged. Output only the translation."
)


class Translator:
    def __init__(self, model: str = CHAT_MODEL, cache_dir: Path = CACHE_DIR, client: OpenAI | None = None):
        self.model = model
        self.cache_path = cache_dir / f"translations-{model}.json"
        self.client = client or make_client()
        self._cache: dict[str, str] = (
            json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        )

    def translate_documents(self, texts: list[str]) -> list[str]:
        keys = [self._key(text) for text in texts]
        missing = list(dict.fromkeys(text for text, key in zip(texts, keys) if key not in self._cache))
        if missing:
            with ThreadPoolExecutor(PARALLEL_REQUESTS) as pool:
                translations = list(pool.map(lambda t: self._translate(DOCUMENT_PROMPT, t), missing))
            for text, translation in zip(missing, translations):
                self._cache[self._key(text)] = translation
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return [self._cache[key] for key in keys]

    def translate_query(self, text: str) -> str:
        return self._translate(QUERY_PROMPT, text)

    def _translate(self, instructions: str, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": text}],
        )
        return response.choices[0].message.content.strip()

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode("utf-8")).hexdigest()
