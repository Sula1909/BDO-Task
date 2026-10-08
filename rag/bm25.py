"""Keyword search with BM25 over the Georgian text.

Georgian adds case endings and postpositions to the end of a word
(შვებულება, შვებულების, შვებულებას, შვებულებაზე; ბებია, ბებიის). Exact-word
matching would treat these as different words, so the tokenizer strips one
common ending and keeps the first STEM_LENGTH letters. It is a crude stemmer,
but it makes most forms of a word match without a morphology library.
"""

import math
import re
from collections import Counter

TOKEN = re.compile(r"\d+(?:\.\d+)*|[^\W\d_]+")
STEM_LENGTH = 6
MIN_STEM = 3
SUFFIXES = sorted(
    [
        "ებისთვის", "ისთვის", "თვის", "იდან", "დან", "ამდე", "მდე",
        "ებში", "ებზე", "ებით", "ების", "ებს", "ები",
        "ში", "ზე", "თან", "ით", "ად", "ის", "მა", "ს", "ი", "ა", "ო", "ე", "მ",
    ],
    key=len,
    reverse=True,
)


def stem(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            word = word[: -len(suffix)]
            break
    return word[:STEM_LENGTH]


def tokenize(text: str) -> list[str]:
    return [raw if raw[0].isdigit() else stem(raw) for raw in TOKEN.findall(text.lower())]


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_length = sum(self.lengths) / len(documents) if documents else 0.0

        n = len(documents)
        doc_freq = Counter(term for doc in documents for term in set(doc))
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: list[str]) -> list[float]:
        terms = [t for t in set(query) if t in self.idf]
        result = []
        for counts, length in zip(self.term_counts, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            result.append(score)
        return result
