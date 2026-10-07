"""Small, dependency-free text helpers: words, sentences, stopwords and token counts."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just let me more most my myself
    no nor not now of off on once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through to too under until up
    us very was we were what when where which while who whom why will with would you your yours yourself
    yourselves okay ok yeah um uh like really going get got gonna kind sort thing things one two lot well right
    know think say said see way much many make made back even still want new go us""".split()
)

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def words(text: str) -> list[str]:
    """Lower-case word tokens. Apostrophes inside a word stay (``don't``)."""
    return [w.lower().replace("’", "'") for w in _WORD.findall(text)]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 2 and not w.isdigit()]


def sentences(text: str) -> list[str]:
    """Split text into sentences. A text with no end mark is one sentence."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def approx_tokens(text: str) -> int:
    """Approximate model tokens: 4/3 of the word count. Used when no tokenizer is loaded."""
    return int(round(len(words(text)) * 4 / 3))


def truncate_words(text: str, max_chars: int) -> str:
    """Cut ``text`` at a word boundary so that it has at most ``max_chars`` characters."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-")[:max_chars]
