"""Summarizers behind one interface, with the same instruction and the same length budget.

A long transcript is never cut. ``Summarizer.summarize`` splits it into chunks that fit
the input budget of the model, summarizes each chunk (map), joins the partial summaries
and summarizes them again (reduce) until the text fits.
"""
from __future__ import annotations

import importlib
import math
from collections import Counter

from .llm import ChatLLM
from .textutils import approx_tokens, content_words, sentences, words

SUMMARY_INSTRUCTION = (
    "Summarize the transcript of a video in plain English. Keep only facts that the transcript states. "
    "Do not add new names, numbers or events. Write at most {max_words} words."
)


def chunk_sentences(text: str, max_tokens: int, count=approx_tokens, overlap: int = 1) -> list[str]:
    """Group sentences into chunks of at most ``max_tokens``. Each chunk repeats ``overlap``
    sentences of the previous chunk. A sentence longer than the budget is split by words."""
    units: list[str] = []
    for s in sentences(text):
        if count(s) <= max_tokens:
            units.append(s)
            continue
        ws = s.split()
        step = max(1, int(max_tokens * 3 / 4))
        units.extend(" ".join(ws[i : i + step]) for i in range(0, len(ws), step))
    chunks: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and count(" ".join(current + [unit])) > max_tokens:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            if current and count(" ".join(current + [unit])) > max_tokens:
                current = []
        current.append(unit)
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


class Summarizer:
    name = "base"
    input_tokens = 900  # input budget of one model call

    def count(self, text: str) -> int:
        return approx_tokens(text)

    def _summarize_once(self, text: str, max_words: int) -> str:
        raise NotImplementedError

    def summarize(self, text: str, max_words: int = 60, _depth: int = 0) -> str:
        text = " ".join(text.split())
        if not text:
            return ""
        if self.count(text) <= self.input_tokens or _depth >= 4:
            return _limit_words(self._summarize_once(text, max_words), max_words)
        chunks = chunk_sentences(text, self.input_tokens, count=self.count)
        per_chunk = max(20, math.ceil(2 * max_words / max(1, len(chunks))) + 10)
        partial = " ".join(self._summarize_once(c, per_chunk) for c in chunks)
        return self.summarize(partial, max_words, _depth + 1)


def _limit_words(text: str, max_words: int) -> str:
    ws = text.split()
    if len(ws) <= max_words:
        return text.strip()
    cut = " ".join(ws[:max_words])
    end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    return cut[: end + 1] if end > len(cut) // 2 else cut + "."


class ExtractiveSummarizer(Summarizer):
    """Offline baseline: picks the sentences with the highest mean content-word frequency."""

    name = "extractive"
    input_tokens = 10**9  # works on any length, no chunking needed

    def _summarize_once(self, text: str, max_words: int) -> str:
        sents = sentences(text)
        if not sents:
            return ""
        freq = Counter(content_words(text))
        top = max(freq.values()) if freq else 1

        def score(i: int, s: str) -> float:
            cw = content_words(s)
            base = sum(freq[w] / top for w in cw) / (len(cw) + 2) if cw else 0.0
            return base + (0.05 if i == 0 else 0.0)

        ranked = sorted(range(len(sents)), key=lambda i: (-score(i, sents[i]), i))
        chosen, used = [], 0
        for i in ranked:
            n = len(words(sents[i]))
            if used + n > max_words and chosen:
                continue
            chosen.append(i)
            used += n
            if used >= max_words * 0.8:
                break
        return " ".join(sents[i] for i in sorted(chosen))


class LLMSummarizer(Summarizer):
    name = "llm"

    def __init__(self, llm: ChatLLM, input_tokens: int = 6000) -> None:
        self.llm = llm
        self.input_tokens = input_tokens
        self.name = f"llm:{llm.name}"

    def _summarize_once(self, text: str, max_words: int) -> str:
        system = SUMMARY_INSTRUCTION.format(max_words=max_words)
        return self.llm.complete(system, f"Transcript:\n{text}", max_tokens=int(max_words * 2) + 20)


class Seq2SeqSummarizer(Summarizer):
    """BART (or another seq2seq model) with the real tokenizer for the chunk budget (extra ``summarize``)."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn", input_tokens: int = 900) -> None:
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        limit = getattr(self.tokenizer, "model_max_length", 1024) or 1024
        self.input_tokens = min(input_tokens, limit - 24)
        self.name = f"seq2seq:{model_name}"

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _summarize_once(self, text: str, max_words: int) -> str:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.input_tokens + 24)
        max_new = int(max_words * 1.5) + 10
        out = self.model.generate(**enc, num_beams=4, do_sample=False, max_new_tokens=max_new, min_new_tokens=min(10, max_new))
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def build_summarizer(name: str, llm: ChatLLM | None = None) -> Summarizer:
    if name == "extractive":
        return ExtractiveSummarizer()
    if name == "bart":
        return Seq2SeqSummarizer()
    if name == "llm":
        if llm is None:
            raise ValueError("the llm summarizer needs an LLM adapter")
        return LLMSummarizer(llm)
    raise ValueError(f"unknown summarizer {name!r}")
