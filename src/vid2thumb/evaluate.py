"""Evaluation of each stage: WER (transcript), ROUGE and support (summary), prompt checks
and CLIPScore (images, extra ``clip``).
"""
from __future__ import annotations

import re
from collections import Counter

from .textutils import STOPWORDS, content_words, words


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: (substitutions + deletions + insertions) / reference words."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: float, n_ref: float, n_hyp: float) -> float:
    if overlap == 0 or n_ref == 0 or n_hyp == 0:
        return 0.0
    p, r = overlap / n_hyp, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_n(reference: str, summary: str, n: int = 1) -> float:
    ref, hyp = _ngrams(words(reference), n), _ngrams(words(summary), n)
    return _f1(sum((ref & hyp).values()), sum(ref.values()), sum(hyp.values()))


def rouge_l(reference: str, summary: str) -> float:
    ref, hyp = words(reference), words(summary)
    if not ref or not hyp:
        return 0.0
    prev = [0] * (len(hyp) + 1)
    for r in ref:
        cur = [0] * (len(hyp) + 1)
        for j, h in enumerate(hyp, 1):
            cur[j] = prev[j - 1] + 1 if r == h else max(prev[j], cur[j - 1])
        prev = cur
    return _f1(prev[-1], len(ref), len(hyp))


def support(summary: str, source: str) -> dict:
    """Share of summary content words and numbers that also occur in the source.

    A value below 1.0 shows words that the summarizer added. It is a cheap check of
    faithfulness, not a proof.
    """
    src = set(words(source))
    cw = content_words(summary)
    numbers = [w for w in words(summary) if re.fullmatch(r"\d+(?:\.\d+)?", w)]
    unsupported = sorted({w for w in cw + numbers if w not in src})
    total = len(cw) + len(numbers)
    return {"support": 1.0 if total == 0 else 1 - sum(w in unsupported for w in cw + numbers) / total, "unsupported": unsupported}


def prompt_checks(prompt: str, max_chars: int) -> dict:
    banned = [w for w in ("logo", "watermark") if w in prompt.lower() and f"no {w}" not in prompt.lower()]
    return {"chars": len(prompt), "within_limit": len(prompt) <= max_chars, "banned_terms": banned}


def summary_metrics(summary: str, transcript: str, reference: str | None = None) -> dict:
    out = {"words": len(words(summary)), **support(summary, transcript)}
    if reference:
        out.update({"rouge1": rouge_n(reference, summary, 1), "rouge2": rouge_n(reference, summary, 2), "rougeL": rouge_l(reference, summary)})
    return out


def content_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words, a cheap text-text check used in reports."""
    sa = {w for w in words(a) if w not in STOPWORDS}
    sb = {w for w in words(b) if w not in STOPWORDS}
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0
