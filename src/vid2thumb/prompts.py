"""Visual prompt builder: a summary becomes a short scene description inside the prompt
limit of the image generator. The summary is never sent as the prompt without change.
"""
from __future__ import annotations

from collections import Counter

from .llm import ChatLLM
from .textutils import content_words, sentences, truncate_words

# Prompt limits in characters (OpenAI image models) and a safe default for local models.
PROMPT_LIMITS = {"dall-e-2": 1000, "dall-e-3": 4000, "gpt-image-1": 32000, "placeholder": 400, "default": 400}
STYLE = "clean, high-contrast video thumbnail, one clear subject, no text, no logos, no real persons"
VISUAL_INSTRUCTION = (
    "Write one image prompt of at most {max_chars} characters for a video thumbnail. Describe one concrete scene "
    "with a subject, a setting and a mood that match the summary. Do not include text, logos or real persons."
)


def limit_for(model: str) -> int:
    return PROMPT_LIMITS.get(model, PROMPT_LIMITS["default"])


def keywords(text: str, k: int = 6) -> list[str]:
    counts = Counter(content_words(text))
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def build_visual_prompt(summary: str, max_chars: int = 400, llm: ChatLLM | None = None) -> str:
    """Return a prompt of at most ``max_chars`` characters.

    With an LLM, the LLM writes the scene. Without an LLM, the prompt uses the first
    summary sentence and the top keywords. Both paths end with the same style text.
    """
    if not summary.strip():
        raise ValueError("the summary is empty")
    style = f" Style: {STYLE}."
    budget = max(40, max_chars - len(style))
    if llm is not None:
        scene = llm.complete(VISUAL_INSTRUCTION.format(max_chars=budget), f"Summary:\n{summary}", max_tokens=200)
    else:
        first = sentences(summary)[0]
        scene = f"A scene about {', '.join(keywords(summary))}. {first}"
    prompt = truncate_words(scene, budget).rstrip(".") + "." + style
    return prompt if len(prompt) <= max_chars else truncate_words(prompt, max_chars)
