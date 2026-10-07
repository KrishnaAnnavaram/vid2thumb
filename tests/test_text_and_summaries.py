"""Chunking, summarizers, prompts and providers (reference problems 2, 3, 4, 7 and 8)."""
import base64
import io

import pytest
from PIL import Image

from vid2thumb.generate import OpenAIImageGenerator, PlaceholderGenerator, cover
from vid2thumb.llm import OpenAIChat, ProviderError, ScriptedLLM, with_retries
from vid2thumb.prompts import PROMPT_LIMITS, build_visual_prompt, limit_for
from vid2thumb.summarize import SUMMARY_INSTRUCTION, ExtractiveSummarizer, LLMSummarizer, chunk_sentences
from vid2thumb.textutils import approx_tokens, sentences, truncate_words, words

LONG = " ".join(f"Sentence number {i} talks about topic {i % 7} and the oven." for i in range(400))


def test_chunks_respect_the_budget_and_cover_all_sentences():
    chunks = chunk_sentences(LONG, max_tokens=120, overlap=0)
    assert len(chunks) > 5
    assert all(approx_tokens(c) <= 120 for c in chunks)
    assert " ".join(chunks) == " ".join(sentences(LONG))


def test_one_very_long_sentence_is_split():
    text = " ".join(["word"] * 1000) + "."
    chunks = chunk_sentences(text, max_tokens=100)
    assert all(approx_tokens(c) <= 100 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 1000


def test_map_reduce_sees_the_whole_transcript_not_the_first_tokens():
    """Problems 2 and 3: no input is cut. Each call stays in the budget and has the same instruction."""
    llm = ScriptedLLM(["Partial summary about the oven and topics."])
    summarizer = LLMSummarizer(llm, input_tokens=500)
    summarizer.summarize(LONG, max_words=50)
    map_calls = [c for c in llm.calls if "Sentence number" in c[1]]
    covered = " ".join(c[1] for c in map_calls)
    assert "Sentence number 0 " in covered and "Sentence number 399 " in covered
    assert all(approx_tokens(c[1]) <= 520 for c in llm.calls)
    assert all(c[0].startswith(SUMMARY_INSTRUCTION.split("{")[0]) for c in llm.calls)


def test_summary_respects_the_word_budget():
    llm = ScriptedLLM([" ".join(["long"] * 300) + "."])
    out = LLMSummarizer(llm).summarize("A short text. Another sentence.", max_words=40)
    assert len(out.split()) <= 41


def test_extractive_summary_is_deterministic_and_faithful(sample):
    text = (sample / "reference.json").read_text(encoding="utf-8")
    s = ExtractiveSummarizer()
    a, b = s.summarize(text, 40), s.summarize(text, 40)
    assert a == b and 0 < len(words(a)) <= 41


def test_openai_chat_uses_current_sdk_shape_and_temperature_zero(fake_openai):
    client = fake_openai(["Short summary."])
    chat = OpenAIChat("gpt-4o-mini", client=client)
    assert chat.complete("sys", "user", 50) == "Short summary."
    kind, kwargs = client.calls[0]
    assert kind == "chat" and kwargs["temperature"] == 0 and kwargs["max_tokens"] == 50
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}


def test_truncated_reply_is_an_error_not_a_summary(fake_openai):
    """Problem 4: a reply cut at max_tokens is not accepted."""
    chat = OpenAIChat("m", client=fake_openai(["cut mid"], finish_reason="length"), attempts=1)
    with pytest.raises(ProviderError):
        chat.complete("s", "u", 10)


def test_missing_key_gives_clear_error():
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIChat("m", api_key="")


def test_retries_with_backoff():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("slow")
        return "ok"

    assert with_retries(flaky, attempts=3, backoff_s=0.5, sleep=sleeps.append) == "ok"
    assert sleeps == [0.5, 1.0]
    with pytest.raises(ProviderError):
        with_retries(lambda: 1 / 0, attempts=2, sleep=lambda s: None)


@pytest.mark.parametrize("model", sorted(PROMPT_LIMITS))
def test_visual_prompt_is_inside_each_limit(model):
    """Problem 7: the prompt never exceeds the limit of the generator."""
    limit = limit_for(model)
    summary = "The baker shapes the dough. " * 500
    prompt = build_visual_prompt(summary, max_chars=limit)
    assert len(prompt) <= limit and "no text" in prompt


def test_llm_visual_prompt_is_cut_to_the_limit():
    llm = ScriptedLLM(["A very long scene " * 200])
    prompt = build_visual_prompt("Bread is baked.", max_chars=300, llm=llm)
    assert len(prompt) <= 300
    assert "do not include text" in llm.calls[0][0].lower() and "no text" in prompt


def test_empty_summary_is_rejected():
    with pytest.raises(ValueError):
        build_visual_prompt("  ")


def test_truncate_words_keeps_word_boundary():
    assert truncate_words("alpha beta gamma", 10) == "alpha beta"


def test_placeholder_generator_is_deterministic():
    g = PlaceholderGenerator(size=(64, 36))
    a, b = g.generate("bread", seed=1), g.generate("bread", seed=1)
    assert a.image.tobytes() == b.image.tobytes()
    assert g.generate("bread", seed=2).image.tobytes() != a.image.tobytes()


def test_openai_image_generator_decodes_and_keeps_metadata(fake_openai):
    """Problems 8 and 9: current images API with base64 output, no expiring URL."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 18), (10, 200, 30)).save(buf, format="PNG")
    client = fake_openai(image_b64=base64.b64encode(buf.getvalue()).decode())
    gen = OpenAIImageGenerator(model="dall-e-3", client=client)
    out = gen.generate("a bread loaf")
    assert out.image.size == (32, 18) and out.meta["revised_prompt"] == "revised"
    kind, kwargs = client.calls[0]
    assert kind == "image" and kwargs["response_format"] == "b64_json" and kwargs["n"] == 1


def test_cover_fills_16_by_9():
    img = cover(Image.new("RGB", (300, 300)), (128, 72))
    assert img.size == (128, 72)
