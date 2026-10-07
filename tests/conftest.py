import pytest

from vid2thumb.synthetic import make_sample


@pytest.fixture(scope="session")
def sample(tmp_path_factory):
    return make_sample(tmp_path_factory.mktemp("samples") / "bread", topic="bread", seed=7)


class FakeChoice:
    def __init__(self, text, finish_reason="stop"):
        self.message = type("M", (), {"content": text})()
        self.finish_reason = finish_reason


class FakeOpenAI:
    """Records calls like the current OpenAI SDK client. No network."""

    def __init__(self, replies=None, finish_reason="stop", image_b64=None):
        self.calls = []
        self.replies = list(replies or ["a summary."])
        self.finish_reason = finish_reason
        self.image_b64 = image_b64
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(("chat", kwargs))
                text = outer.replies[min(len(outer.calls), len(outer.replies)) - 1]
                return type("R", (), {"choices": [FakeChoice(text, outer.finish_reason)]})()

        class _Images:
            def generate(self, **kwargs):
                outer.calls.append(("image", kwargs))
                item = type("D", (), {"b64_json": outer.image_b64, "revised_prompt": "revised"})()
                return type("R", (), {"data": [item]})()

        self.chat = type("C", (), {"completions": _Completions()})()
        self.images = _Images()


@pytest.fixture
def fake_openai():
    return FakeOpenAI
