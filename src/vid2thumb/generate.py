"""Image generators behind one interface, and the thumbnail crop.

Each image is saved to disk as PNG with its prompt and metadata. No code keeps only a
temporary URL.
"""
from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from .llm import openai_client, with_retries

THUMB_SIZE = (1280, 720)


@dataclass
class GeneratedImage:
    image: Image.Image
    prompt: str
    generator: str
    meta: dict = field(default_factory=dict)


class ImageGenerator(Protocol):
    name: str

    def generate(self, prompt: str, seed: int = 0) -> GeneratedImage: ...


class PlaceholderGenerator:
    """Offline generator: a deterministic abstract image from the prompt hash and the seed.

    It proves that the pipeline is connected. It does not depict the prompt.
    """

    name = "placeholder"

    def __init__(self, size: tuple[int, int] = THUMB_SIZE) -> None:
        self.size = size

    def generate(self, prompt: str, seed: int = 0) -> GeneratedImage:
        digest = hashlib.sha256(f"{seed}:{prompt}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        w, h = self.size
        c1, c2 = rng.integers(30, 225, size=3), rng.integers(30, 225, size=3)
        t = np.linspace(0, 1, w)[None, :, None]
        grad = (c1 * (1 - t) + c2 * t).repeat(h, axis=0).astype(np.uint8)
        img = Image.fromarray(grad)
        draw = ImageDraw.Draw(img)
        for _ in range(6):
            x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(h // 10, h // 3))
            colour = tuple(int(v) for v in rng.integers(0, 255, size=3))
            draw.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=8)
        return GeneratedImage(img, prompt, self.name, {"seed": seed})


class OpenAIImageGenerator:
    """``client.images.generate`` with base64 output (extra ``openai``)."""

    def __init__(self, model: str = "dall-e-3", api_key: str = "", client=None, size: str = "1792x1024", attempts: int = 3):
        self.client = client or openai_client(api_key, timeout_s=120.0)
        self.model = model
        self.size = size
        self.attempts = attempts
        self.name = f"openai:{model}"

    def generate(self, prompt: str, seed: int = 0) -> GeneratedImage:
        def call():
            return self.client.images.generate(model=self.model, prompt=prompt, size=self.size, n=1, response_format="b64_json")

        result = with_retries(call, attempts=self.attempts)
        item = result.data[0]
        image = Image.open(io.BytesIO(base64.b64decode(item.b64_json))).convert("RGB")
        meta = {"size": self.size, "revised_prompt": getattr(item, "revised_prompt", None)}
        return GeneratedImage(image, prompt, self.name, meta)


def cover(image: Image.Image, size: tuple[int, int] = THUMB_SIZE) -> Image.Image:
    """Scale and centre-crop ``image`` so that it fills ``size`` exactly (16:9 by default)."""
    w, h = size
    src = image.convert("RGB")
    scale = max(w / src.width, h / src.height)
    resized = src.resize((max(w, round(src.width * scale)), max(h, round(src.height * scale))), Image.LANCZOS)
    left = (resized.width - w) // 2
    top = (resized.height - h) // 2
    return resized.crop((left, top, left + w, top + h))


def build_generator(name: str, model: str = "dall-e-3", api_key: str = "") -> ImageGenerator:
    if name == "placeholder":
        return PlaceholderGenerator()
    if name == "openai":
        return OpenAIImageGenerator(model=model, api_key=api_key)
    raise ValueError(f"unknown generator {name!r}")
