"""Synthetic sample folders for the offline demo and the tests.

A sample folder holds what the pipeline gets from a video:

* ``audio.wav``: 16 kHz tone bursts (speech stand-in) with long silences between them
* ``frames/``: PNG frames at 1 fps with three scenes and one blurred frame
* ``transcript.srt``: auto-caption-like subtitles with seeded word errors
* ``reference.json``: the exact transcript, a reference summary and the scene starts

Nothing here is real speech or real video. The sample tests the plumbing and the
metrics, not the quality of real models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .media import write_wav
from .transcript import Segment, Transcript

TOPICS = {
    "bread": {
        "lines": [
            "Today we bake a simple sourdough bread at home.",
            "The starter must be active and full of bubbles before you mix the dough.",
            "Mix flour, water, salt and the starter, then rest the dough for one hour.",
            "Fold the dough four times during the first two hours of the rise.",
            "Shape the loaf, put it in a basket and leave it in the fridge overnight.",
            "Bake the loaf in a hot covered pot for twenty minutes, then uncover it.",
            "The crust gets dark and crisp, and the crumb stays soft and open.",
            "Let the bread cool for one hour before you cut it.",
        ],
        "summary": "A home sourdough bread recipe: mix flour, water, salt and an active starter, fold the dough, "
        "rest it overnight in the fridge and bake it in a hot covered pot for a crisp crust.",
    },
    "bicycle": {
        "lines": [
            "In this video we repair a flat tyre on a road bicycle.",
            "First release the brake and remove the wheel from the frame.",
            "Use two tyre levers to lift one side of the tyre off the rim.",
            "Pull out the inner tube and find the hole with water and soap.",
            "Check the inside of the tyre for glass or a sharp stone.",
            "Fit a new tube, push the tyre back on the rim with your thumbs.",
            "Pump the tyre to the pressure that is printed on the sidewall.",
            "Put the wheel back in the frame and close the brake again.",
        ],
        "summary": "How to repair a flat road bicycle tyre: remove the wheel, lift the tyre with levers, find the hole, "
        "check the tyre for glass, fit a new tube and pump it to the printed pressure.",
    },
}
SCENE_COLOURS = [(40, 90, 200), (60, 170, 80), (220, 140, 40)]


def _noisy(text: str, rng: np.random.Generator, error_rate: float) -> str:
    out = []
    for w in text.split():
        r = rng.random()
        if r < error_rate / 2:
            continue  # deletion
        if r < error_rate:
            out.append(w[::-1].lower())  # substitution
        else:
            out.append(w)
    return " ".join(out)


def _frame(colour, shape: int, size=(320, 180), blur: bool = False, rng=None) -> Image.Image:
    img = Image.new("RGB", size, colour)
    d = ImageDraw.Draw(img)
    w, h = size
    for k in range(4):
        x = int(rng.integers(20, w - 60)) if rng is not None else 40 + 60 * k
        y = int(rng.integers(20, h - 60)) if rng is not None else 40
        box = [x, y, x + 50, y + 50]
        if shape == 0:
            d.ellipse(box, fill=(250, 250, 250), outline=(0, 0, 0), width=3)
        elif shape == 1:
            d.rectangle(box, fill=(20, 20, 20), outline=(255, 255, 255), width=3)
        else:
            d.polygon([(x, y + 50), (x + 25, y), (x + 50, y + 50)], fill=(255, 255, 0), outline=(0, 0, 0))
    for i in range(0, w, 16):  # fine texture, so sharpness differs from the blurred copy
        d.line([(i, h - 12), (i + 8, h - 2)], fill=(0, 0, 0), width=1)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(4))
    return img


def make_sample(out_dir: str | Path, topic: str = "bread", seed: int = 42, error_rate: float = 0.08) -> Path:
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}, use one of {sorted(TOPICS)}")
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    lines = TOPICS[topic]["lines"]

    rate, t, chunks, segments = 16000, 0.0, [], []
    for i, line in enumerate(lines):
        silence = 1.5 if i % 2 else 0.4  # some silences are longer than 1 s
        chunks.append(np.zeros(int(silence * rate), dtype=np.float32))
        t += silence
        length = 0.25 * len(line.split())
        tt = np.arange(int(length * rate)) / rate
        chunks.append((0.3 * np.sin(2 * np.pi * (180 + 20 * i) * tt)).astype(np.float32))
        segments.append(Segment(round(t, 3), round(t + length, 3), line))
        t += length
    chunks.append(np.zeros(int(0.5 * rate), dtype=np.float32))
    write_wav(out / "audio.wav", np.concatenate(chunks), rate)

    noisy = Transcript([Segment(s.start, s.end, _noisy(s.text, rng, error_rate)) for s in segments], source="synthetic auto-captions")
    (out / "transcript.srt").write_text(noisy.to_srt(), encoding="utf-8")

    n_frames = int(np.ceil(t)) + 1
    starts = [0, n_frames // 3, 2 * n_frames // 3]
    for i in range(n_frames):
        scene = sum(i >= s for s in starts) - 1
        frame_rng = np.random.default_rng(seed * 1000 + scene)
        img = _frame(SCENE_COLOURS[scene], scene, blur=(i == starts[0] + 1), rng=frame_rng)
        img.save(out / "frames" / f"frame_{i + 1:05d}.png")
    reference = {
        "topic": topic,
        "transcript": " ".join(lines),
        "summary": TOPICS[topic]["summary"],
        "scene_starts": starts,
        "blurred_frame": starts[0] + 1,
    }
    (out / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return out
