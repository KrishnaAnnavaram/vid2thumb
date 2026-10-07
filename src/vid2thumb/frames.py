"""The best-frame branch: scene detection, frame quality and ranking of REAL video frames.

* Scene cuts come from the distance between colour histograms of neighbouring frames.
* Each scene gives its sharpest, well-exposed frame as a keyframe candidate.
* Candidates are ranked by quality, or by quality plus image-text relevance to the
  summary (CLIP, extra ``clip``).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .media import Frame


def histogram(pixels: np.ndarray, bins: int = 8) -> np.ndarray:
    q = (pixels.reshape(-1, 3).astype(np.int32) * bins) // 256
    idx = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]
    h = np.bincount(idx, minlength=bins**3).astype(float)
    return h / h.sum()


def scene_boundaries(frames: list[Frame], threshold: float = 0.35) -> list[int]:
    """Indices where a new scene starts (always includes 0). Distance = half the L1 distance of histograms."""
    if not frames:
        return []
    hists = [histogram(f.pixels) for f in frames]
    starts = [0]
    for i in range(1, len(frames)):
        if 0.5 * np.abs(hists[i] - hists[i - 1]).sum() > threshold:
            starts.append(i)
    return starts


def sharpness(pixels: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian of the grey image."""
    g = pixels.astype(float).mean(axis=2)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def exposure(pixels: np.ndarray) -> float:
    """1.0 for a mean brightness of 0.5, 0.0 for a black or white frame."""
    mean = pixels.astype(float).mean() / 255.0
    return float(max(0.0, 1.0 - abs(mean - 0.5) * 2))


def colourfulness(pixels: np.ndarray) -> float:
    """Hasler and Suesstrunk colourfulness, divided by 100."""
    p = pixels.astype(float)
    rg = p[..., 0] - p[..., 1]
    yb = 0.5 * (p[..., 0] + p[..., 1]) - p[..., 2]
    return float((np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 100.0)


@dataclass
class ScoredFrame:
    frame: Frame
    scene: int
    quality: float
    relevance: float | None = None
    score: float = 0.0


def quality_scores(frames: list[Frame]) -> np.ndarray:
    """Weighted mix of rank-normalised sharpness, exposure and colourfulness, in [0, 1]."""
    if not frames:
        return np.array([])
    feats = np.array([[sharpness(f.pixels), exposure(f.pixels), colourfulness(f.pixels)] for f in frames])
    ranks = feats.argsort(axis=0).argsort(axis=0) / max(1, len(frames) - 1)
    ranks[:, 1] = feats[:, 1]  # exposure is already absolute
    return ranks @ np.array([0.5, 0.3, 0.2])


class RelevanceScorer(Protocol):
    name: str

    def score(self, images: list[np.ndarray], text: str) -> np.ndarray: ...


class ClipScorer:
    """CLIPScore = 100 * max(cos(image, text), 0) with a Hugging Face CLIP model (extra ``clip``)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        transformers = importlib.import_module("transformers")
        self.torch = importlib.import_module("torch")
        self.model = transformers.CLIPModel.from_pretrained(model_name).eval()
        self.processor = transformers.CLIPProcessor.from_pretrained(model_name)
        self.name = f"clip:{model_name}"

    def score(self, images: list[np.ndarray], text: str) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.processor(text=[text], images=list(images), return_tensors="pt", padding=True, truncation=True)
            out = self.model(**inputs)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            cos = (img @ txt.T).squeeze(-1).numpy()
        return 100.0 * np.clip(cos, 0, None)


def select_keyframes(frames: list[Frame], threshold: float = 0.35) -> list[ScoredFrame]:
    """One candidate per scene: the frame with the best quality score in that scene."""
    if not frames:
        return []
    starts = scene_boundaries(frames, threshold) + [len(frames)]
    quality = quality_scores(frames)
    out = []
    for scene, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        best = a + int(np.argmax(quality[a:b]))
        out.append(ScoredFrame(frames[best], scene, float(quality[best])))
    return out


def rank_frames(candidates: list[ScoredFrame], summary: str = "", scorer: RelevanceScorer | None = None,
                weight: float = 0.7) -> list[ScoredFrame]:
    """Sort candidates. With a scorer: ``weight`` * normalised relevance + (1 - weight) * quality."""
    if scorer is not None and candidates and summary:
        rel = scorer.score([c.frame.pixels for c in candidates], summary)
        span = float(rel.max() - rel.min()) or 1.0
        for c, r in zip(candidates, rel):
            c.relevance = float(r)
            c.score = weight * (float(r) - float(rel.min())) / span + (1 - weight) * c.quality
    else:
        for c in candidates:
            c.score = c.quality
    return sorted(candidates, key=lambda c: (-c.score, c.frame.index))
