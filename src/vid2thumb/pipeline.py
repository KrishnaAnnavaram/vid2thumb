"""One run: transcript -> summary -> visual prompt -> generated image, and in parallel
frames -> keyframes -> best real frame. Every output is saved in ``<out>/<run_id>/``.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .asr import Transcriber
from .config import Settings
from .evaluate import prompt_checks, summary_metrics, wer
from .frames import RelevanceScorer, rank_frames, select_keyframes
from .generate import ImageGenerator, cover
from .ingest import Source
from .llm import ChatLLM
from .media import extract_frames, load_frames
from .prompts import build_visual_prompt, limit_for
from .summarize import Summarizer
from .transcript import Transcript


@dataclass
class Components:
    transcriber: Transcriber
    summarizer: Summarizer
    generator: ImageGenerator
    scorer: RelevanceScorer | None = None
    prompt_llm: ChatLLM | None = None


@dataclass
class RunResult:
    run_dir: Path
    transcript: Transcript
    summary: str
    prompt: str
    best_frame: Path | None
    generated: Path
    metrics: dict = field(default_factory=dict)


def _frames_for(source: Source, work: Path) -> list:
    if source.kind == "sample":
        return load_frames(sorted(source.frames_dir.glob("*.png"))) if source.frames_dir else []
    return load_frames(extract_frames(source.path, work / "frames", fps=1.0))


def run(source: Source, settings: Settings, parts: Components, run_id: str | None = None) -> RunResult:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(settings.out_dir) / (run_id or f"{source.title}-{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    # 1. transcript
    transcript = parts.transcriber.transcribe(source)
    (run_dir / "transcript.json").write_text(transcript.to_json(), encoding="utf-8")
    (run_dir / "transcript.srt").write_text(transcript.to_srt(), encoding="utf-8")
    metrics["transcript"] = {"words": len(transcript.text.split()), "segments": len(transcript.segments), "seconds": transcript.duration}

    # 2. summary, with the same instruction and budget for each summarizer
    summary = parts.summarizer.summarize(transcript.text, settings.summary_words)
    (run_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    # 3. visual prompt inside the limit of the generator
    max_chars = limit_for(settings.image_model if settings.generator == "openai" else "placeholder")
    prompt = build_visual_prompt(summary, max_chars=max_chars, llm=parts.prompt_llm)
    (run_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    metrics["prompt"] = prompt_checks(prompt, max_chars)

    # 4. generated image, saved as a file with its metadata
    image = parts.generator.generate(prompt, seed=settings.seed)
    generated = run_dir / "thumbnail_generated.png"
    cover(image.image).save(generated)

    # 5. best real frame
    best_path = None
    with tempfile.TemporaryDirectory() as tmp:
        frames = _frames_for(source, Path(tmp))
        ranked = rank_frames(select_keyframes(frames), summary, parts.scorer)
        if ranked:
            keydir = run_dir / "keyframes"
            keydir.mkdir(exist_ok=True)
            for c in ranked:
                Image.fromarray(c.frame.pixels).save(keydir / f"scene{c.scene:02d}_t{c.frame.time:07.1f}.png")
            best_path = run_dir / "thumbnail_frame.png"
            cover(Image.fromarray(ranked[0].frame.pixels)).save(best_path)
        metrics["frames"] = {
            "frames": len(frames),
            "scenes": len(ranked),
            "best_frame_index": ranked[0].frame.index if ranked else None,
            "candidates": [{"index": c.frame.index, "scene": c.scene, "quality": round(c.quality, 4),
                            "relevance": c.relevance, "score": round(c.score, 4)} for c in ranked],
        }
    if parts.scorer is not None:
        imgs = [np.asarray(Image.open(generated).convert("RGB"))]
        if best_path:
            imgs.append(np.asarray(Image.open(best_path).convert("RGB")))
        scores = parts.scorer.score(imgs, summary)
        metrics["clipscore"] = {"generated": float(scores[0]), "best_frame": float(scores[1]) if best_path else None}

    # 6. reference checks (sample folders and labelled videos)
    ref_path = source.path / "reference.json" if source.kind == "sample" else source.path.with_suffix(".reference.json")
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        metrics["transcript"]["wer"] = wer(ref["transcript"], transcript.text)
        metrics["summary"] = summary_metrics(summary, transcript.text, ref.get("summary"))
        if "scene_starts" in ref:
            metrics["frames"]["reference_scenes"] = len(ref["scene_starts"])
    else:
        metrics["summary"] = summary_metrics(summary, transcript.text)

    record = {
        "vid2thumb_version": __version__,
        "created_utc": stamp,
        "source": {"kind": source.kind, "path": str(source.path), "title": source.title, "licence": source.licence},
        "components": {"transcriber": parts.transcriber.name, "summarizer": parts.summarizer.name,
                       "generator": image.generator, "scorer": getattr(parts.scorer, "name", None),
                       "prompt_writer": getattr(parts.prompt_llm, "name", "rules")},
        "settings": settings.public(),
        "generator_meta": image.meta,
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return RunResult(run_dir, transcript, summary, prompt, best_path, generated, metrics)
