"""Command line: ``vid2thumb <command>``. Run ``vid2thumb --help`` for the list."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from .asr import TranscriptionError, build_transcriber
from .config import ConfigError, Settings, load_dotenv
from .frames import ClipScorer, rank_frames, select_keyframes
from .generate import build_generator
from .ingest import IngestError, download_url, resolve
from .llm import OpenAIChat, ProviderError
from .media import MediaError, load_frames
from .pipeline import Components, run
from .prompts import build_visual_prompt
from .summarize import build_summarizer
from .synthetic import TOPICS, make_sample


def build_components(settings: Settings) -> Components:
    llm = OpenAIChat(settings.llm_model, settings.openai_api_key) if settings.summarizer == "llm" else None
    return Components(
        transcriber=build_transcriber(settings.transcriber, settings.whisper_model),
        summarizer=build_summarizer(settings.summarizer, llm),
        generator=build_generator(settings.generator, settings.image_model, settings.openai_api_key),
        scorer=ClipScorer() if settings.ranker == "clip" else None,
        prompt_llm=llm,
    )


def _source(args, settings):
    if getattr(args, "url", None):
        return download_url(args.url, Path(settings.out_dir) / "downloads", args.i_own_this, settings.max_video_minutes)
    return resolve(args.input)


def _print_run(result) -> None:
    m = result.metrics
    print(f"run folder: {result.run_dir}")
    print(f"summary ({m['summary']['words']} words): {result.summary}")
    print(f"prompt ({m['prompt']['chars']} chars, within limit: {m['prompt']['within_limit']}): {result.prompt}")
    t = m["transcript"]
    print(f"transcript: {t['words']} words, {t['segments']} segments" + (f", WER {t['wer']:.3f}" if "wer" in t else ""))
    s = m["summary"]
    rouge = f", ROUGE-1 {s['rouge1']:.3f}, ROUGE-L {s['rougeL']:.3f}" if "rouge1" in s else ""
    print(f"summary support {s['support']:.3f}{rouge}")
    f = m["frames"]
    print(f"frames: {f['frames']} frames, {f['scenes']} scenes" + (f" (reference {f['reference_scenes']})" if "reference_scenes" in f else "")
          + f", best frame index {f['best_frame_index']}")
    if "clipscore" in m:
        print(f"CLIPScore: generated {m['clipscore']['generated']:.1f}, best frame {m['clipscore']['best_frame']}")


def cmd_make_sample(args, settings):
    path = make_sample(args.out, topic=args.topic, seed=args.seed, error_rate=args.error_rate)
    print(f"wrote sample folder {path}")
    return 0


def cmd_run(args, settings):
    result = run(_source(args, settings), settings, build_components(settings), run_id=args.run_id)
    _print_run(result)
    return 0


def cmd_transcribe(args, settings):
    transcript = build_transcriber(settings.transcriber, settings.whisper_model).transcribe(_source(args, settings))
    print(transcript.to_srt())
    return 0


def cmd_summarize(args, settings):
    text = Path(args.text).read_text(encoding="utf-8")
    llm = OpenAIChat(settings.llm_model, settings.openai_api_key) if settings.summarizer == "llm" else None
    summary = build_summarizer(settings.summarizer, llm).summarize(text, args.words or settings.summary_words)
    print(summary)
    print("\nprompt: " + build_visual_prompt(summary))
    return 0


def cmd_frames(args, settings):
    source = resolve(args.input)
    if source.kind != "sample":
        raise IngestError("the frames command reads a sample folder; use 'run' for a video file")
    ranked = rank_frames(select_keyframes(load_frames(sorted(source.frames_dir.glob("*.png")))))
    for c in ranked:
        print(f"scene {c.scene}: frame {c.frame.index} at {c.frame.time:.0f} s, quality {c.quality:.3f}")
    return 0


def cmd_evaluate(args, settings):
    lines = [json.loads(ln) for ln in Path(args.manifest).read_text(encoding="utf-8").splitlines() if ln.strip()]
    parts = build_components(settings)
    rows = []
    for i, item in enumerate(lines):
        result = run(resolve(item["input"]), settings, parts, run_id=f"eval-{i:03d}")
        m = result.metrics
        rows.append({
            "input": item["input"],
            "wer": m["transcript"].get("wer"),
            "rouge1": m["summary"].get("rouge1"),
            "rougeL": m["summary"].get("rougeL"),
            "support": m["summary"]["support"],
            "scenes_match": m["frames"].get("reference_scenes") == m["frames"]["scenes"] if "reference_scenes" in m["frames"] else None,
            "prompt_within_limit": m["prompt"]["within_limit"],
        })
    summary = {}
    for key in ("wer", "rouge1", "rougeL", "support", "scenes_match", "prompt_within_limit"):
        values = [float(r[key]) for r in rows if r[key] is not None]
        if values:
            summary[key] = round(float(np.mean(values)), 4)
    report = {"items": len(rows), "mean": summary, "rows": rows}
    out = Path(settings.out_dir) / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"items": len(rows), "mean": summary}, indent=2))
    print(f"wrote {out}")
    return 0


def cmd_demo(args, settings):
    print("vid2thumb offline demo: synthetic samples, sidecar transcripts, extractive summary, placeholder image\n")
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.jsonl"
        items = []
        for topic in sorted(TOPICS):
            path = make_sample(Path(tmp) / topic, topic=topic, seed=settings.seed)
            items.append(json.dumps({"input": str(path)}))
        manifest.write_text("\n".join(items), encoding="utf-8")
        parts = build_components(settings)
        for topic in sorted(TOPICS):
            _print_run(run(resolve(str(Path(tmp) / topic)), settings, parts, run_id=f"demo-{topic}"))
            print()
        args.manifest = str(manifest)
        return cmd_evaluate(args, settings)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vid2thumb", description="Video to transcript, summary, generated thumbnail and best real frame.")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("make-sample", help="write a synthetic sample folder")
    m.add_argument("--out", default="data/sample_bread")
    m.add_argument("--topic", choices=sorted(TOPICS), default="bread")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--error-rate", type=float, default=0.08, help="word error rate of the synthetic captions")
    m.set_defaults(func=cmd_make_sample)

    def input_args(c, url=True):
        group = c.add_mutually_exclusive_group(required=True)
        group.add_argument("--input", help="video file or sample folder")
        if url:
            group.add_argument("--url", help="video URL (extra 'download')")
            c.add_argument("--i-own-this", action="store_true", help="I own the video or have the right to use it")

    r = sub.add_parser("run", help="full pipeline for one video or sample folder")
    input_args(r)
    r.add_argument("--run-id")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("transcribe", help="print the transcript as SRT")
    input_args(t)
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("summarize", help="summarize a text file and print the visual prompt")
    s.add_argument("--text", required=True)
    s.add_argument("--words", type=int)
    s.set_defaults(func=cmd_summarize)

    f = sub.add_parser("frames", help="print the keyframe ranking of a sample folder")
    f.add_argument("--input", required=True)
    f.set_defaults(func=cmd_frames)

    e = sub.add_parser("evaluate", help="run a JSONL manifest and write evaluation.json")
    e.add_argument("--manifest", required=True)
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("demo", help="offline demo on two synthetic samples")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_dotenv()
        return args.func(args, Settings.from_env())
    except (ConfigError, IngestError, TranscriptionError, MediaError, ProviderError, ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
