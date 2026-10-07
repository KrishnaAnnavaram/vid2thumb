"""The full run, the outputs on disk (problem 9), secrets (problem 1), reproducibility (problem 10) and the CLI."""
import json
import re
from pathlib import Path

import pytest

from vid2thumb.asr import SidecarTranscriber, TranscriptionError
from vid2thumb.cli import main
from vid2thumb.config import ConfigError, Settings
from vid2thumb.generate import PlaceholderGenerator
from vid2thumb.ingest import resolve
from vid2thumb.pipeline import Components, run
from vid2thumb.summarize import ExtractiveSummarizer

SRC = Path(__file__).resolve().parents[1] / "src"


def _parts():
    return Components(SidecarTranscriber(), ExtractiveSummarizer(), PlaceholderGenerator(size=(160, 90)))


def test_run_writes_every_output(sample, tmp_path):
    settings = Settings(out_dir=tmp_path)
    result = run(resolve(str(sample)), settings, _parts(), run_id="r1")
    for name in ("transcript.json", "transcript.srt", "summary.txt", "prompt.txt", "thumbnail_generated.png",
                 "thumbnail_frame.png", "metrics.json", "run.json"):
        assert (result.run_dir / name).exists(), name
    assert len(list((result.run_dir / "keyframes").glob("*.png"))) == 3
    m = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert 0 < m["transcript"]["wer"] < 0.2
    assert m["frames"]["scenes"] == m["frames"]["reference_scenes"] == 3
    assert m["prompt"]["within_limit"] and m["summary"]["support"] == 1.0


def test_two_runs_give_identical_outputs(sample, tmp_path):
    settings = Settings(out_dir=tmp_path)
    a = run(resolve(str(sample)), settings, _parts(), run_id="a")
    b = run(resolve(str(sample)), settings, _parts(), run_id="b")
    assert a.summary == b.summary and a.prompt == b.prompt
    assert a.generated.read_bytes() == b.generated.read_bytes()
    assert a.best_frame.read_bytes() == b.best_frame.read_bytes()


def test_api_key_is_never_written(sample, tmp_path, monkeypatch):
    """Problem 1: the key comes from the environment and is not in the run record or repr."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-value-123")
    settings = Settings.from_env()
    assert "test-key-value-123" not in repr(settings)
    assert settings.public()["openai_api_key_set"] is True
    result = run(resolve(str(sample)), Settings(out_dir=tmp_path, openai_api_key="test-key-value-123"), _parts(), run_id="k")
    for f in result.run_dir.iterdir():
        if f.suffix in {".json", ".txt", ".srt"}:
            assert "test-key-value-123" not in f.read_text(encoding="utf-8")


def test_no_key_pattern_in_source_code():
    pattern = re.compile(r"sk-[A-Za-z0-9]{20,}")
    for f in SRC.rglob("*.py"):
        assert not pattern.search(f.read_text(encoding="utf-8")), f


def test_missing_sidecar_is_a_clear_error(tmp_path):
    folder = tmp_path / "s"
    (folder / "frames").mkdir(parents=True)
    with pytest.raises(TranscriptionError, match="whisper"):
        SidecarTranscriber().transcribe(resolve(str(folder)))


@pytest.mark.parametrize("name,value", [("VID2THUMB_SUMMARIZER", "pegasus"), ("VID2THUMB_SEED", "x")])
def test_bad_settings(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_cli_commands(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["make-sample", "--out", "s1", "--topic", "bicycle"]) == 0
    assert main(["run", "--input", "s1", "--run-id", "x"]) == 0
    out = capsys.readouterr().out
    assert "WER" in out and (tmp_path / "outputs" / "x" / "thumbnail_frame.png").exists()
    assert main(["frames", "--input", "s1"]) == 0
    assert "scene 0" in capsys.readouterr().out
    (tmp_path / "t.txt").write_text("Bread needs flour. Bread needs water. The oven is hot.", encoding="utf-8")
    assert main(["summarize", "--text", "t.txt", "--words", "10"]) == 0
    assert "prompt:" in capsys.readouterr().out
    (tmp_path / "m.jsonl").write_text(json.dumps({"input": "s1"}), encoding="utf-8")
    assert main(["evaluate", "--manifest", "m.jsonl"]) == 0
    assert json.loads((tmp_path / "outputs" / "evaluation.json").read_text())["items"] == 1


def test_cli_error_code(capsys):
    assert main(["run", "--input", "no_such_folder"]) == 1
    assert "error:" in capsys.readouterr().err


def test_llm_summarizer_without_key_fails_cleanly(monkeypatch, capsys, sample):
    monkeypatch.setenv("VID2THUMB_SUMMARIZER", "llm")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["run", "--input", str(sample)]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_whisper_adapter_loads():
    pytest.importorskip("faster_whisper")
    from vid2thumb.asr import detect_device

    assert detect_device()[0] in {"cpu", "cuda"}
