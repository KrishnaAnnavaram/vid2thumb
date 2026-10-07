"""Audio, transcripts, frames (problem 5) and metrics (problem 6)."""
import shutil

import numpy as np
import pytest
from PIL import Image, ImageFilter

from vid2thumb.evaluate import prompt_checks, rouge_l, rouge_n, summary_metrics, support, wer
from vid2thumb.frames import exposure, rank_frames, scene_boundaries, select_keyframes, sharpness
from vid2thumb.ingest import IngestError, licence_allows, resolve
from vid2thumb.media import Frame, MediaError, extract_audio, load_frames, read_wav, speech_regions, trim_silence, write_wav
from vid2thumb.transcript import load_transcript, parse_subtitles


def test_wav_round_trip(tmp_path):
    x = (0.5 * np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
    write_wav(tmp_path / "a.wav", x, 16000)
    y, rate = read_wav(tmp_path / "a.wav")
    assert rate == 16000 and np.allclose(x, y, atol=1e-4)


def test_speech_regions_merge_short_silences():
    rate = 16000
    tone = 0.3 * np.ones(rate // 2, dtype=np.float32)
    gap_short, gap_long = np.zeros(rate // 2, np.float32), np.zeros(2 * rate, np.float32)
    audio = np.concatenate([tone, gap_short, tone, gap_long, tone])
    regions = speech_regions(audio, rate, pad_ms=0)
    assert len(regions) == 2  # the 0.5 s gap stays, the 2 s gap splits


def test_trim_silence_maps_times_back(sample):
    samples, rate = read_wav(sample / "audio.wav")
    trimmed, tmap = trim_silence(samples, rate)
    assert len(trimmed) < len(samples)
    assert tmap.to_original(0.0) == pytest.approx(tmap.regions[0][0])
    second_start = tmap.regions[0][1] - tmap.regions[0][0]
    assert tmap.to_original(second_start + 0.01) == pytest.approx(tmap.regions[1][0] + 0.01, abs=1e-6)


def test_subtitle_parsing():
    srt = "1\n00:00:01,000 --> 00:00:02,500\nHello <i>there</i>\n\n2\n00:00:03,000 --> 00:00:04,000\nBye\n"
    t = parse_subtitles(srt)
    assert [s.text for s in t.segments] == ["Hello there", "Bye"] and t.segments[0].end == 2.5
    vtt = "WEBVTT\n\n00:01.000 --> 00:02.000\nHi\n"
    assert parse_subtitles(vtt).segments[0].start == 1.0
    again = parse_subtitles(t.to_srt())
    assert [s.text for s in again.segments] == ["Hello there", "Bye"]


def test_sidecar_file_of_sample(sample):
    t = load_transcript(sample / "transcript.srt")
    assert len(t.segments) == 8 and t.duration > 10


def test_wer_rouge_and_support():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert wer("the cat sat", "the bat sat down") == pytest.approx(2 / 3)
    assert rouge_n("a b c d", "a b c d", 1) == 1.0
    assert rouge_n("a b c d", "a b x y", 2) == pytest.approx(0.333, abs=0.01)
    assert rouge_l("a b c d", "a c d") == pytest.approx(2 * 1 * 0.75 / 1.75)
    s = support("Bake the bread for 45 minutes in Paris.", "bake bread twenty minutes")
    assert "paris" in s["unsupported"] and "45" in s["unsupported"] and s["support"] < 1


def test_summary_metrics_and_prompt_checks():
    m = summary_metrics("bread is baked", "bread is baked today", reference="bread is baked")
    assert m["rouge1"] == 1.0 and m["support"] == 1.0
    assert prompt_checks("x" * 50, 40)["within_limit"] is False


def _frame(i, pixels):
    return Frame(i, float(i), None, pixels)


def test_scene_detection_and_quality(sample):
    frames = load_frames(sorted((sample / "frames").glob("*.png")))
    import json

    ref = json.loads((sample / "reference.json").read_text(encoding="utf-8"))
    assert scene_boundaries(frames) == ref["scene_starts"]
    keys = select_keyframes(frames)
    assert len(keys) == 3
    assert ref["blurred_frame"] not in [k.frame.index for k in keys]


def test_sharpness_and_exposure():
    rng = np.random.default_rng(0)
    sharp = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    blurred = np.asarray(Image.fromarray(sharp).filter(ImageFilter.GaussianBlur(3)))
    assert sharpness(sharp) > 5 * sharpness(blurred)
    assert exposure(np.full((4, 4, 3), 128, np.uint8)) > 0.95
    assert exposure(np.zeros((4, 4, 3), np.uint8)) == 0.0


def test_relevance_scorer_changes_the_ranking():
    """Problem 5: real frames are ranked against the summary when a scorer is given."""
    a = _frame(0, np.full((20, 20, 3), 200, np.uint8))
    b = _frame(1, np.full((20, 20, 3), 90, np.uint8))
    keys = select_keyframes([a, b], threshold=0.1)

    class Fake:
        name = "fake"

        def score(self, images, text):
            return np.array([10.0 if img.mean() < 100 else 1.0 for img in images])

    ranked = rank_frames(keys, "a dark kitchen", Fake())
    assert ranked[0].frame.index == 1 and ranked[0].relevance == 10.0


def test_resolve_and_licence(sample, tmp_path):
    assert resolve(str(sample)).kind == "sample"
    with pytest.raises(IngestError):
        resolve("https://example.com/v")
    with pytest.raises(IngestError):
        resolve(str(tmp_path / "missing.mp4"))
    assert licence_allows("Creative Commons Attribution license (reuse allowed)", False)
    assert not licence_allows("Standard YouTube License", False)
    assert licence_allows(None, True)


def test_ffmpeg_audio_extraction(tmp_path, sample):
    if shutil.which("ffmpeg") is None:
        with pytest.raises(MediaError):
            extract_audio(tmp_path / "x.mp4", tmp_path / "x.wav")
        pytest.skip("ffmpeg is not installed")
    out = extract_audio(sample / "audio.wav", tmp_path / "copy.wav")
    assert read_wav(out)[1] == 16000
