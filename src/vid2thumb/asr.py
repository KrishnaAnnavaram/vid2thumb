"""Speech-to-text behind one small interface.

* ``SidecarTranscriber``: reads an existing subtitle or transcript file (offline).
* ``WhisperTranscriber``: faster-whisper (extra ``asr``). It runs on the CPU or a GPU,
  trims silence first and maps the segment times back to the original audio.
"""
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import Protocol

from .ingest import Source
from .media import extract_audio, read_wav, trim_silence, write_wav
from .transcript import Segment, Transcript, load_transcript


class TranscriptionError(RuntimeError):
    pass


class Transcriber(Protocol):
    name: str

    def transcribe(self, source: Source) -> Transcript: ...


class SidecarTranscriber:
    name = "sidecar"

    def transcribe(self, source: Source) -> Transcript:
        path = source.sidecar()
        if path is None:
            raise TranscriptionError(
                f"{source.path}: no transcript file found (.srt, .vtt, .json or .txt). Use VID2THUMB_TRANSCRIBER=whisper"
            )
        return load_transcript(path)


def detect_device() -> tuple[str, str]:
    """``("cuda", "float16")`` if torch sees a GPU, else ``("cpu", "int8")``."""
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


class WhisperTranscriber:
    name = "whisper"

    def __init__(self, model_size: str = "small", language: str | None = None) -> None:
        fw = importlib.import_module("faster_whisper")  # optional extra "asr"
        device, compute = detect_device()
        self.model = fw.WhisperModel(model_size, device=device, compute_type=compute)
        self.language = language
        self.device = device

    def transcribe(self, source: Source) -> Transcript:
        with tempfile.TemporaryDirectory() as tmp:
            wav = source.audio or extract_audio(source.path, Path(tmp) / "audio.wav")
            samples, rate = read_wav(wav)
            trimmed, tmap = trim_silence(samples, rate)
            if trimmed.size == 0:
                return Transcript([], source=f"whisper:{source.path.name}")
            clean = Path(tmp) / "speech.wav"
            write_wav(clean, trimmed, rate)
            # temperature 0 and beam search: the same audio gives the same text
            segments, info = self.model.transcribe(str(clean), language=self.language, temperature=0.0, beam_size=5)
            out = [Segment(tmap.to_original(s.start), tmap.to_original(s.end), s.text.strip()) for s in segments]
        return Transcript(out, language=info.language, source=f"whisper:{source.path.name}")


def build_transcriber(name: str, model_size: str = "small") -> Transcriber:
    if name == "sidecar":
        return SidecarTranscriber()
    if name == "whisper":
        return WhisperTranscriber(model_size)
    raise ValueError(f"unknown transcriber {name!r}")
