"""Audio and frame helpers.

* WAV read and write with the standard ``wave`` module (16-bit PCM).
* Energy-based speech detection and silence trimming, with a time map back to the
  original timeline, so that transcript times stay correct after trimming.
* ``ffmpeg`` wrappers to get 16 kHz mono audio and frames from a video file. The
  ``ffmpeg`` program is necessary only for real video files.
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


class MediaError(RuntimeError):
    """A media file cannot be read, or a necessary program is absent."""


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Return mono float samples in [-1, 1] and the sample rate."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise MediaError(f"{path}: only 16-bit PCM WAV is supported")
        rate, channels = wf.getframerate(), wf.getnchannels()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def write_wav(path: str | Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


@dataclass
class TimeMap:
    """Kept regions of the original audio, as (start, end) in seconds."""

    regions: list[tuple[float, float]]

    def to_original(self, t: float) -> float:
        """Map a time on the trimmed audio to the time on the original audio."""
        offset = 0.0
        for start, end in self.regions:
            length = end - start
            if t <= offset + length:
                return start + (t - offset)
            offset += length
        return self.regions[-1][1] if self.regions else t

    @property
    def kept_seconds(self) -> float:
        return sum(e - s for s, e in self.regions)


def speech_regions(
    samples: np.ndarray, rate: int, frame_ms: int = 30, threshold_db: float = -40.0, min_silence_ms: int = 1000,
    pad_ms: int = 150,
) -> list[tuple[float, float]]:
    """Regions with sound. A silence shorter than ``min_silence_ms`` stays inside a region."""
    hop = max(1, int(rate * frame_ms / 1000))
    n = len(samples) // hop
    if n == 0:
        return []
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    loud = 20 * np.log10(rms) > threshold_db
    regions: list[list[float]] = []
    for i in np.flatnonzero(loud):
        start, end = i * hop / rate, (i + 1) * hop / rate
        if regions and start - regions[-1][1] < min_silence_ms / 1000.0:
            regions[-1][1] = end
        else:
            regions.append([start, end])
    total = len(samples) / rate
    pad = pad_ms / 1000.0
    return [(max(0.0, s - pad), min(total, e + pad)) for s, e in regions]


def trim_silence(samples: np.ndarray, rate: int, **kwargs) -> tuple[np.ndarray, TimeMap]:
    regions = speech_regions(samples, rate, **kwargs)
    if not regions:
        return samples[:0], TimeMap([])
    parts = [samples[int(s * rate) : int(e * rate)] for s, e in regions]
    return np.concatenate(parts), TimeMap(regions)


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise MediaError("ffmpeg is not on PATH: install it to read video files")
    return exe


def extract_audio(video: str | Path, out_wav: str | Path, rate: int = 16000) -> Path:
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(rate), "-sample_fmt", "s16", str(out_wav)]
    subprocess.run(cmd, check=True, timeout=600)
    return Path(out_wav)


def extract_frames(video: str | Path, out_dir: str | Path, fps: float = 1.0, width: int = 640) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps},scale={width}:-2", str(out / "frame_%05d.png")]
    subprocess.run(cmd, check=True, timeout=900)
    return sorted(out.glob("frame_*.png"))


def load_frames(paths: list[Path], fps: float = 1.0) -> list["Frame"]:
    frames = []
    for i, p in enumerate(sorted(paths)):
        with Image.open(p) as img:
            frames.append(Frame(index=i, time=i / fps, path=p, pixels=np.asarray(img.convert("RGB"))))
    return frames


@dataclass
class Frame:
    index: int
    time: float
    path: Path | None
    pixels: np.ndarray  # H x W x 3, uint8
