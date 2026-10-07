"""Input resolution: a local video file, a sample folder or a URL (extra ``download``).

A URL is downloaded only if the licence is in the allow list or the user states that
they own the video. Downloads of videos that you do not have the right to use can
break the terms of the platform.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
ALLOWED_LICENCES = ("creative commons", "cc by", "cc0", "public domain")


class IngestError(ValueError):
    """The input cannot be used."""


@dataclass
class Source:
    kind: str  # "video" or "sample"
    path: Path
    title: str
    licence: str = "unknown"

    @property
    def audio(self) -> Path | None:
        p = self.path / "audio.wav"
        return p if self.kind == "sample" and p.exists() else None

    @property
    def frames_dir(self) -> Path | None:
        p = self.path / "frames"
        return p if self.kind == "sample" and p.is_dir() else None

    def sidecar(self) -> Path | None:
        """A transcript file next to the media: same stem (video) or ``transcript.*`` (sample)."""
        base = self.path / "transcript" if self.kind == "sample" else self.path.with_suffix("")
        for suffix in (".srt", ".vtt", ".json", ".txt"):
            candidate = base.with_suffix(suffix) if self.kind == "video" else Path(str(base) + suffix)
            if candidate.exists():
                return candidate
        return None


def resolve(path_or_url: str) -> Source:
    if path_or_url.startswith(("http://", "https://")):
        raise IngestError("for a URL, use download_url(), or the CLI option --url with --i-own-this or a free licence")
    path = Path(path_or_url)
    if path.is_dir():
        if not (path / "audio.wav").exists() and not (path / "frames").is_dir():
            raise IngestError(f"{path}: a sample folder needs audio.wav, frames/ or both")
        return Source("sample", path, path.name)
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return Source("video", path, path.stem)
    raise IngestError(f"{path}: not a video file ({sorted(VIDEO_SUFFIXES)}) and not a sample folder")


def licence_allows(licence: str | None, i_own_this: bool) -> bool:
    if i_own_this:
        return True
    text = (licence or "").lower()
    return any(word in text for word in ALLOWED_LICENCES)


def download_url(url: str, out_dir: str | Path, i_own_this: bool = False, max_minutes: float = 30.0) -> Source:
    """Download a video with yt-dlp after a licence check and a duration check."""
    yt_dlp = importlib.import_module("yt_dlp")  # optional extra "download"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    licence = info.get("license") or "unknown"
    if not licence_allows(licence, i_own_this):
        raise IngestError(f"licence {licence!r} is not in the allow list: use --i-own-this only for your own video")
    minutes = (info.get("duration") or 0) / 60.0
    if minutes > max_minutes:
        raise IngestError(f"video has {minutes:.1f} minutes, the limit is {max_minutes} (VID2THUMB_MAX_MINUTES)")
    template = str(out / "%(id)s.%(ext)s")
    with yt_dlp.YoutubeDL({"quiet": True, "outtmpl": template, "format": "mp4/bestvideo+bestaudio/best"}) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    return Source("video", path, info.get("title", path.stem), licence)
