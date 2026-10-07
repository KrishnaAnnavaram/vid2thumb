"""Transcript data: timed segments, SRT / WebVTT read and write."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    source: str = ""

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def duration(self) -> float:
        return max((s.end for s in self.segments), default=0.0)

    def to_json(self) -> str:
        return json.dumps({"language": self.language, "source": self.source, "segments": [asdict(s) for s in self.segments]}, indent=2)

    def to_srt(self) -> str:
        blocks = []
        for i, s in enumerate(self.segments, 1):
            blocks.append(f"{i}\n{_fmt(s.start)} --> {_fmt(s.end)}\n{s.text.strip()}\n")
        return "\n".join(blocks)


def _seconds(match: re.Match) -> float:
    h, m, s, ms = match.groups()
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _fmt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_subtitles(text: str, source: str = "") -> Transcript:
    """Parse SRT or WebVTT text. Cue numbers, the WEBVTT header and tags are removed."""
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            if "-->" in line:
                times = list(_TIME.finditer(line))
                if len(times) < 2:
                    break
                body = " ".join(lines[i + 1 :])
                body = re.sub(r"<[^>]+>", "", body).strip()
                if body:
                    segments.append(Segment(_seconds(times[0]), _seconds(times[1]), body))
                break
    return Transcript(segments=segments, source=source)


def load_transcript(path: str | Path) -> Transcript:
    """Read ``.srt``, ``.vtt``, ``.json`` (this project) or ``.txt`` (one segment, no times)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".srt", ".vtt"}:
        return parse_subtitles(text, source=path.name)
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return Transcript([Segment(**s) for s in data["segments"]], data.get("language", "en"), data.get("source", path.name))
    return Transcript([Segment(0.0, 0.0, " ".join(text.split()))], source=path.name)
