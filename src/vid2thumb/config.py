"""Settings from environment variables. Secrets come only from the environment or a local .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

TRANSCRIBERS = ("sidecar", "whisper")
SUMMARIZERS = ("extractive", "bart", "llm")
GENERATORS = ("placeholder", "openai")
RANKERS = ("quality", "clip")


class ConfigError(ValueError):
    """An environment variable has a value that the code cannot use."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Read ``KEY=VALUE`` lines from a local ``.env`` file. Existing variables win."""
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _get(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _num(name: str, default, kind=int):
    raw = _get(name, str(default))
    try:
        return kind(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = _get(name, default).lower()
    if value not in choices:
        raise ConfigError(f"{name} must be one of {choices}, got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    out_dir: Path = field(default_factory=lambda: Path("outputs"))
    transcriber: str = "sidecar"
    summarizer: str = "extractive"
    generator: str = "placeholder"
    ranker: str = "quality"
    summary_words: int = 60
    whisper_model: str = "small"
    llm_model: str = "gpt-4o-mini"
    image_model: str = "dall-e-3"
    max_video_minutes: float = 30.0
    seed: int = 42
    openai_api_key: str = field(default="", repr=False)  # never printed

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            out_dir=Path(_get("VID2THUMB_OUT", "outputs")),
            transcriber=_choice("VID2THUMB_TRANSCRIBER", "sidecar", TRANSCRIBERS),
            summarizer=_choice("VID2THUMB_SUMMARIZER", "extractive", SUMMARIZERS),
            generator=_choice("VID2THUMB_GENERATOR", "placeholder", GENERATORS),
            ranker=_choice("VID2THUMB_RANKER", "quality", RANKERS),
            summary_words=_num("VID2THUMB_SUMMARY_WORDS", 60),
            whisper_model=_get("VID2THUMB_WHISPER_MODEL", "small"),
            llm_model=_get("VID2THUMB_LLM_MODEL", "gpt-4o-mini"),
            image_model=_get("VID2THUMB_IMAGE_MODEL", "dall-e-3"),
            max_video_minutes=_num("VID2THUMB_MAX_MINUTES", 30.0, float),
            seed=_num("VID2THUMB_SEED", 42),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
        )

    def public(self) -> dict:
        """Settings for the run record. The API key is replaced by a yes/no value."""
        return {
            "transcriber": self.transcriber,
            "summarizer": self.summarizer,
            "generator": self.generator,
            "ranker": self.ranker,
            "summary_words": self.summary_words,
            "whisper_model": self.whisper_model,
            "llm_model": self.llm_model,
            "image_model": self.image_model,
            "max_video_minutes": self.max_video_minutes,
            "seed": self.seed,
            "openai_api_key_set": bool(self.openai_api_key),
        }
