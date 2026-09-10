"""Classification settings: model ids and concurrency, from flags, env or screenshots.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from vex import config

CONFIG_FILENAME = "vex.toml"
#: What version 1 called it. Read when there is no vex.toml beside it.
LEGACY_CONFIG_FILENAME = "screenshots.toml"

ENV_TEXT_MODEL = "VEX_TEXT_MODEL"
ENV_VISION_MODEL = "VEX_VISION_MODEL"
ENV_CONCURRENCY = "VEX_CONCURRENCY"
ENV_MAC_INBOX = "VEX_MAC_INBOX"

#: The version 1 spellings, still read. See config.env_value.
LEGACY_ENV_TEXT_MODEL = "SCREENSHOTS_TEXT_MODEL"
LEGACY_ENV_VISION_MODEL = "SCREENSHOTS_VISION_MODEL"
LEGACY_ENV_CONCURRENCY = "SCREENSHOTS_CONCURRENCY"
LEGACY_ENV_MAC_INBOX = "SCREENSHOTS_MAC_INBOX"

#: See docs/model-choice.md for how these were picked.
DEFAULT_TEXT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_VISION_MODEL = "google/gemini-2.5-flash-lite"

DEFAULT_CONCURRENCY = 6
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 8


@dataclass(frozen=True, slots=True)
class ClassifySettings:
    """Resolved model ids and worker count for a classify or bakeoff run."""

    text_model: str
    vision_model: str
    concurrency: int


def config_path(root: Path) -> Path | None:
    """The root's config file: ``vex.toml``, else the version 1 ``screenshots.toml``."""
    for name in (CONFIG_FILENAME, LEGACY_CONFIG_FILENAME):
        path = Path(root) / name
        if path.is_file():
            return path
    return None


def read_toml(root: Path) -> dict[str, object]:
    """The whole of the root's config file, or an empty mapping."""
    path = config_path(root)
    if path is None:
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def read_config_file(root: Path) -> dict[str, object]:
    """The ``[classify]`` table from the root's config file, if there is one."""
    section = read_toml(root).get("classify")
    return section if isinstance(section, dict) else {}


#: Where macOS is told to drop its screenshots. Deliberately a path in the home
#: directory rather than under the root: once the root lives on an external drive,
#: screenshots still have somewhere to land while the drive is unplugged, and get
#: swept up on the next ingest after it is back.
DEFAULT_MAC_INBOX = Path.home() / "Screenshots" / "inbox" / "mac"


def mac_inbox(root: Path) -> Path:
    """The macOS drop zone: env, then the config file, then the home default."""
    from_env = config.env_value(ENV_MAC_INBOX, LEGACY_ENV_MAC_INBOX)
    if from_env:
        return Path(from_env).expanduser()

    parsed = read_toml(root)
    paths = parsed.get("paths")
    configured = paths.get("mac_inbox") if isinstance(paths, dict) else None
    if configured is None:
        configured = parsed.get("mac_inbox")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser()
    return DEFAULT_MAC_INBOX


def _clamp_concurrency(value: object, fallback: int) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return fallback
    return max(MIN_CONCURRENCY, min(MAX_CONCURRENCY, number))


def resolve(
    root: Path,
    *,
    model: str | None = None,
    concurrency: int | None = None,
) -> ClassifySettings:
    """Work out which models to use. Explicit flag, then env, then file, then default.

    A ``--model`` override applies to both tiers, since it is a deliberate one-run choice.
    """
    config.load_env()
    file_settings = read_config_file(root)

    text_model = (
        model
        or config.env_value(ENV_TEXT_MODEL, LEGACY_ENV_TEXT_MODEL)
        or str(file_settings.get("text_model") or "")
        or DEFAULT_TEXT_MODEL
    )
    vision_model = (
        model
        or config.env_value(ENV_VISION_MODEL, LEGACY_ENV_VISION_MODEL)
        or str(file_settings.get("vision_model") or "")
        or DEFAULT_VISION_MODEL
    )
    workers = DEFAULT_CONCURRENCY
    if "concurrency" in file_settings:
        workers = _clamp_concurrency(file_settings["concurrency"], workers)
    from_env = config.env_value(ENV_CONCURRENCY, LEGACY_ENV_CONCURRENCY)
    if from_env:
        workers = _clamp_concurrency(from_env, workers)
    if concurrency is not None:
        workers = _clamp_concurrency(concurrency, workers)

    return ClassifySettings(
        text_model=text_model,
        vision_model=vision_model,
        concurrency=workers,
    )
