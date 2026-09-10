"""Root directory resolution and the on-disk layout."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from vex import taxonomy

ENV_ROOT = "VEX_ROOT"
#: The version 1 spelling. Still read, and due to go after one release.
LEGACY_ENV_ROOT = "SCREENSHOTS_ROOT"

#: The default stays ~/Screenshots: it is named for what the folder holds,
#: not for the tool that fills it, and an existing library should not move.
DEFAULT_ROOT = Path.home() / "Screenshots"

INBOX_SOURCES = ("mac", "iphone", "other")

LAYOUT_DIRS = (
    "inbox/mac",
    "inbox/iphone",
    "inbox/other",
    "library",
    "duplicates",
    "skipped",
    "thumbs",
    "views",
    "notes",
)

_dotenv_loaded = False


def load_env() -> None:
    """Load a .env from the current working directory tree, once."""
    global _dotenv_loaded  # noqa: PLW0603
    if not _dotenv_loaded:
        load_dotenv()
        _dotenv_loaded = True


def env_value(*names: str) -> str | None:
    """The first of these environment variables that is set to something.

    The ``VEX_`` names arrived with version 2. The ``SCREENSHOTS_`` ones are what
    version 1 used, and are still read so an existing setup keeps working. They
    are due to go after one release, and the new name always wins.
    """
    load_env()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve_root(root: Path | str | None = None) -> Path:
    """Resolve the library root: explicit argument, then env, then ~/Screenshots."""
    load_env()
    if root is not None:
        return Path(root).expanduser().resolve()
    from_env = env_value(ENV_ROOT, LEGACY_ENV_ROOT)
    if from_env:
        return Path(from_env).expanduser().resolve()
    return DEFAULT_ROOT.resolve()


def ensure_layout(root: Path) -> Path:
    """Create the directory layout under root if it is not already there."""
    root = Path(root)
    for relative in LAYOUT_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def get_root(root: Path | str | None = None) -> Path:
    """Resolve the root, make sure the layout exists, and load the library's taxonomy.

    Every command starts here, so this is where a library's own ``taxonomy.toml``
    takes effect. Without one, the built-in default stands.
    """
    resolved = ensure_layout(resolve_root(root))
    taxonomy.load(resolved)
    return resolved


def relative_path(root: Path, path: Path) -> str:
    """Path relative to root, with forward slashes, for storage in the catalogue."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def absolute_path(root: Path, relative: str) -> Path:
    """Turn a stored relative path back into an absolute path."""
    return Path(root) / Path(relative)


def thumb_path(root: Path, file_hash: str) -> Path:
    """Thumbnail location for a given file hash."""
    return Path(root) / "thumbs" / f"{file_hash}.jpg"
