# AGENTS.md

Orientation for coding agents working in this repository. `SPEC.md` is the build contract
and `README.md` is written for a user; this file is the short version for someone about to
change the code.

## Setup

```bash
uv sync --extra mac        # macOS, with Apple Vision and the Photos integration
uv sync --extra tesseract  # elsewhere, with the untested Tesseract fallback
uv sync                    # no OCR engine at all
```

Never activate the virtual environment by hand. Run everything through `uv run`.

## Checks

```bash
uv run ruff check          # lint
uv run ruff format --check # formatting
uv run pytest              # the whole suite, about ten seconds
```

All three must pass before anything is committed. Ruff is configured with `select = ["ALL"]`
and an explicit ignore list in `pyproject.toml`; add to that list rather than scattering
`noqa` comments.

Tests use synthetic images generated with Pillow. No real screenshots are checked in, and no
test may reach the network, the Photos library or a real library root. The autouse fixtures
in `tests/conftest.py` redirect the macOS drop zone into `tmp_path` and reset the taxonomy
between tests; leave them alone.

## Command map

| Command | Module |
| --- | --- |
| `ingest` | `vex/commands/ingest.py` |
| `thumbs` | `vex/commands/thumbs.py` |
| `ocr` | `vex/commands/ocr.py` |
| `faces` | `vex/commands/faces.py` |
| `classify` | `vex/commands/classify.py` |
| `bakeoff` | `vex/commands/bakeoff.py` |
| `views` | `vex/commands/views.py` |
| `notes` | `vex/commands/notes.py` |
| `search` | `vex/commands/search.py` |
| `stats` | `vex/commands/stats.py` |
| `purge` | `vex/commands/purge.py` |
| `keep` / `unkeep` | `vex/commands/keep.py` |
| `run` | `vex/commands/run.py` |
| `photos export` / `delete` / `backlog` | `vex/commands/photos.py` |

Every command module exposes a `command` callable, registered in `vex/cli.py`. Every
command takes `--root` and resolves it through `config.get_root()`.

## Module map

| Module | What lives there |
| --- | --- |
| `config.py` | Root resolution, the on-disk layout, relative-path helpers. Resolving a root also loads its taxonomy. |
| `db.py` | Schema, migrations, connection handling, the FTS index. |
| `taxonomy.py` | The taxonomy loader and validator, the rules tier, result normalisation, the JSON schema handed to models. |
| `default_taxonomy.toml` | The generic built-in taxonomy, used when a root has none of its own. |
| `classify.py` | The three tiers, the prompts, the cache, the run loop and the dry-run costing. |
| `openrouter.py` | The HTTP client: chat completions, JSON coaxing, live pricing, retry and backoff. |
| `bakeoff.py` | Model comparison over a sample, and the JSON dump. |
| `capture.py` | Capture-date extraction in the precedence order the spec sets out. |
| `files.py` | Hashing, image properties, iCloud dataless-file handling. |
| `naming.py` | Filename patterns, source detection, exFAT-safe naming. |
| `thumbs.py` | Thumbnail generation. |
| `ocr.py` | Text recognition. **macOS-only APIs live here.** |
| `faces.py` | Face counting. **macOS-only APIs live here.** |
| `photos.py` | Photos library export and deletion. **macOS-only APIs live here.** |
| `purge.py` | Which originals may go, and why the rest may not. |
| `backlog.py` | The unattended overnight loop. |
| `settings.py` | Model ids and concurrency, from flags, env or `vex.toml`. |
| `console.py` | Shared console and table helpers. |

## Invariants

These are not style preferences. Breaking one loses somebody's files.

1. **Never delete anything outside `library/` and `views/`.** `views/` is generated and may be
   rebuilt at will. `library/` originals may only go through `purge`, and only when the rules
   in `purge.py` all pass. Everything else, including `inbox/`, `duplicates/`, `skipped/`,
   `thumbs/` and `notes/`, is either moved or left alone. Notes are the one exception, and
   only ones no catalogue row claims.
2. **Never run `photos delete` without `--yes`, and never call it from the backlog loop.**
   Deleting from Photos is a separate, deliberate, human-confirmed step. The loop exports and
   purges; it must never delete from Photos.
3. **Treat the root's own `inbox/` as trusted and everything else as the user's.** Files
   inside the inbox may be moved, parked in `skipped/` or discarded as re-exports. A file
   somewhere else that ingest was pointed at is read and never touched, and its sidecar is
   never removed.
4. **Every path in the catalogue is relative to the root, with forward slashes.** Use
   `config.relative_path()` and `config.absolute_path()`. Storing an absolute path breaks the
   moment the root moves to an external drive.
5. **No symlinks, no hard links, ever.** The drive may be exFAT. `views/` holds copies.
6. **Commit per row.** Long passes over thousands of rows commit each row as it completes, so
   an interrupted run keeps everything it had already done. Do not batch commits to make a
   loop faster.
7. **macOS-only APIs stay in `ocr.py`, `faces.py` and `photos.py`.** They raise a named
   `...UnavailableError` with a message saying what to install. No other module may import
   pyobjc, osxphotos or photoscript.
8. **`--dry-run` changes nothing and makes no paid calls.** `classify --dry-run` uses the
   public pricing list and works without an API key.
9. **Never print or log an API key.**

## Conventions

- **Australian English** in code, comments, docstrings and output. Not American spelling.
- **No emojis** anywhere: output, comments, commit messages, documentation.
- **Stage git changes explicitly** with `git add <paths>`. Never `git add -A` or `git add .`.
- Comments explain why, not what. If a constant has a reason behind it, the reason goes in a
  `#:` comment above it.
- Type annotations everywhere, `from __future__ import annotations` at the top of every
  module.
- Errors the user can fix are printed as one red line with a non-zero exit, not a traceback.
- Prefer adding to the taxonomy file over adding to the code. Categories, keywords, structured
  shapes and the purge policy are data.
- **Read the environment through `config.env_value()`.** Version 1 spelled the variables
  `SCREENSHOTS_*` and called the config file `screenshots.toml`. Both are still read, with
  the `VEX_*` name winning when both are set, and are due to go after one release. Do not
  reach for `os.environ` directly for any of them.
- The SQLite table is named `screenshots` and stays that way. Renaming it would orphan every
  existing catalogue.
