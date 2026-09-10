# screenshots - build spec

A cross-platform (macOS + Windows) CLI that consolidates screenshots from many sources into one library, catalogues them in SQLite, OCRs them, classifies them, and exposes them via full-text search, category folder views, and Markdown notes.

This file is the build contract: what the tool has to do and the rules it has to hold to.

## Stack and conventions

- Python 3.12+, `uv` project, `ruff` clean, `pytest`. Load the `modern-python` skill before scaffolding.
- Console script: `screenshots` (package dir `vex/`). CLI via `typer`. Progress output via `rich`.
- Cross-platform: `pathlib` everywhere, **no symlinks, no hard links** (the drive will be exFAT). Filenames must be exFAT- and Windows-safe: no `:` `*` `?` `"` `<` `>` `|`, no trailing dots. Never rely on case sensitivity.
- Everything idempotent and re-runnable. Never delete a user's file except: moving inbox files into the library, and rebuilding the generated `views/` tree.
- Australian English in all text. No emojis anywhere.
- Config: root directory from `--root`, else env `VEX_ROOT`, else `~/Screenshots`. All paths stored in the catalogue are **relative to root** with forward slashes, so the whole root can be moved to an external drive later.
- Secrets: `OPENROUTER_API_KEY` from the environment or a `.env` in the repo (python-dotenv). Never commit `.env`. Never print keys.

## Layout under root

```
inbox/
  mac/        drop zone for macOS screenshots (Desktop etc.)
  iphone/     drop zone for Photos exports
  other/
library/
  YYYY/MM/    canonical files: <YYYYMMDD-HHMMSS>_<source>_<hash8>.<ext>
duplicates/   exact duplicates moved here on ingest (never silently deleted)
thumbs/       <hash>.jpg, max side 512px, quality 80
views/        generated, copies not links: views/<category>/<library filename>
notes/        generated Markdown: notes/<category>/<YYYYMMDD-HHMMSS>_<hash8>.md
catalog.sqlite
```

## Catalogue schema (SQLite, WAL mode)

```sql
CREATE TABLE screenshots (
  hash TEXT PRIMARY KEY,            -- sha256 hex of file bytes
  phash TEXT,                       -- perceptual hash (imagehash.phash, hex)
  path TEXT NOT NULL UNIQUE,        -- relative to root, forward slashes
  original_name TEXT NOT NULL,
  original_path TEXT,               -- where it came from, absolute, informational
  source TEXT NOT NULL,             -- 'mac' | 'iphone' | 'other'
  captured_at TEXT,                 -- ISO 8601 local time, seconds precision
  captured_at_source TEXT,          -- 'exif' | 'xmp' | 'filename' | 'sidecar' | 'birthtime' | 'mtime'
  device TEXT,
  width INTEGER, height INTEGER, bytes INTEGER, format TEXT,
  ingested_at TEXT NOT NULL,
  near_duplicate_of TEXT,           -- hash of an earlier row with phash within distance <= 6
  ocr_text TEXT, ocr_engine TEXT, ocr_at TEXT,
  category TEXT,                    -- see taxonomy
  tags TEXT,                        -- JSON array of strings
  caption TEXT,                     -- one line, <= 120 chars
  classify_confidence REAL,
  classify_method TEXT,             -- 'rules' | 'text-model' | 'vision-model' | 'manual'
  classify_model TEXT,              -- model id used, if any
  classified_at TEXT,
  structured TEXT,                  -- JSON: extra structure from the model (e.g. recipe), nullable
  is_junk INTEGER NOT NULL DEFAULT 0,
  reviewed INTEGER NOT NULL DEFAULT 0,
  notes TEXT,
  photos_uuid TEXT,                 -- Photos library UUID, when the file came from Photos
  faces INTEGER,                    -- count of faces detected; NULL means never checked
  faces_at TEXT,
  keep INTEGER NOT NULL DEFAULT 0,  -- pinned by hand: purge never touches it
  purged_at TEXT                    -- when the library original was deleted; path stays
);
CREATE TABLE ingest_log (
  id INTEGER PRIMARY KEY, run_at TEXT, source_path TEXT, hash TEXT,
  action TEXT,                      -- 'added' | 'duplicate' | 'skipped' | 'error'
  detail TEXT
);
CREATE TABLE classify_cache (
  hash TEXT, model TEXT, mode TEXT,  -- mode: 'text' | 'vision'
  response_json TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, created_at TEXT,
  PRIMARY KEY (hash, model, mode)
);
-- FTS5 external-content index over ocr_text, caption, tags, original_name, kept in sync by triggers.
CREATE VIRTUAL TABLE screenshots_fts USING fts5(ocr_text, caption, tags, original_name, content='screenshots', content_rowid='rowid');
```

`vex/db.py` owns the schema, migrations (simple `PRAGMA user_version`), and `connect(root) -> sqlite3.Connection`.

## Commands

### `screenshots ingest [PATHS...] [--source auto|mac|iphone|other] [--copy] [--dry-run]`
- With no PATHS: process everything under `inbox/`. With PATHS: files or directories anywhere (e.g. `~/Desktop`); only image files (`png jpg jpeg heic webp gif`) whose name matches a screenshot pattern **or** `--source` is given explicitly. Screenshot patterns: macOS `Screenshot YYYY-MM-DD at HH.MM.SS[ (n)].png`, `Screen Shot ...`, iPhone `IMG_NNNN.PNG/png` (including doubled extensions like `IMG_8425.jpg.png`), `Screenshot_*`, CleanShot names. Non-matching files are logged `skipped`, never touched. Directories are not recursed unless `--recursive`.
- **iCloud evicted files (macOS):** if a file is dataless (`st_size > 0` and `st_blocks == 0`, or reading raises), run `brctl download <path>` and poll until readable (timeout 120 s, then log `error` and continue). Report how many needed downloading.
- Per file: sha256; if hash exists in the catalogue -> move to `duplicates/<hash8>_<original_name>` (or leave in place with `--copy`), log `duplicate`. Otherwise: phash, dimensions, format, capture date, device, then move (default) or copy into `library/YYYY/MM/<YYYYMMDD-HHMMSS>_<source>_<hash8>.<ext>`, insert row, generate thumbnail, log `added`. `near_duplicate_of` = hash of any existing row whose phash Hamming distance <= 6 (full scan is fine; the table is small).
- Capture date precedence: EXIF DateTimeOriginal -> XMP/PNG text chunks (iOS screenshots often carry `xmp:CreateDate` or a `date:create` tEXt chunk) -> filename (macOS pattern, `Screenshot_DD-M-YYYY_HHMMSS...`) -> `<name>.json` sidecar next to the file (osxphotos style) -> min(st_birthtime, st_mtime) on macOS, st_mtime elsewhere. Record which one was used.
- Source detection when `auto`: macOS pattern -> `mac`; `IMG_` -> `iphone`; else `other`.
- Device: from EXIF Model if present, else `mac` / `iphone` by source.
- Summary table at the end: added, duplicates, near-duplicates, skipped, errors, downloaded-from-icloud.

### `screenshots thumbs [--force]`
Generate missing thumbnails (`thumbs/<hash>.jpg`, RGB, max side 512, quality 80).

### `screenshots ocr [--engine auto|vision|tesseract] [--limit N] [--force]`
- Rows with `ocr_text IS NULL` (or all with `--force`).
- `vision` (macOS only): Apple Vision `VNRecognizeTextRequest` via `pyobjc-framework-Vision` + `pyobjc-framework-Quartz`, accurate recognition level, language correction on, recognition languages `["en-AU", "en"]`. Text ordered top-to-bottom by bounding box, one observation per line. Optional dependency group `mac`.
- `tesseract`: `pytesseract` on the full-size image if the binary is on PATH. Optional dependency group `tesseract`.
- `auto`: vision on macOS if importable, else tesseract, else exit with a clear message.
- Store engine name and timestamp. Empty results store `""` not NULL so they are not retried forever.

### `screenshots faces [--limit N] [--force]`
macOS only, needs the `mac` extra. Rows with `faces IS NULL` (or all with
`--force`). Apple Vision `VNDetectFaceRectanglesRequest` over `thumbs/<hash>.jpg`
rather than the original: 512 px is ample for face rectangles and it keeps a long
pass off the full-size decode path. Stores the count in `faces` and the time in
`faces_at`, committing per row. Same autorelease-pool handling as the OCR pass,
so memory stays flat over thousands of rows. Only a count is stored: no face
location, no identifier, nothing amounting to recognising a person. On other
platforms the command exits with a message naming the reason.

`ingest` also face-checks newly added rows, best effort, so a normal run needs no
separate pass. Any failure there leaves `faces` null for a later `faces` run.

### `screenshots purge [--dry-run] [--yes] [--category C ...] [--min-text N]`
Deletes library **originals** whose value is already held by the note and the
thumbnail. The thumbnail, the note and the catalogue row always stay. `path` is
`NOT NULL UNIQUE` and keeps its historical value; `purged_at` is what says the
original is gone.

Eligible only when all of these hold:
- the category is marked `purgeable = true` in the taxonomy, **or** `is_junk = 1`,
  which is purgeable whatever its text says
- `faces = 0`; `NULL` means not yet checked, so not eligible
- `length(ocr_text) >= --min-text` (default 80), except for junk
- `thumbs/<hash>.jpg` exists
- `keep = 0`
- `purged_at IS NULL` and the library file is still there

Every other category is protected, as is anything with `faces > 0`. `--category`
narrows the purgeable set and can never widen it; naming a protected category is
an error.

Default is a dry run: files and bytes by category, plus a tally of why the rest
were held back (faces, thin text, category, unchecked, pinned). `--yes` performs
it, printing a running count every 200 rows. No Vision needed, so it runs on any
platform.

The rest of the CLI treats a purged row as a row without an original: `notes`
still renders it, from the thumbnail, with `purged` in the frontmatter; `views`
leaves it out; `stats` reports the purged count and the bytes reclaimed; `search`
still returns it and marks it; `ingest` sees a re-offered copy as a duplicate by
hash, logs it as such, and never resurrects the row.

### `screenshots keep HASH...` / `screenshots unkeep HASH...`
Set or clear the `keep` pin. Accepts full hashes or unique prefixes of at least 8
characters.

### `screenshots classify [--method auto|rules|text|vision] [--limit N] [--force] [--dry-run] [--model ID]`
Built by a second agent. Contract: reads `ocr_text`, `original_name`, `width/height`, `thumbs/<hash>.jpg`; writes `category`, `tags`, `caption`, `classify_confidence`, `classify_method`, `classify_model`, `classified_at`, `structured`, `is_junk`; caches raw model responses in `classify_cache`.

Three tiers, cheapest first:
1. `rules`: deterministic keyword matching on OCR text and filename. Free.
2. `text`: a cheap, fast text model via **OpenRouter** (OpenAI-compatible `/chat/completions`, base URL `https://openrouter.ai/api/v1`) given the OCR text plus dimensions and source. JSON output. Used when OCR text is substantial (>= 40 characters).
3. `vision`: a cheap vision model via OpenRouter with the thumbnail as image input. Used when OCR text is thin or the text tier reports confidence < 0.7.

`auto` runs 1 -> 2 -> 3 as needed. Model ids are configurable (`--model`, env `VEX_TEXT_MODEL` / `VEX_VISION_MODEL`, or `vex.toml` in root). Defaults chosen by benchmarking: the classify agent must fetch OpenRouter's live model list and pricing, shortlist the cheapest models that support image input and JSON output, run a small bake-off on ~20 real thumbnails, and record the results and the chosen defaults in `docs/model-choice.md`. Cost per call is computed from the usage block and stored. `--dry-run` prints how many rows would hit each tier and an estimated cost. Concurrency 4-8 with retry and backoff.

### Taxonomy (single primary `category`, free-form `tags` up to 6)
The taxonomy is data, not code. It is read from `taxonomy.toml` in the root, and from a generic built-in default (`vex/default_taxonomy.toml`) when the root has none. A library is expected to define its own.

Each `[[category]]` table carries a `name` (a lowercase slug, since it becomes a directory name), a `description` written for the model, an optional `structured` shape, a `purgeable` flag that drives the purge policy, and the `strong` / `weak` / `tags` keyword sets the rules tier scores against. A top-level `context` string opens the system prompt. `junk` and `other` are required: every capture has to have somewhere to land. Whitespace inside a keyword phrase is significant, so `"def "` does not match "default".

An invalid file is reported as one line naming the category and the key, with a non-zero exit, never a traceback.

`structured` JSON by shape:
- `recipe`: `{"title": str, "ingredients": [str], "method": [str], "serves": str|null, "source": str|null}`
- `use-note`: `{"use": str}` - one sentence on how this could be reused (a post angle, a product idea, a prompt to keep). Null if nothing.

### `screenshots views [--category C]`
Delete and rebuild `views/<category>/` as **copies** of library files for rows that are not junk. Also `views/_near-duplicates/` containing pairs for review, and `views/_unclassified/`.

### `screenshots notes [--category C] [--force]`
One Markdown file per non-junk row at `notes/<category>/<YYYYMMDD-HHMMSS>_<hash8>.md`. YAML frontmatter: hash, captured, source, category, tags, caption, confidence, path. Body: `![thumb](../../thumbs/<hash>.jpg)`, the caption, then the OCR text in a fenced block. For a category whose structured shape is `recipe`, with `structured` present: `# title`, `## Ingredients` list, `## Method` numbered list, then the OCR text under `## Source text`. For a `use-note` category, a `## Use` line from `structured.use`. Skip files that exist unless `--force`; delete notes for rows that became junk.

### `screenshots search QUERY [--category C] [--faces/--no-faces] [--limit 20] [--json]`
FTS5 `MATCH` with `bm25` ranking; prints captured date, category, caption, a snippet, and the library path. `--faces` keeps only rows with a face, `--no-faces` only rows checked and found to have none. Purged rows are still returned and are marked as purged.

### `screenshots stats`
Counts by source, by category, by year-month, OCR coverage, face-check coverage and how many have a face, classification coverage, rows pinned with `keep`, near-duplicate pairs, purged originals and bytes reclaimed, disk usage of the library still on disk.

## Tests
Synthetic PNGs generated with Pillow (no fixtures checked in larger than a few KB). Cover: filename date parsing, exFAT-safe naming, exact dedup, near-dup detection, ingest move + catalogue row, FTS search, notes rendering, views rebuild, rules classification. Model tiers are tested with a mocked HTTP client, and face detection with a fake detector, so no test needs Vision. Also: face-count storage, purge eligibility, dry run versus real purge, junk purge, a `keep` pin blocking a purge, re-ingest of a purged hash logged as a duplicate, notes rendering for a purged row, and the migration from schema 2 to 3 on a populated catalogue.

## Photos library (in scope, built)

The iPhone screenshots live in the macOS Photos library, flagged
`ZISDETECTEDSCREENSHOT=1`. With Optimise Mac Storage on, every original is held
only in iCloud, so each one has to be downloaded on the way out.

### `screenshots photos export [--year YYYY] [--limit N] [--dry-run]`
macOS only, needs the `mac` extra. Exports assets osxphotos reports as
screenshots into `inbox/iphone/`, downloading missing originals from iCloud.
Writes a JSON sidecar `<exported name>.json` beside each file:

```json
{
  "photos_uuid": "2CF6584B-E1C2-4913-9B12-F2D712608243",
  "original_filename": "IMG_0013.PNG",
  "date": "2022-05-28T17:16:54+10:00",
  "device": null,
  "source": "photos",
  "exported_at": "2026-09-07T22:10:59"
}
```

`date` is the key the capture-date reader already looks for, so the sidecar
slots into the existing precedence without a special case. Ingest reads
`photos_uuid` onto the row, and deletes the sidecar once the row exists, but
only for a sidecar inside `inbox/`. Re-runs skip anything whose UUID is already
catalogued or already sitting in the inbox, so this is safe to repeat.

### `screenshots photos delete [--year YYYY] [--dry-run] [--yes]`
Deletes assets from Photos, but only for catalogued rows that have a
`photos_uuid`, whose library file is on disk, and whose OCR is done. Nothing
outside the catalogue is ever touched. Listing is the default; `--yes` is
required to act. Deletion goes through PhotoKit in chunks of 200, printing a
count per chunk, and items land in Photos' Recently Deleted rather than going
immediately.

### `screenshots run`
The weekly pipeline, in order: ingest, thumbs, ocr, classify (`--method auto`),
views, notes. Stops at the first failing step with a message naming it.

## Out of scope for now
The review web UI.
