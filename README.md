# vex

A command-line tool that gathers screenshots from everywhere they pile up, files them into
one library, and makes them findable. It hashes and de-duplicates them, reads the text out
of them, sorts them into categories you define, and writes a Markdown note per screenshot
that stays readable long after the original is gone. Once a note and a thumbnail hold
everything a screenshot was for, it can delete the original and give you the disk space
back.

## This is version 2

Version 1 was one script that sent every image to a vision model and wrote the answers to a
JSON file. It proved the idea and then sat unused, because a JSON dump is not somewhere you
go looking for a recipe. Extracting the data was never the hard part. Having somewhere to
put it was.

What changed:

- **OCR first, models second.** Apple Vision reads the text locally and for free. A keyword
  pass settles what it can. Only what is left goes to a model, and only a thumbnail goes
  when the text was too thin to work with. Version 1 paid a vision model for every image.
- **A catalogue instead of a dump.** SQLite with full-text search, so you can type "wonton
  wrappers" and find the screenshot. Version 1 gave you a JSON file to grep.
- **Notes.** One Markdown file per screenshot, with the thumbnail, the caption and the text.
  They open in any notes app and stay readable when the original is gone.
- **A purge policy.** Once a note and a thumbnail hold what a screenshot was for, the
  original can go. It has to prove that first, and anything with a face in it is spared.
- **A pipeline rather than a run.** Every command is idempotent, so it is something you
  point at the same folder every week, not a one-off job.
- **Photos export and delete.** It pulls screenshots out of the macOS Photos library,
  catalogues them, and can then remove them from Photos as a separate, deliberate step.

The original script is still here, on the `v1` branch and the `v1.0.0` tag.

## macOS only

This is a macOS tool. Two of the things it does have no portable equivalent:

- **Reading text** uses Apple Vision. There is a Tesseract fallback in the code, but it is
  untested and unsupported. Treat it as a starting point rather than a feature.
- **Pulling screenshots out of the Photos app**, and deleting them from it afterwards,
  uses PhotoKit through osxphotos and photoscript. There is no equivalent anywhere else.

Everything else, the catalogue, the classifier, the notes and the search, is ordinary
Python. It will run elsewhere. You will just have to bring your own text.

## Quick start

```bash
git clone <this repo> && cd screenshots
uv sync --extra mac

# Point it somewhere. The root is created on first use.
export VEX_ROOT=~/Screenshots

# Drop some screenshots into the inbox, then:
uv run vex ingest ~/Desktop --copy
uv run vex thumbs
uv run vex ocr
uv run vex search "invoice"
```

That much works offline and costs nothing. To sort them into categories you need an
[OpenRouter](https://openrouter.ai) key in a `.env` beside the code:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

```bash
uv run vex classify --dry-run   # tier split and estimated cost, no calls made
uv run vex classify
uv run vex notes
```

`uv run vex run` does ingest, thumbs, ocr, classify, views and notes in one go.

## What stays on your machine

Worth being precise about, because a screenshot library is one of the most revealing things
on a computer.

| Thing | Where it goes |
| --- | --- |
| Your screenshots | Never leave the machine. Nothing uploads an image file. |
| Text recognition | Apple Vision, entirely local. No network call. |
| Face detection | Apple Vision, entirely local. Only a **count** is stored, never a location or an identifier. |
| Classification, text tier | The recognised text, the filename, the dimensions and the source go to the model. |
| Classification, vision tier | A 512 px JPEG thumbnail goes to the model, plus up to 800 characters of the text. |
| Your API key | Read from the environment or a `.env` that is git-ignored. Never printed, never written to the catalogue. |
| Everything else | Stays in the root. The catalogue is a plain SQLite file you can open yourself. |

The rules tier and every `--dry-run` work with no key and no network at all. If you only
ever run `classify --method rules`, nothing leaves the machine.

## The root

Everything lives under one directory, from `--root`, else `VEX_ROOT`, else
`~/Screenshots`.

```
inbox/
  mac/              drop zone for macOS screenshots
  iphone/           drop zone for Photos exports
  other/
library/
  YYYY/MM/          canonical files: <YYYYMMDD-HHMMSS>_<source>_<hash8>.<ext>
duplicates/         exact duplicates moved here on ingest, never deleted
skipped/            inbox files nothing downstream can read, moved aside not deleted
thumbs/             <hash>.jpg, max side 512 px, quality 80
views/              generated category folders, copies rather than links
notes/              generated Markdown, one file per screenshot
logs/               backlog run logs
bakeoff/            model comparison dumps
catalog.sqlite      the catalogue
taxonomy.toml       optional: the categories this library is filed under
screenshots.toml    optional: model ids, concurrency, the macOS drop zone
```

Every path stored in the catalogue is relative to the root and uses forward slashes, so the
whole root can be moved to an external drive. Filenames are kept safe for exFAT and Windows,
and nothing is linked, only copied.

## Commands

Every command takes `--root`. All of them are idempotent and safe to re-run.

### ingest

```bash
uv run vex ingest                     # everything under inbox/
uv run vex ingest ~/Desktop --copy --dry-run
uv run vex ingest ~/Downloads --recursive
```

| Flag | What it does |
| --- | --- |
| `--source auto\|mac\|iphone\|other` | Where a file came from. `auto` guesses from the filename and any sidecar. |
| `--copy` | Copy instead of moving, leaving the originals where they are. |
| `--dry-run` | Report what would happen and change nothing. |
| `--recursive` | Recurse into the given directories. |

Writes: files into `library/YYYY/MM/`, exact duplicates into `duplicates/`, unreadable inbox
files into `skipped/`, and one row per screenshot in the catalogue.

Files are identified by the SHA-256 of their bytes, so the same screenshot arriving twice
under two names is caught. Perceptually similar images are linked through `near_duplicate_of`
rather than removed. Capture dates are read in order from EXIF `DateTimeOriginal`, XMP or PNG
text chunks, the filename, a JSON sidecar, then the file's own timestamps, and the winner is
recorded so you can tell a good date from a guess.

On macOS, iCloud-evicted placeholders are downloaded with `brctl download` and polled for up
to 120 seconds before being skipped.

An inbox file nothing can read is moved to `skipped/` rather than left in place, because left
in place it would be retried forever. This only happens inside the root's own inbox. A file
elsewhere that you pointed ingest at stays exactly where you left it.

### thumbs

```bash
uv run vex thumbs
uv run vex thumbs --force
```

| Flag | What it does |
| --- | --- |
| `--force` | Regenerate thumbnails that already exist. |

Writes: `thumbs/<hash>.jpg`, longest side 512 px, quality 80.

### ocr

```bash
uv run vex ocr
uv run vex ocr --engine vision --limit 100
uv run vex ocr --force
```

| Flag | What it does |
| --- | --- |
| `--engine auto\|vision\|tesseract` | Which engine. `auto` is Vision on macOS, else Tesseract. |
| `--limit N` | Only process this many rows. |
| `--force` | Re-run on rows that already have text. |

Writes: the `ocr_text`, `ocr_engine` and `ocr_at` columns, and the full-text search index.

Apple Vision runs at the accurate recognition level with language correction on. An empty
result is stored as an empty string rather than null, so it is not retried on every run.
Images longer than 2500 px are decoded down to that bound: Vision does not need full
resolution to read a screenshot, and decoding a 16 MB PNG at native size is what drives peak
memory on a long pass. Each row is committed as it is read, so an interrupted run keeps what
it had already done.

### faces

```bash
uv run vex faces
uv run vex faces --limit 200 --force
```

| Flag | What it does |
| --- | --- |
| `--limit N` | Only process this many rows. |
| `--force` | Re-check rows that already have a count. |

Writes: the `faces` count and `faces_at`.

macOS only. Detection runs over the 512 px thumbnail, not the original. **Only the number of
faces is stored.** No location, no landmark, no identifier, nothing that would amount to
recognising a person. The number exists so `purge` can leave anything with a person in it
alone. A null `faces` means never checked, which is not the same as none, and purge treats it
as a reason to keep the original.

### classify

```bash
uv run vex classify --dry-run
uv run vex classify
uv run vex classify --method rules              # free, no network
uv run vex classify --method vision --category other --force
```

| Flag | What it does |
| --- | --- |
| `--method auto\|rules\|text\|vision` | Which tiers to use. |
| `--limit N` | Only process this many rows. |
| `--force` | Re-classify rows that already have a category. |
| `--dry-run` | Report the tier split and a live-priced estimate. Makes no calls. |
| `--model ID` | Override the model id for both model tiers, this run only. |
| `--concurrency N` | Worker threads, 1 to 8. Default 6. |
| `--category NAME` | Only rows already filed under this category. Repeatable. |

Writes: `category`, `tags`, `caption`, `structured`, `classify_method`,
`classify_confidence`, `is_junk`, and one row per call in `classify_cache`.

Three tiers, cheapest first:

1. **rules** - keyword matching over the recognised text and the original filename. Free, no
   network. Deliberately conservative: it declines rather than guess.
2. **text** - the text, dimensions and source sent to a cheap text model.
3. **vision** - the 512 px thumbnail sent to a cheap vision model.

`auto` runs the rules tier, then the text tier when there are at least 40 characters of text,
then escalates to vision when the text is thin or the text tier reports confidence below 0.7.

**The rules tier is a hint, not a verdict.** Keywords are good at spotting evidence and bad at
reading it in context: a post about prompting that happens to use the word "recipe" was being
filed as a recipe with 0.8 confidence. So in `auto` every row still goes to a model, and the
keyword guess rides along in the prompt as one weak signal the model is told to ignore when
the screenshot says otherwise. The single exception is a confident junk verdict. A lock screen
is unambiguous and there is nothing for a model to add.

Every raw response is cached against the hash, model and tier, so a re-run never pays for the
same call twice. A cached reply is only reused if it was produced under the current response
schema, so a prompt change is picked up rather than papered over.

Defaults are `google/gemini-2.5-flash-lite` for both model tiers. Override with `--model`, or
`VEX_TEXT_MODEL` / `VEX_VISION_MODEL`, or a `vex.toml` in the root:

```toml
# <root>/vex.toml
[classify]
text_model = "google/gemini-2.5-flash-lite"
vision_model = "google/gemini-3.1-flash-lite"
concurrency = 6
```

`docs/model-choice.md` has the shortlist, the prices per million tokens and the reasoning.

### bakeoff

```bash
uv run vex bakeoff --sample 20
uv run vex bakeoff --sample 30 --models google/gemini-2.5-flash-lite,google/gemma-3-12b-it
```

| Flag | What it does |
| --- | --- |
| `--sample N` | How many screenshots to test each model on. Default 20. |
| `--models A,B,C` | Comma-separated model ids to compare. |
| `--tier both\|text\|vision` | Which tiers to run. |
| `--seed N` | Seed the sample so a run can be repeated. |
| `--concurrency N` | Worker threads, 1 to 8. |

Writes: `bakeoff/bakeoff-<timestamp>.json`, and every response into `classify_cache`, so a
later `classify --model <winner>` reuses calls the bakeoff already paid for.

Agreement is measured against the majority category across the models on each screenshot, not
a hand-labelled truth set. It says which model is the outlier, not which one is right.

### views

```bash
uv run vex views
uv run vex views --category recipe
```

| Flag | What it does |
| --- | --- |
| `--category NAME` | Only rebuild this category. |

Writes: `views/<category>/`, as copies of the library files. Copies rather than links,
because the drive may be exFAT. Junk rows and purged rows are left out.

### notes

```bash
uv run vex notes
uv run vex notes --force
uv run vex notes --category recipe
```

| Flag | What it does |
| --- | --- |
| `--category NAME` | Only write notes for this category, and sweep nothing. |
| `--force` | Rewrite notes that already exist. |

Writes: `notes/<category>/<stamp>_<hash8>.md`, one per non-junk row, with the thumbnail, the
caption, any structured payload and the recognised text. A purged row still gets one. The
thumbnail is what is left of it.

A full run also deletes notes no row claims any more: a re-classified row leaves its old note
orphaned under the previous category, and a row that became junk leaves one behind entirely. A
run scoped with `--category` sweeps nothing outside that category, because asking about one
category should not tidy the rest. `._*` AppleDouble companions, which macOS writes beside
every file on an exFAT drive, are never read, counted or deleted.

### search

```bash
uv run vex search "invoice"
uv run vex search "recipe OR ingredients" --category recipe --limit 50
uv run vex search "invoice" --json --no-faces
```

| Flag | What it does |
| --- | --- |
| `--category NAME` | Restrict to one category. |
| `--faces` / `--no-faces` | Only rows with a face, or only rows checked and found to have none. |
| `--limit N` | Maximum results. Default 20. |
| `--json` | Print results as JSON instead of a table. |

Writes: nothing. SQLite FTS5 with BM25 ranking. Rows whose original has been purged are still
returned, marked `(purged)` in the table or `"purged": true` in the JSON.

### stats

```bash
uv run vex stats
```

Writes: nothing. Counts by source, category and month, coverage for OCR, classification and
face checks, how much has been purged and how much that reclaimed.

### run

```bash
uv run vex run
```

Writes: whatever its steps write. Ingest, thumbs, ocr, classify, views and notes, in that
order, stopping at the first step that fails so a broken step never leaves the later ones
working from half-finished input.

## The taxonomy

Categories are not baked into the code. They come from `taxonomy.toml` in the root. Without
one, a generic built-in default applies, which is `vex/default_taxonomy.toml` in this
repo: recipe, receipt, code, article, social-post, message, map, shopping, event, document,
meme, screenshot-of-ui, junk and other.

It is meant to be replaced. Copy the built-in file into your root and write the categories
your own screenshots actually fall into. Mine look nothing like the default.

```toml
# <root>/taxonomy.toml
context = "You file screenshots for the person who took them."

[[category]]
name = "recipe"
description = "A food recipe: ingredient list, method steps, a recipe card or a cooking post."
structured = "recipe"
purgeable = false
strong = ["ingredients", "preheat the oven", "recipe"]
weak = ["tbsp", "tsp", "serves", "bake", "oven", "simmer"]
tags = ["recipe", "cooking"]

[[category]]
name = "article"
description = "A written piece read on the web: a news story, a blog post, a newsletter."
structured = "use-note"
purgeable = true
strong = ["min read", "continue reading"]
weak = ["published", "newsletter", "read more"]
tags = ["article", "reading"]

[[category]]
name = "junk"
description = "Blank captures, accidental screenshots, lock screens, loading placeholders."
strong = ["swipe up to open", "enter passcode", "face id"]
weak = ["loading", "please wait", "low battery"]
tags = ["junk"]

[[category]]
name = "other"
description = "Something real and worth keeping that does not fit any other category."
```

| Key | What it does |
| --- | --- |
| `name` | Lowercase slug. It becomes a directory name under `notes/` and `views/`. |
| `description` | One or two sentences telling the model what belongs here. |
| `structured` | Optional, `recipe` or `use-note`. The extra field the model fills in. |
| `purgeable` | Whether `purge` may delete an original once the note and thumbnail exist. |
| `strong` | Phrases close to decisive for the rules tier, worth 3 each. |
| `weak` | Phrases that only count together, worth 1 each. |
| `tags` | Tags added to anything the rules tier files here. |

`context` is the opening line of the system prompt, and is where you say whose library this
is. Order matters only in that it is the order categories are offered to the model in. `junk`
and `other` are required: every capture has to have somewhere to land.

Two structured shapes are built in. `recipe` asks the model to transcribe a title, ingredient
list, method, serves and source, and the note renders them as headings. `use-note` asks for
one sentence on how the screenshot could be reused, which is what makes a folder of saved
posts worth more than a folder of images.

Whitespace inside a keyword is significant. `"def "` with the trailing space is what stops it
matching "default", and stripping it widens the rule by a factor of twenty.

A rules verdict needs a score of at least 3 and a margin of at least 2 over the runner-up, or
it declines and lets a model decide.

A broken `taxonomy.toml` is reported as one line and a non-zero exit, naming the category and
the key. Changing the taxonomy does not re-file anything already catalogued: rows keep the
category they were given. Run `classify --force` to move them, then `notes` and `views` to
sweep the folders the old names left behind.

## Purge, keep and unkeep

```bash
uv run vex purge                            # dry run, the default
uv run vex purge --category junk --yes
uv run vex purge --min-text 120 --yes
uv run vex keep 4f2a91c3                    # 8-character prefixes are fine when unique
uv run vex unkeep 4f2a91c3
```

| Flag | What it does |
| --- | --- |
| `--dry-run` | Report what would go and delete nothing. The default. |
| `--yes` | Actually delete the originals. |
| `--category NAME` | Only purge these categories. Repeatable. Narrows, never widens. |
| `--min-text N` | Characters of recognised text a row needs to qualify. Default 80. |

Writes: deletes library originals, and sets `purged_at` on their rows.

Purging deletes **only the original file**. The thumbnail, the note and the catalogue row
always stay, so a purged screenshot is still searchable, still renders in its note, and still
blocks a re-ingest of the same bytes. The row keeps its `path` as the historical location.

A row is only eligible when all of this holds:

- its category is marked `purgeable = true` in the taxonomy, or it is marked junk, which is
  purgeable whatever its text says
- `faces` is 0. Null means never checked, so the original stays
- it has at least `--min-text` characters of recognised text
- its thumbnail is on disk. Purging without one would lose the image entirely
- it is not pinned with `keep`

Everything else is protected, as is anything with a person in it. Naming a protected category
with `--category` is an error rather than a silent no-op.

`keep` pins a row so purge never touches it, whatever the taxonomy says. `unkeep` releases it.
Both take one or more hashes, and an 8-character prefix is fine as long as it is unique.

The dry run prints a table of files and bytes by category and a tally of why the rest were
held back, which is usually the more interesting half.

## The Photos loop

If your phone screenshots live in the macOS Photos library rather than on disk, these pull
them out. macOS only, and they need the `mac` extra.

```bash
uv run vex photos export --year 2022 --limit 40 --dry-run
uv run vex photos export --year 2022 --limit 40
uv run vex photos delete --year 2022          # lists, changes nothing
uv run vex photos delete --year 2022 --yes
```

`photos export` takes `--year`, `--limit`, `--dry-run` and `--root`. It writes each file into
`inbox/iphone/` with a JSON sidecar carrying the asset UUID, the original filename and the
capture date. Ingest reads the UUID onto the row and then removes the sidecar. Which assets
count as screenshots is Photos' own judgement, read straight out of its database, read-only.
Re-running skips anything already exported or catalogued.

`photos delete` takes `--year`, `--dry-run`, `--yes`, `--chunk` and `--root`. Listing is the
default and `--yes` is required to act. It only ever considers rows this catalogue knows
about, that carry a `photos_uuid`, whose text has been read, and where there is still
something to go back to: the library original, or for a purged row, its thumbnail. A row with
neither is refused whatever else is true. Items go to Photos' Recently Deleted and sit there
for 30 days. macOS shows one confirmation dialogue per PhotoKit request, so by default the
whole run is a single request and a single dialogue; `--chunk N` splits it if you ever need to.

### The overnight backlog

```bash
uv run vex photos backlog --dry-run
uv run vex photos backlog
uv run vex photos backlog --years 2023,2024 --batch 150 --max-batches 4
```

| Flag | What it does |
| --- | --- |
| `--years A,B,C` | Years to work through, oldest first. |
| `--batch N` | Items to export per batch. Default 300. |
| `--min-free-gb F` | Stop cleanly when free space falls below this. Default 10. |
| `--max-batches N` | Stop after this many batches. |
| `--dry-run` | Print the plan and change nothing. |

Writes: everything its steps write, plus a log at `logs/backlog-<YYYYMMDD-HHMM>.log`.

Each time round the loop it exports a batch for the current year, runs ingest, thumbs, ocr,
classify, faces, notes and `purge --yes` over whatever landed, then goes again. Views is left
out on purpose: it copies the very originals the purge is about to delete. When a year's
export finds nothing new it moves to the next.

**Purging each batch as it goes is what makes this work on a nearly full disk: the space a
batch reclaims pays for the next one.** This command never deletes anything from Photos. That
stays a separate, deliberate step.

`Ctrl-C` or `SIGTERM` finishes the step that is running, writes the summary and exits. A second
`Ctrl-C` gets you out immediately. A batch that fails is retried once after 60 seconds; if it
fails again the year is skipped and the loop carries on. When more than half a batch fails,
that is Photos refusing the pace rather than bad items, so the loop waits five minutes,
doubling, capped at thirty, and halves what it asks for down to a floor of 50, winning it back
gradually once batches come through clean.

#### Why it stops when the disk gets low

Photos will not fetch an iCloud original onto a nearly full volume. It refuses in milliseconds
with `CloudPhotoLibraryErrorDomain` code 1005, "Disk space is very low", so a whole batch fails
inside the same second. No retry, no wait and no smaller batch makes any difference, because
nothing was ever attempted.

This was diagnosed after an overnight run that exported cleanly for an hour and then failed 96
to 100 per cent of every batch until morning: 404 refusals in one hour, every one of them code
1005, at 4.1 GB free of 228 GB. The 2195 screenshots that had come down and the 2384 that would
not were identical in every property osxphotos reports. The gate sat near 5 GB free, and a
batch would briefly succeed right after a purge freed a little space.

The volume that matters is the one holding the **Photos library**, not the one holding this
library. Once the root moves to an external drive those come apart.

| Volume | Floor | When it applies |
| --- | --- | --- |
| The one holding the Photos library | 6 GB | Always. This is the one that stops downloads. |
| The one holding this library root | 10 GB | Skipped when it is a separate volume with more than 50 GB free. |

Both are measured and reported in every batch line, and whichever is lower decides.
`photos export` makes the same check before it loads the Photos database, so a manual run gets
the message rather than a silent wall of failures. The fix is on the Mac rather than in this
tool: free up real disk space.

## Development

```bash
uv sync --extra mac
uv run ruff check
uv run ruff format --check
uv run pytest
```

Tests use synthetic images generated with Pillow. No real screenshots are checked in.
`SPEC.md` is the build contract and `AGENTS.md` is the orientation for coding agents.

## A second version of vex

This is the second time I have built this shape: a local SQLite catalogue over a pile of
files, filled in by small idempotent passes that each do one thing and can be re-run forever
without harm. The first was [vex](https://github.com/teejayen/vex).

Most of what is here is what I would do differently a second time. Classification is tiered,
so the cheap pass handles what it can and the expensive one only sees what it has to. Every
model response is cached against the prompt schema, so changing the prompt invalidates the
cache instead of quietly reusing it. And deleting anything has to prove first that its value
is already held somewhere else.

## Licence

MIT. See `LICENSE`.
