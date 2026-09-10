"""The unattended loop that drains the Photos backlog, a batch at a time.

This is the overnight job. It exports a batch of screenshots out of Photos, runs
the whole pipeline over them, purges what the notes have already captured, and
goes round again until the years are drained, the disk gets tight, or someone
asks it to stop. Nothing here ever deletes anything from Photos.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import FrameType

import typer

from vex import db, photos
from vex.commands import classify as classify_command
from vex.commands import faces as faces_command
from vex.commands import ingest as ingest_command
from vex.commands import notes as notes_command
from vex.commands import ocr as ocr_command
from vex.commands import purge as purge_command
from vex.commands import thumbs as thumbs_command
from vex.console import console

DEFAULT_YEARS: tuple[int, ...] = (2023, 2024, 2025, 2026)
DEFAULT_BATCH = 300

#: Photos stops serving iCloud originals well before the disk is actually full,
#: so the floor lives with the rest of that knowledge in the photos module.
DEFAULT_MIN_FREE_GB = photos.DEFAULT_MIN_FREE_GB

#: How long to wait before the one retry a failed batch gets.
RETRY_DELAY_SECONDS = 60

BYTES_PER_GB = 1024**3
BYTES_PER_MIB = 1024 * 1024

#: A batch of iCloud downloads can be genuinely slow, so the export subprocess is
#: given a generous allowance per item before it is treated as hung.
EXPORT_SECONDS_PER_ITEM = 60
EXPORT_MINIMUM_SECONDS = 1800

#: A year whose exports keep failing must not be hammered forever, so this many
#: batches in a row that write nothing gives up on the year rather than spinning.
#: Only once the backoff has run all the way up, so throttling alone never does it.
MAX_BARREN_BATCHES = 3

#: Above this share of failures, Photos is not refusing these particular items, it
#: is refusing the pace. Back off rather than retrying harder.
HIGH_FAILURE_RATIO = 0.5
#: Below this share, whatever it was has passed, so wind back up.
CLEAN_FAILURE_RATIO = 0.2

BACKOFF_START_SECONDS = 300
BACKOFF_CAP_SECONDS = 1800
BACKOFF_FACTOR = 2

#: However hard it is being throttled, a batch this small is still worth running.
MIN_BATCH = 50
#: How much of the batch size a clean batch wins back, as a fraction of the current
#: size. Gradual on the way up, halved on the way down.
BATCH_RECOVERY = 0.5

#: Outcomes a batch can end with.
OUTCOME_OK = "ok"
OUTCOME_DRAINED = "drained"
OUTCOME_SKIPPED = "skipped"
OUTCOME_BARREN = "nothing written"


class ExportFailedError(RuntimeError):
    """Raised when the export subprocess did not come back cleanly."""


class YearListError(ValueError):
    """Raised when --years cannot be read as a list of years."""


def parse_years(text: str) -> list[int]:
    """Read '2023,2024' into [2023, 2024], keeping the order given."""
    years: list[int] = []
    for part in text.replace(" ", ",").split(","):
        if not part:
            continue
        try:
            year = int(part)
        except ValueError as exc:
            raise YearListError(f"'{part}' is not a year.") from exc
        if year not in years:
            years.append(year)
    if not years:
        raise YearListError("No years given.")
    return years


# ---------------------------------------------------------------------------
# Counting what a batch actually did
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The catalogue totals a batch is measured against."""

    rows: int = 0
    read: int = 0
    classified: int = 0
    purged: int = 0
    purged_bytes: int = 0
    cost_usd: float = 0.0
    discarded: int = 0


def take_snapshot(root: Path) -> Snapshot:
    """Read the catalogue totals. The difference across a batch is what it did.

    Counting from the catalogue rather than from what each command printed keeps
    the loop honest: it reports what actually landed, not what a step claimed.
    """
    conn = db.connect(root)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0]
        read = conn.execute(
            "SELECT COUNT(*) FROM screenshots WHERE ocr_text IS NOT NULL"
        ).fetchone()[0]
        classified = conn.execute(
            "SELECT COUNT(*) FROM screenshots WHERE category IS NOT NULL"
        ).fetchone()[0]
        purged, purged_bytes = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM screenshots WHERE purged_at IS NOT NULL"
        ).fetchone()
        cost = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM classify_cache").fetchone()[0]
        # Photos re-exports of rows already held, deleted rather than parked. This
        # is what a year that was already half-ingested costs to walk through again.
        discarded = conn.execute(
            "SELECT COUNT(*) FROM ingest_log WHERE action = 'duplicate' AND detail LIKE ?",
            (f"%{ingest_command.DISCARDED_DETAIL}%",),
        ).fetchone()[0]
    finally:
        conn.close()
    return Snapshot(rows, read, classified, purged, purged_bytes, float(cost), discarded)


@dataclass
class BatchResult:
    """What one time round the loop exported, processed and reclaimed."""

    number: int
    year: int
    outcome: str = OUTCOME_OK
    exported: int = 0
    added: int = 0
    read: int = 0
    classified: int = 0
    cost_usd: float = 0.0
    purged_files: int = 0
    purged_bytes: int = 0
    discarded: int = 0
    failed: int = 0
    outstanding: int = 0
    batch_size: int = 0
    waited: int = 0
    free_gb: float = 0.0
    note: str = ""

    def measure(self, before: Snapshot, after: Snapshot) -> None:
        """Fill in the counts from the catalogue totals either side of the batch."""
        self.added = after.rows - before.rows
        self.read = after.read - before.read
        self.classified = after.classified - before.classified
        self.purged_files = after.purged - before.purged
        self.purged_bytes = after.purged_bytes - before.purged_bytes
        self.cost_usd = round(after.cost_usd - before.cost_usd, 4)
        self.discarded = after.discarded - before.discarded


# ---------------------------------------------------------------------------
# Logging to the run log and the terminal at once
# ---------------------------------------------------------------------------


def log_path(root: Path, when: datetime | None = None) -> Path:
    """Where this run's log goes."""
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M")  # noqa: DTZ005
    return Path(root) / "logs" / f"backlog-{stamp}.log"


class RunLog:
    """Every line goes to the log file with a timestamp; most also go to stdout."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, text: str) -> None:
        stamp = datetime.now().replace(microsecond=0).isoformat()  # noqa: DTZ005
        with self.path.open("a", encoding="utf-8") as handle:
            for line in text.splitlines() or [""]:
                handle.write(f"{stamp}  {line}\n")

    def say(self, text: str) -> None:
        """A line for both the log and the terminal."""
        self._write(text)
        console.print(text)

    def problem(self, text: str) -> None:
        """Something went wrong, but the loop carries on."""
        self._write(f"WARNING  {text}")
        console.print(f"[yellow]{text}[/yellow]")

    def detail(self, text: str) -> None:
        """Verbose output that belongs in the log but would drown the terminal."""
        if text.strip():
            self._write(text.rstrip())

    def rule(self, text: str) -> None:
        """A visible divider between batches."""
        self._write(f"===== {text} =====")
        console.rule(f"[bold]{text}")


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

PipelineStep = Callable[[Path], None]
ExportStep = Callable[[Path, int, int, "RunLog | None"], "ExportOutcome"]


@dataclass(frozen=True, slots=True)
class ExportOutcome:
    """What an export step wrote, and what Photos still has waiting."""

    exported: int = 0
    new: int = 0
    failed: int = 0
    detected: int = 0
    #: False when the export did not say how much is left, so nothing can be concluded.
    reported: bool = True

    @property
    def drained(self) -> bool:
        """True only when the export said Photos has nothing left for the year.

        Deliberately not ``exported == 0``: a batch whose downloads all failed
        writes nothing either, and rotating away on that abandons the year. An
        export that reported nothing at all is never taken as drained; the barren
        counter is what stops the loop dwelling on it.
        """
        return self.reported and self.new == 0


def inbox_items(inbox: Path) -> set[Path]:
    """Files sitting in the iPhone inbox, ignoring the JSON sidecars."""
    if not inbox.is_dir():
        return set()
    return {p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() != ".json"}


def export_batch(root: Path, year: int, limit: int, log: RunLog | None = None) -> ExportOutcome:
    """Export up to ``limit`` screenshots for one year, and report what happened.

    This is the one step that runs as a subprocess. osxphotos loads the whole
    Photos database, several hundred megabytes resident, on every export call,
    and in-process that memory is not reliably handed back between batches. Over
    an overnight run it accumulates. A subprocess gives it back to the operating
    system every time it exits.

    Its whole output goes to the run log. Without that, an export that quietly
    failed on most of its items looks exactly like one that had little to do.
    """
    inbox = Path(root) / "inbox" / "iphone"
    before = inbox_items(inbox)
    command = [
        sys.executable,
        "-m",
        "vex",
        "photos",
        "export",
        "--year",
        str(year),
        "--limit",
        str(limit),
        "--root",
        str(root),
    ]
    timeout = max(EXPORT_MINIMUM_SECONDS, EXPORT_SECONDS_PER_ITEM * limit)
    completed = subprocess.run(  # noqa: S603
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    if log is not None:
        log.detail(completed.stdout)
        log.detail(completed.stderr)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise ExportFailedError(f"photos export exited {completed.returncode}: {tail}")

    landed = len(inbox_items(inbox) - before)
    reported = photos.parse_result(completed.stdout)
    if reported is None:
        # An older export that does not print the line. Nothing can be concluded
        # about what is left, so the year is never declared drained on this alone.
        return ExportOutcome(exported=landed, new=landed, failed=0, reported=False)
    return ExportOutcome(
        exported=landed, new=reported.new, failed=reported.failed, detected=reported.detected
    )


def pipeline_steps() -> list[tuple[str, PipelineStep]]:
    """The steps run over each exported batch, in order.

    Faces has to come before purge, which will not touch a row whose face count
    is unknown, and notes before it too, so the note exists before the original
    goes. Views is left out on purpose: it copies the very originals the purge is
    about to delete.
    """
    return [
        ("ingest", lambda root: ingest_command.command(root=root)),
        ("thumbs", lambda root: thumbs_command.command(root=root)),
        ("ocr", lambda root: ocr_command.command(root=root)),
        ("classify", lambda root: classify_command.command(method="auto", root=root)),
        ("faces", lambda root: faces_command.command(root=root)),
        ("notes", lambda root: notes_command.command(root=root)),
        ("purge", lambda root: purge_command.command(yes=True, root=root)),
    ]


# ---------------------------------------------------------------------------
# Pacing against a throttling Photos daemon
# ---------------------------------------------------------------------------


@dataclass
class Pacer:
    """Keeps the loop's pace in step with what Photos is actually willing to give.

    An hour of successful downloads followed by 98% failures is not a library
    full of bad items, it is a daemon that has had enough. Retrying harder makes
    it worse and burns the barren counter, so the loop waits instead, and asks
    for less when it comes back.
    """

    full_batch: int
    batch_size: int = 0
    wait_seconds: int = 0

    def __post_init__(self) -> None:
        if not self.batch_size:
            self.batch_size = self.full_batch

    @property
    def at_cap(self) -> bool:
        """True once the backoff has escalated as far as it goes."""
        return self.wait_seconds >= BACKOFF_CAP_SECONDS

    def observe(self, exported: int, failed: int) -> str | None:
        """Take in a batch's outcome. Returns the reason to wait, or None to carry on."""
        attempted = exported + failed
        if not attempted:
            return None
        ratio = failed / attempted
        if ratio > HIGH_FAILURE_RATIO:
            self.wait_seconds = (
                BACKOFF_START_SECONDS
                if not self.wait_seconds
                else min(self.wait_seconds * BACKOFF_FACTOR, BACKOFF_CAP_SECONDS)
            )
            self.batch_size = max(MIN_BATCH, self.batch_size // 2)
            return (
                f"{failed} of {attempted} exports failed ({ratio:.0%}), which reads as "
                f"throttling rather than bad items"
            )
        if ratio < CLEAN_FAILURE_RATIO:
            self.wait_seconds = 0
            grown = self.batch_size + max(1, int(self.batch_size * BATCH_RECOVERY))
            self.batch_size = min(self.full_batch, grown)
        return None


# ---------------------------------------------------------------------------
# Stopping politely
# ---------------------------------------------------------------------------


@dataclass
class StopFlag:
    """Set when the run has been asked to stop at the next step boundary."""

    requested: bool = False
    reason: str = ""

    def request(self, reason: str) -> None:
        self.requested = True
        self.reason = reason


@contextmanager
def stop_on_signals(flag: StopFlag) -> Iterator[StopFlag]:
    """Turn SIGINT and SIGTERM into a request to finish the current step and stop.

    The original handler is put back after the first signal, so a second Ctrl-C
    behaves normally and gets you out immediately.
    """
    previous: dict[int, object] = {}

    def handle(number: int, _frame: FrameType | None) -> None:
        signal.signal(number, previous[number])  # a second signal is not ours
        flag.request(signal.Signals(number).name)
        console.print(
            f"\n[yellow]{signal.Signals(number).name} received. "
            "Finishing the current step, then writing the summary.[/yellow]"
        )

    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.getsignal(number)
        signal.signal(number, handle)
    try:
        yield flag
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Summary:
    """Everything the run did, for the table at the end."""

    started_at: str
    finished_at: str
    reason: str
    batches: list[BatchResult]
    log: Path

    def total(self, attribute: str) -> float:
        return sum(getattr(batch, attribute) for batch in self.batches)


def free_bytes(path: Path) -> int:
    """Free space on the volume holding a path."""
    return shutil.disk_usage(path).free


@dataclass
class BacklogRun:
    """One overnight run. Every moving part is injectable so the tests can drive it."""

    root: Path
    log: RunLog
    years: list[int] = field(default_factory=lambda: list(DEFAULT_YEARS))
    batch: int = DEFAULT_BATCH
    min_free_gb: float = DEFAULT_MIN_FREE_GB
    max_batches: int | None = None
    export: ExportStep = export_batch
    steps: list[tuple[str, PipelineStep]] = field(default_factory=pipeline_steps)
    sleep: Callable[[float], None] = time.sleep
    free: Callable[[Path], int] = free_bytes
    stop: StopFlag = field(default_factory=StopFlag)
    #: Overridden only by the tests; in a real run osxphotos says where it is.
    photos_library: Path | None = None
    pacer: Pacer | None = None

    def __post_init__(self) -> None:
        if self.pacer is None:
            self.pacer = Pacer(full_batch=self.batch)

    def free_gb(self) -> float:
        return self.free(self.root) / BYTES_PER_GB

    def space(self) -> photos.DiskCheck:
        """Free space on the library volume and on the Photos volume.

        They are one disk today and will not be once the root moves to an external
        drive, at which point only the Photos side decides whether anything
        downloads at all.
        """
        return photos.build_check(
            self.root,
            photos_library=self.photos_library,
            free_bytes=self.free,
            root_floor=self.min_free_gb,
        )

    def run(self) -> Summary:
        """Work through the years in batches until something says stop."""
        started_at = datetime.now().replace(microsecond=0).isoformat()  # noqa: DTZ005
        self.log.say(
            f"Backlog run over {', '.join(str(y) for y in self.years)}, "
            f"batches of {self.batch}, stopping under {self.min_free_gb:.1f} GB free on the "
            f"library volume or {photos.PHOTOS_MIN_FREE_GB:.1f} GB on the Photos volume."
        )
        batches: list[BatchResult] = []
        index = 0
        barren = 0
        reason = "every year is drained"

        while index < len(self.years):
            if self.stop.requested:
                reason = f"stopped on {self.stop.reason}"
                break
            if self.max_batches is not None and len(batches) >= self.max_batches:
                reason = f"reached the {self.max_batches} batch limit"
                break
            check = self.space()
            if check.blocked:
                volume = "the Photos volume" if check.blocked == "photos" else "the library volume"
                reason = f"{volume} is too full: {check.summary()}"
                self.log.problem(f"Stopping. {photos.low_disk_message(check)}")
                break

            result = self.batch_once(len(batches) + 1, self.years[index])
            batches.append(result)

            if result.outcome == OUTCOME_BARREN:
                barren += 1
                # Only give up once the backoff has run all the way up. Before
                # that, a run of empty batches is throttling, not a spent year.
                if barren >= MAX_BARREN_BATCHES and self.pacer.at_cap:
                    self.log.problem(
                        f"{barren} batches in a row wrote nothing for {self.years[index]} "
                        f"at the longest wait, with {result.outstanding} still outstanding. "
                        "Moving on."
                    )
                    result.note += f", gave up after {barren} barren batches"
                    index += 1
                    barren = 0
            else:
                barren = 0
                if result.outcome in (OUTCOME_DRAINED, OUTCOME_SKIPPED):
                    index += 1

            result.waited = self.pace(result)

        if self.stop.requested and not reason.startswith("stopped"):
            reason = f"stopped on {self.stop.reason}"

        finished_at = datetime.now().replace(microsecond=0).isoformat()  # noqa: DTZ005
        self.log.say(f"Finished: {reason}.")
        return Summary(started_at, finished_at, reason, batches, self.log.path)

    def pace(self, result: BatchResult) -> int:
        """Wait out a throttled batch before starting the next. Returns seconds waited."""
        reason = self.pacer.observe(result.exported, result.failed)
        if reason is None or self.stop.requested:
            return 0

        check = self.space()
        if check.near_floor:
            # Not throttling. Photos is refusing because a volume is nearly full,
            # and no amount of waiting or shrinking the batch will change that.
            self.log.problem(photos.low_disk_message(check))
            self.stop.request("low disk space")
            result.note = "stopped, disk too full for Photos to serve originals"
            return 0

        wait = self.pacer.wait_seconds
        self.log.problem(
            f"{reason}. Waiting {wait // 60} minutes before the next batch, "
            f"and asking for {self.pacer.batch_size} items rather than {result.batch_size}."
        )
        self.sleep(wait)
        return wait

    def batch_once(self, number: int, year: int) -> BatchResult:
        """One batch, with the single retry a transient failure is allowed."""
        self.log.rule(f"batch {number}, {year}")
        before = take_snapshot(self.root)
        result = BatchResult(number=number, year=year, batch_size=self.pacer.batch_size)

        for attempt in (1, 2):
            try:
                outcome = self.attempt(year)
            except Exception as exc:  # noqa: BLE001 - one bad batch must not end the night
                message = f"{type(exc).__name__}: {exc}"
                if attempt == 1 and not self.stop.requested:
                    self.log.problem(
                        f"Batch {number} ({year}) failed: {message}. "
                        f"Retrying once in {RETRY_DELAY_SECONDS}s."
                    )
                    self.sleep(RETRY_DELAY_SECONDS)
                    continue
                self.log.problem(
                    f"Batch {number} ({year}) failed again: {message}. Skipping {year}."
                )
                result.outcome = OUTCOME_SKIPPED
                result.note = message
            else:
                result.exported = outcome.exported
                result.failed = outcome.failed
                result.outstanding = outcome.new
                if outcome.drained:
                    result.outcome = OUTCOME_DRAINED
                    result.note = "nothing new left in Photos"
                elif outcome.exported == 0:
                    # Items are still waiting, so this is failure, not completion.
                    result.outcome = OUTCOME_BARREN
                    result.note = f"wrote nothing, {outcome.new} still waiting"
                if self.stop.requested:
                    result.note = f"stopped on {self.stop.reason}"
            break

        after = take_snapshot(self.root)
        result.measure(before, after)
        space = self.space()
        result.free_gb = space.root_gb
        result.photos_free_gb = space.photos_gb
        self.log.say(
            f"Batch {number} ({year}): exported {result.exported}, added {result.added}, "
            f"read {result.read}, classified {result.classified}, "
            f"cost ${result.cost_usd:.4f}, purged {result.purged_files} "
            f"({result.purged_bytes / BYTES_PER_MIB:.1f} MiB), "
            f"discarded {result.discarded} re-exports, "
            f"{result.failed} export failures, batch size {result.batch_size}, "
            f"{result.free_gb:.1f} GB free on the library volume, "
            f"{result.photos_free_gb:.1f} GB on the Photos volume."
        )
        return result

    def attempt(self, year: int) -> ExportOutcome:
        """Export one batch for a year, then run the pipeline over whatever landed."""
        outcome = self.export(self.root, year, self.pacer.batch_size, self.log)
        message = f"export {year}: {outcome.exported} written, {outcome.new} still outstanding"
        if outcome.failed:
            self.log.problem(f"{message}, {outcome.failed} failed")
        else:
            self.log.say(message)

        for name, step in self.steps:
            if self.stop.requested:
                self.log.problem(f"Not starting '{name}': a stop was requested.")
                break
            self.log.say(f"step: {name}")
            try:
                step(self.root)
            except typer.Exit as exc:
                if exc.exit_code:
                    raise
                self.log.say(f"'{name}' had nothing to do.")
        return outcome


# ---------------------------------------------------------------------------
# The plan, and the table at the end
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class YearPlan:
    """How much of one year is still sitting in Photos."""

    year: int
    detected: int
    already: int
    remaining: int

    def batches(self, batch: int) -> int:
        """How many times round the loop this year would take."""
        return -(-self.remaining // batch) if batch > 0 else 0


def survey(root: Path, years: list[int]) -> list[YearPlan]:
    """Ask Photos what is left in each year.

    The Photos database is loaded once for the whole survey rather than once per
    year, because loading it is the expensive part.
    """
    from vex import photos  # noqa: PLC0415 - macOS only, imported on use

    photos.require_photos()
    conn = db.connect(root)
    try:
        catalogued = {
            row[0]
            for row in conn.execute(
                "SELECT photos_uuid FROM screenshots WHERE photos_uuid IS NOT NULL"
            ).fetchall()
        }
    finally:
        conn.close()
    already = catalogued | photos.exported_uuids(Path(root) / "inbox" / "iphone")

    # Movies and raw images are not worth planning around; they never export.
    found, _ = photos.split_by_type(photos.load_screenshots(None)[0])
    plans: list[YearPlan] = []
    for year in years:
        in_year = [p for p in found if p.date and p.date.year == year]
        outstanding = [p for p in in_year if p.uuid not in already]
        plans.append(
            YearPlan(year, len(in_year), len(in_year) - len(outstanding), len(outstanding))
        )
    return plans


def human_gb(value: float) -> str:
    return f"{value:.1f} GB"


def summary_rows(summary: Summary) -> list[tuple[str, ...]]:
    """The summary table as plain rows, so it can be printed and logged alike."""
    rows = [
        (
            str(batch.number),
            str(batch.year),
            str(batch.exported),
            str(batch.added),
            str(batch.read),
            str(batch.classified),
            f"${batch.cost_usd:.4f}",
            str(batch.purged_files),
            f"{batch.purged_bytes / BYTES_PER_MIB:.1f} MiB",
            str(batch.discarded),
            str(batch.failed),
            str(batch.outstanding),
            f"{batch.waited // 60}m" if batch.waited else "",
            human_gb(batch.free_gb),
            batch.note or batch.outcome,
        )
        for batch in summary.batches
    ]
    rows.append(
        (
            "Total",
            "",
            str(int(summary.total("exported"))),
            str(int(summary.total("added"))),
            str(int(summary.total("read"))),
            str(int(summary.total("classified"))),
            f"${summary.total('cost_usd'):.4f}",
            str(int(summary.total("purged_files"))),
            f"{summary.total('purged_bytes') / BYTES_PER_MIB:.1f} MiB",
            str(int(summary.total("discarded"))),
            str(int(summary.total("failed"))),
            "",
            f"{int(summary.total('waited')) // 60}m",
            "",
            "",
        )
    )
    return rows


SUMMARY_COLUMNS = (
    "Batch",
    "Year",
    "Exported",
    "Added",
    "Read",
    "Classified",
    "Cost",
    "Purged",
    "Reclaimed",
    "Re-exports",
    "Failed",
    "Waiting",
    "Waited",
    "Free",
    "Outcome",
)


def report(summary: Summary, log: RunLog) -> None:
    """Print the closing table and write the same numbers into the run log."""
    from rich.table import Table  # noqa: PLC0415

    rows = summary_rows(summary)
    table = Table(title="Backlog run", title_justify="left")
    for column in SUMMARY_COLUMNS:
        table.add_column(column, justify="left" if column in ("Batch", "Outcome") else "right")
    for row in rows[:-1]:
        table.add_row(*row)
    table.add_section()
    table.add_row(*rows[-1], style="bold")
    console.print(table)
    console.print(f"Started {summary.started_at}, finished {summary.finished_at}.")
    console.print(f"Reason for stopping: {summary.reason}.")
    console.print(f"Log: {summary.log}")

    log.detail("SUMMARY")
    log.detail("  ".join(SUMMARY_COLUMNS))
    for row in rows:
        log.detail("  ".join(row))
    log.detail(f"started {summary.started_at}, finished {summary.finished_at}, {summary.reason}")
