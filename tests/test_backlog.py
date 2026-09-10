"""The overnight backlog loop: year rotation, disk floor, retries and the summary."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from vex import backlog, db, photos

GB = backlog.BYTES_PER_GB


def make_log(root: Path) -> backlog.RunLog:
    return backlog.RunLog(root / "logs" / "backlog-test.log")


def fixed_free(gigabytes: float):  # noqa: ANN201
    return lambda _: int(gigabytes * GB)


def counting_export(per_year: dict[int, list[int]]):  # noqa: ANN201
    """An export that writes a scripted number of items for each year.

    ``new`` is what Photos still held when the export started, which is how the
    real command reports it, so the year drains on the call that finds nothing.
    """

    def export(_root: Path, year: int, _limit: int, _log: object = None) -> backlog.ExportOutcome:
        queue = per_year.get(year, [])
        waiting = sum(queue)
        wrote = queue.pop(0) if queue else 0
        return backlog.ExportOutcome(exported=wrote, new=waiting)

    return export


def barren_export(waiting: int = 500):  # noqa: ANN201
    """An export whose downloads all fail: nothing written, plenty still waiting."""

    def export(_root: Path, _year: int, limit: int, _log: object = None) -> backlog.ExportOutcome:
        return backlog.ExportOutcome(exported=0, new=waiting, failed=limit)

    return export


def run_loop(root: Path, **kwargs: object) -> backlog.Summary:
    defaults = {
        "root": root,
        "log": make_log(root),
        "years": [2023, 2024],
        "batch": 10,
        "steps": [],
        "sleep": lambda _: None,
        "free": fixed_free(50),
        "export": counting_export({}),
    }
    return backlog.BacklogRun(**{**defaults, **kwargs}).run()


# ---------------------------------------------------------------------------
# Years
# ---------------------------------------------------------------------------


def test_parse_years_keeps_the_order_and_drops_repeats() -> None:
    assert backlog.parse_years("2023,2024,2025") == [2023, 2024, 2025]
    assert backlog.parse_years("2026, 2023") == [2026, 2023]
    assert backlog.parse_years("2024,2024") == [2024]
    with pytest.raises(backlog.YearListError, match="not a year"):
        backlog.parse_years("2023,last")
    with pytest.raises(backlog.YearListError, match="No years"):
        backlog.parse_years(",")


def test_a_year_rotates_once_the_export_finds_nothing(root: Path) -> None:
    export = counting_export({2023: [10, 10], 2024: [4]})
    summary = run_loop(root, export=export)

    # 2023 twice then empty, 2024 once then empty: five batches, both years drained.
    assert [(b.year, b.exported) for b in summary.batches] == [
        (2023, 10),
        (2023, 10),
        (2023, 0),
        (2024, 4),
        (2024, 0),
    ]
    assert summary.reason == "every year is drained"
    assert summary.batches[2].outcome == backlog.OUTCOME_DRAINED


def test_max_batches_stops_the_loop(root: Path) -> None:
    export = counting_export({2023: [10] * 20})
    summary = run_loop(root, export=export, max_batches=3)

    assert len(summary.batches) == 3
    assert "3 batch limit" in summary.reason


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------


def test_a_low_disk_stops_before_the_first_batch(root: Path) -> None:
    summary = run_loop(root, free=fixed_free(1.2), export=counting_export({2023: [10]}))

    assert summary.batches == []
    assert "1.2 GB free" in summary.reason
    assert "too full" in summary.reason


def test_the_loop_stops_when_the_disk_fills_mid_run(root: Path) -> None:
    readings = iter([50 * GB, 50 * GB, int(0.5 * GB)])

    summary = run_loop(
        root,
        export=counting_export({2023: [10] * 10}),
        # Two readings per check now, one volume each, so the disk fills after a batch.
        free=lambda _: next(readings, int(0.5 * GB)),
    )

    assert len(summary.batches) == 1
    assert "0.5 GB free" in summary.reason


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_a_transient_failure_is_retried_once_then_the_year_is_skipped(root: Path) -> None:
    attempts: list[int] = []
    waits: list[float] = []

    def flaky(_root: Path, year: int, _limit: int, _log: object = None) -> backlog.ExportOutcome:
        attempts.append(year)
        if year == 2023:
            raise backlog.ExportFailedError("iCloud download timed out")
        return backlog.ExportOutcome(exported=0, new=0)

    summary = run_loop(root, export=flaky, sleep=waits.append)

    assert attempts == [2023, 2023, 2024]  # tried twice, then moved on
    assert waits == [backlog.RETRY_DELAY_SECONDS]
    assert summary.batches[0].outcome == backlog.OUTCOME_SKIPPED
    assert "iCloud download timed out" in summary.batches[0].note
    assert summary.reason == "every year is drained"

    logged = (root / "logs" / "backlog-test.log").read_text(encoding="utf-8")
    assert "Retrying once" in logged
    assert "Skipping 2023" in logged


def test_a_retry_that_works_keeps_the_year(root: Path) -> None:
    calls: list[int] = []

    def flaky(_root: Path, year: int, _limit: int, _log: object = None) -> backlog.ExportOutcome:
        calls.append(year)
        if len(calls) == 1:
            raise OSError("network went away")
        return backlog.ExportOutcome(exported=5, new=20)

    summary = run_loop(root, years=[2023], export=flaky, sleep=lambda _: None, max_batches=1)

    assert summary.batches[0].outcome == backlog.OUTCOME_OK
    assert summary.batches[0].exported == 5


def test_a_failing_pipeline_step_fails_the_batch(root: Path) -> None:
    def broken(_root: Path) -> None:
        raise RuntimeError("classify blew up")

    summary = run_loop(
        root,
        years=[2023],
        export=counting_export({2023: [10, 10]}),
        steps=[("classify", broken)],
    )

    assert summary.batches[0].outcome == backlog.OUTCOME_SKIPPED
    assert "classify blew up" in summary.batches[0].note


def test_a_step_with_nothing_to_do_is_not_a_failure(root: Path) -> None:
    def nothing_to_do(_root: Path) -> None:
        raise typer.Exit(code=0)

    summary = run_loop(
        root,
        years=[2023],
        export=counting_export({2023: [10]}),
        steps=[("ocr", nothing_to_do)],
    )

    assert summary.batches[0].outcome == backlog.OUTCOME_OK


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------


def test_a_stop_request_finishes_the_step_then_ends_the_run(root: Path) -> None:
    stop = backlog.StopFlag()
    ran: list[str] = []

    def first(_root: Path) -> None:
        ran.append("first")
        stop.request("SIGTERM")  # as if the signal landed mid-step

    def second(_root: Path) -> None:
        ran.append("second")

    summary = run_loop(
        root,
        years=[2023, 2024],
        export=counting_export({2023: [10] * 5}),
        steps=[("first", first), ("second", second)],
        stop=stop,
    )

    # The step that was running finished; the next one never started.
    assert ran == ["first"]
    assert len(summary.batches) == 1
    assert summary.reason == "stopped on SIGTERM"
    assert summary.batches[0].note == "stopped on SIGTERM"


def test_signal_handling_restores_the_previous_handlers() -> None:
    import signal  # noqa: PLC0415

    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    flag = backlog.StopFlag()
    with backlog.stop_on_signals(flag) as installed:
        assert installed is flag
        assert signal.getsignal(signal.SIGINT) is not before[0]
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


# ---------------------------------------------------------------------------
# Counting and the summary
# ---------------------------------------------------------------------------


def catalogue_a_row(root: Path, file_hash: str, **values: object) -> None:
    conn = db.connect(root)
    db.insert_screenshot(
        conn,
        {
            "hash": file_hash,
            "path": f"library/2026/07/{file_hash[:8]}.png",
            "original_name": f"{file_hash[:8]}.png",
            "source": "iphone",
            "captured_at": "2026-07-20T14:03:50",
            "ingested_at": "2026-07-21T09:00:00",
            "bytes": 2 * backlog.BYTES_PER_MIB,
            **values,
        },
    )
    conn.close()


def test_counts_come_from_the_catalogue_not_from_what_a_step_said(root: Path) -> None:
    def pretend_pipeline(_root: Path) -> None:
        catalogue_a_row(
            root,
            "a" * 64,
            ocr_text="plenty of words",
            category="social-post",
            purged_at="2026-07-21T10:00:00",
        )
        conn = db.connect(root)
        db.cache_classification(conn, "a" * 64, "some/model", "text", "{}", 10, 5, 0.0125)
        conn.close()

    summary = run_loop(
        root,
        years=[2023],
        export=counting_export({2023: [1]}),
        steps=[("pipeline", pretend_pipeline)],
    )

    first = summary.batches[0]
    assert first.exported == 1
    assert first.added == 1
    assert first.read == 1
    assert first.classified == 1
    assert first.purged_files == 1
    assert first.purged_bytes == 2 * backlog.BYTES_PER_MIB
    assert first.cost_usd == pytest.approx(0.0125)
    assert first.free_gb == pytest.approx(50.0)


def test_the_summary_reports_every_batch_and_a_total(root: Path) -> None:
    log = make_log(root)
    summary = run_loop(root, log=log, export=counting_export({2023: [7, 3], 2024: [2]}))
    backlog.report(summary, log)

    rows = backlog.summary_rows(summary)
    assert rows[-1][0] == "Total"
    assert rows[-1][2] == "12"  # 7 + 3 + 2 exported

    logged = (root / "logs" / "backlog-test.log").read_text(encoding="utf-8")
    assert "SUMMARY" in logged
    assert "every year is drained" in logged
    # Every line in the log is timestamped.
    assert all(line[:4].isdigit() for line in logged.splitlines() if line.strip())


def test_the_log_path_is_stamped_with_the_start_time(root: Path) -> None:
    from datetime import datetime  # noqa: PLC0415

    path = backlog.log_path(root, datetime(2026, 9, 7, 22, 30))  # noqa: DTZ001
    assert path == root / "logs" / "backlog-20260907-2230.log"


# ---------------------------------------------------------------------------
# The export subprocess
# ---------------------------------------------------------------------------


def test_export_counts_what_landed_in_the_inbox(root: Path, monkeypatch) -> None:  # noqa: ANN001
    import subprocess  # noqa: PLC0415

    inbox = root / "inbox" / "iphone"
    (inbox / "IMG_0001.png").write_bytes(b"already here")

    reported = photos.ExportResult(detected=900, new=880, pending=50, exported=1, failed=49)

    def fake_run(command: list[str], **_kwargs: object) -> object:
        assert command[1:3] == ["-m", "vex"]
        assert "--year" in command
        assert "2024" in command
        (inbox / "IMG_0002.png").write_bytes(b"new")
        (inbox / "IMG_0002.png.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, f"a table\n{reported.format()}\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    log = make_log(root)
    outcome = backlog.export_batch(root, 2024, 50, log)

    # The sidecar is not counted, only the image beside it.
    assert outcome.exported == 1
    assert outcome.failed == 49
    assert outcome.new == 880
    assert not outcome.drained
    # The whole subprocess output reaches the log, or a failing export is invisible.
    assert "a table" in log.path.read_text(encoding="utf-8")


def test_an_export_without_a_result_line_is_not_treated_as_drained(root: Path, monkeypatch) -> None:  # noqa: ANN001
    import subprocess  # noqa: PLC0415

    def fake_run(command: list[str], **_kwargs: object) -> object:
        return subprocess.CompletedProcess(command, 0, "no machine-readable line here", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    outcome = backlog.export_batch(root, 2024, 50)
    assert outcome.exported == 0
    assert not outcome.drained


def test_a_failing_export_subprocess_raises(root: Path, monkeypatch) -> None:  # noqa: ANN001
    import subprocess  # noqa: PLC0415

    def fake_run(command: list[str], **_kwargs: object) -> object:
        return subprocess.CompletedProcess(command, 2, "", "Photos library is locked")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(backlog.ExportFailedError, match="Photos library is locked"):
        backlog.export_batch(root, 2024, 50)


def test_discarded_reexports_are_counted_per_batch(root: Path) -> None:
    """The loop reports what re-walking an already-ingested year cost."""
    from vex.commands import ingest as ingest_command  # noqa: PLC0415

    logged: list[str] = [
        f"hash already catalogued, photos_uuid recorded, {ingest_command.DISCARDED_DETAIL}",
        f"hash already catalogued, original purged, {ingest_command.DISCARDED_DETAIL}",
        "hash already catalogued",  # parked, not discarded
    ]

    def pretend_ingest(_root: Path) -> None:
        conn = db.connect(root)
        for index, detail in enumerate(logged):
            db.log_ingest(
                conn, "2026-09-08T01:00:00", f"/inbox/{index}", "h" * 64, "duplicate", detail
            )
        conn.close()
        logged.clear()

    summary = run_loop(
        root,
        years=[2023],
        export=counting_export({2023: [3]}),
        steps=[("ingest", pretend_ingest)],
    )

    assert summary.batches[0].discarded == 2
    assert summary.batches[1].discarded == 0  # the drained batch discards nothing

    rows = backlog.summary_rows(summary)
    assert "Re-exports" in backlog.SUMMARY_COLUMNS
    assert rows[-1][backlog.SUMMARY_COLUMNS.index("Re-exports")] == "2"

    written = (root / "logs" / "backlog-test.log").read_text(encoding="utf-8")
    assert "discarded 2 re-exports" in written


# ---------------------------------------------------------------------------
# A year whose downloads fail is not a drained year
# ---------------------------------------------------------------------------


def test_a_batch_that_wrote_nothing_does_not_rotate_the_year(root: Path) -> None:
    """The bug this guards: an all-failed batch looks exactly like a drained one."""
    summary = run_loop(root, years=[2023, 2024], export=barren_export(3807), max_batches=2)

    assert [b.year for b in summary.batches] == [2023, 2023]
    first = summary.batches[0]
    assert first.outcome == backlog.OUTCOME_BARREN
    assert first.exported == 0
    assert first.failed == 10  # the whole batch was asked for and none arrived
    assert first.outstanding == 3807
    assert "3807 still waiting" in first.note

    written = (root / "logs" / "backlog-test.log").read_text(encoding="utf-8")
    assert "10 failed" in written


def test_a_year_is_given_up_on_only_once_the_backoff_is_spent(root: Path) -> None:
    """Three empty batches is not enough on its own: throttling must be ruled out."""
    waits: list[float] = []
    summary = run_loop(
        root, years=[2023, 2024], export=barren_export(500), sleep=waits.append, max_batches=6
    )

    years = [b.year for b in summary.batches]
    # Four batches to walk the backoff up to its cap, then the give-up can fire.
    assert years == [2023, 2023, 2023, 2023, 2023, 2024]
    # Five minutes, doubling, capped at thirty. The wait carries across the year
    # change, because throttling is the machine's mood, not the year's.
    assert waits[:5] == [300, 600, 1200, 1800, 1800]
    assert max(waits) == backlog.BACKOFF_CAP_SECONDS
    assert "gave up after" in summary.batches[4].note


def test_a_batch_that_writes_something_resets_the_barren_count(root: Path) -> None:
    calls: list[int] = []

    def sometimes(
        _root: Path, _year: int, limit: int, _log: object = None
    ) -> backlog.ExportOutcome:
        calls.append(limit)
        # Nothing, nothing, then one through: never three barren in a row.
        wrote = 0 if len(calls) % 3 else 5
        return backlog.ExportOutcome(exported=wrote, new=400, failed=limit - wrote)

    summary = run_loop(root, years=[2023, 2024], export=sometimes, max_batches=6)

    assert {b.year for b in summary.batches} == {2023}
    assert summary.reason == "reached the 6 batch limit"


def test_the_summary_shows_failures_and_what_is_still_waiting(root: Path) -> None:
    summary = run_loop(root, years=[2023], export=barren_export(3807), max_batches=1)
    rows = backlog.summary_rows(summary)

    assert rows[0][backlog.SUMMARY_COLUMNS.index("Failed")] == "10"
    assert rows[0][backlog.SUMMARY_COLUMNS.index("Waiting")] == "3807"
    assert rows[-1][backlog.SUMMARY_COLUMNS.index("Failed")] == "10"


# ---------------------------------------------------------------------------
# The result line itself
# ---------------------------------------------------------------------------


def test_the_export_result_line_survives_a_round_trip() -> None:
    original = photos.ExportResult(detected=4579, new=3807, pending=300, exported=13, failed=287)
    assert photos.parse_result(f"some table\n{original.format()}\ntrailing") == original
    assert photos.parse_result("nothing machine readable here") is None
    assert photos.parse_result("export-result detected=oops") is None


def test_a_result_line_of_zero_new_means_drained() -> None:
    assert photos.ExportResult(detected=10, new=0).drained
    assert not photos.ExportResult(detected=10, new=1).drained


# ---------------------------------------------------------------------------
# Pacing against a throttling Photos daemon
# ---------------------------------------------------------------------------


def test_the_backoff_doubles_from_five_minutes_and_caps_at_thirty() -> None:
    pacer = backlog.Pacer(full_batch=300)
    waits = []
    for _ in range(6):
        assert pacer.observe(exported=0, failed=100) is not None
        waits.append(pacer.wait_seconds)

    assert waits == [300, 600, 1200, 1800, 1800, 1800]
    assert pacer.at_cap


def test_only_a_majority_of_failures_counts_as_throttling() -> None:
    pacer = backlog.Pacer(full_batch=300)

    # Half failed is bad luck, not a refusal to serve.
    assert pacer.observe(exported=50, failed=50) is None
    assert pacer.wait_seconds == 0
    assert pacer.batch_size == 300

    # Just over half is.
    assert pacer.observe(exported=40, failed=60) is not None
    assert pacer.wait_seconds == 300


def test_a_clean_batch_resets_the_backoff_and_wins_back_the_batch_size() -> None:
    pacer = backlog.Pacer(full_batch=300)
    pacer.observe(exported=0, failed=300)
    pacer.observe(exported=0, failed=150)
    assert pacer.wait_seconds == 600
    assert pacer.batch_size == 75  # halved, then halved again

    # A batch that mostly worked clears the wait entirely.
    assert pacer.observe(exported=95, failed=5) is None
    assert pacer.wait_seconds == 0
    assert pacer.batch_size == 112

    # ... and the size climbs back gradually rather than all at once.
    sizes = []
    for _ in range(4):
        pacer.observe(exported=100, failed=0)
        sizes.append(pacer.batch_size)
    assert sizes == [168, 252, 300, 300]


def test_a_middling_batch_holds_everything_steady() -> None:
    pacer = backlog.Pacer(full_batch=300)
    pacer.observe(exported=0, failed=300)
    assert (pacer.wait_seconds, pacer.batch_size) == (300, 150)

    # Between the two thresholds: no worse, no better, so change nothing.
    assert pacer.observe(exported=70, failed=30) is None
    assert (pacer.wait_seconds, pacer.batch_size) == (300, 150)


def test_the_batch_size_never_falls_below_the_floor() -> None:
    pacer = backlog.Pacer(full_batch=300)
    for _ in range(10):
        pacer.observe(exported=0, failed=100)
    assert pacer.batch_size == backlog.MIN_BATCH


def test_an_empty_batch_teaches_the_pacer_nothing() -> None:
    pacer = backlog.Pacer(full_batch=300)
    assert pacer.observe(exported=0, failed=0) is None
    assert pacer.wait_seconds == 0
    assert pacer.batch_size == 300


def test_the_loop_asks_for_less_after_a_throttled_batch(root: Path) -> None:
    asked: list[int] = []
    waits: list[float] = []

    def watching(_root: Path, _year: int, limit: int, _log: object = None) -> backlog.ExportOutcome:
        asked.append(limit)
        # Heavily throttled at first, then it clears.
        failed = limit if len(asked) <= 2 else 0
        return backlog.ExportOutcome(exported=limit - failed, new=1000, failed=failed)

    summary = run_loop(
        root, years=[2023], batch=200, export=watching, sleep=waits.append, max_batches=5
    )

    # 200, halved, halved again, then winning it back.
    assert asked == [200, 100, 50, 75, 112]
    assert waits == [300, 600]  # a clean batch is never made to wait at all
    assert summary.batches[0].waited == 300
    assert summary.batches[2].waited == 0
    assert [b.batch_size for b in summary.batches] == [200, 100, 50, 75, 112]


def test_a_throttled_batch_is_logged_with_its_reason(root: Path) -> None:
    log = make_log(root)
    run_loop(root, log=log, years=[2023], export=barren_export(900), max_batches=1)

    written = log.path.read_text(encoding="utf-8")
    assert "throttling rather than bad items" in written
    assert "Waiting 5 minutes before the next batch" in written


def test_a_stop_request_is_not_made_to_wait_out_a_backoff(root: Path) -> None:
    """Half an hour is a long time to ignore a Ctrl-C."""
    stop = backlog.StopFlag()
    waits: list[float] = []

    def throttled(
        _root: Path, _year: int, limit: int, _log: object = None
    ) -> backlog.ExportOutcome:
        stop.request("SIGINT")
        return backlog.ExportOutcome(exported=0, new=900, failed=limit)

    summary = run_loop(root, years=[2023], export=throttled, sleep=waits.append, stop=stop)

    assert waits == []
    assert summary.reason == "stopped on SIGINT"


# ---------------------------------------------------------------------------
# Failures that rest cannot fix
# ---------------------------------------------------------------------------


def test_failures_near_the_disk_floor_stop_the_run_instead_of_waiting(root: Path) -> None:
    """Photos refuses to download onto a full disk. Waiting it out achieves nothing."""
    log = make_log(root)
    waits: list[float] = []
    summary = run_loop(
        root,
        log=log,
        years=[2023, 2024],
        export=barren_export(3807),
        free=fixed_free(11),  # above the floor, but inside the headroom
        sleep=waits.append,
        max_batches=5,
    )

    assert waits == []  # no thirty-minute wait for something rest cannot fix
    assert len(summary.batches) == 1
    assert summary.reason == "stopped on low disk space"
    assert summary.batches[0].note == "stopped, disk too full for Photos to serve originals"

    written = log.path.read_text(encoding="utf-8")
    assert "CloudPhotoLibrary error 1005" in written
    assert "Disk space is very low" in written
    assert "Free up disk space" in written


def test_the_same_failures_with_room_to_spare_are_still_treated_as_throttling(
    root: Path,
) -> None:
    waits: list[float] = []
    summary = run_loop(
        root,
        years=[2023],
        export=barren_export(3807),
        free=fixed_free(60),
        sleep=waits.append,
        max_batches=2,
    )

    assert waits == [300, 600]
    assert summary.reason == "reached the 2 batch limit"


def test_a_clean_batch_near_the_floor_is_left_alone(root: Path) -> None:
    """Only a failing batch is read as a disk problem; a working one is working."""
    summary = run_loop(
        root,
        years=[2023],
        export=counting_export({2023: [50, 50]}),
        free=fixed_free(11),
        max_batches=2,
    )

    assert len(summary.batches) == 2
    assert summary.reason == "reached the 2 batch limit"


def test_the_low_disk_message_names_the_error_and_the_fix() -> None:
    same_disk = photos.DiskCheck(root_gb=4.1, photos_gb=4.1, same_volume=True)
    message = photos.low_disk_message(same_disk)
    assert "4.1 GB free" in message
    assert "Photos needs 6 GB free on its own volume" in message
    assert "the library volume needs 10 GB" in message
    assert f"CloudPhotoLibrary error {photos.CLOUD_LOW_DISK_CODE}" in message
    assert "Free up disk space" in message


def test_the_message_separates_the_volumes_once_they_come_apart() -> None:
    on_a_drive = photos.DiskCheck(root_gb=1000.0, photos_gb=5.2, same_volume=False)
    message = photos.low_disk_message(on_a_drive)
    assert "1000.0 GB free on the library volume" in message
    assert "5.2 GB on the Photos volume" in message
    # The roomy drive is not a constraint, so its floor is not quoted at all.
    assert "library volume needs" not in message


def test_the_loop_stops_on_the_photos_volume_even_with_a_roomy_drive(root: Path) -> None:
    """A terabyte of library drive is no help when Photos has nowhere to write."""
    library = root / "fake-photos-library"
    library.mkdir()

    def free(path: Path) -> int:
        gb = 5.0 if path == library else 1000.0
        return int(gb * GB)

    summary = run_loop(
        root,
        years=[2023],
        export=counting_export({2023: [50] * 5}),
        free=free,
        photos_library=library,
    )

    assert summary.batches == []
    assert "Photos volume is too full" in summary.reason


def test_each_batch_line_carries_both_volumes(root: Path) -> None:
    library = root / "fake-photos-library"
    library.mkdir()
    log = make_log(root)

    def free(path: Path) -> int:
        gb = 40.0 if path == library else 900.0
        return int(gb * GB)

    run_loop(
        root,
        log=log,
        years=[2023],
        export=counting_export({2023: [50]}),
        free=free,
        photos_library=library,
        max_batches=1,
    )

    written = log.path.read_text(encoding="utf-8")
    assert "900.0 GB free on the library volume" in written
    assert "40.0 GB on the Photos volume" in written
