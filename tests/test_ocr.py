"""Engine selection and the OCR command."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from tests.conftest import make_image
from vex import db, ocr
from vex.cli import app

runner = CliRunner()


def test_resolve_engine_prefers_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "vision_available", lambda: True)
    monkeypatch.setattr(ocr, "tesseract_available", lambda: True)
    assert ocr.resolve_engine("auto") == "vision"


def test_resolve_engine_falls_back_to_tesseract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "vision_available", lambda: False)
    monkeypatch.setattr(ocr, "tesseract_available", lambda: True)
    assert ocr.resolve_engine("auto") == "tesseract"


def test_resolve_engine_raises_when_nothing_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "vision_available", lambda: False)
    monkeypatch.setattr(ocr, "tesseract_available", lambda: False)
    with pytest.raises(ocr.OcrUnavailableError):
        ocr.resolve_engine("auto")


def test_requesting_an_unavailable_engine_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "vision_available", lambda: False)
    with pytest.raises(ocr.OcrUnavailableError, match="Apple Vision"):
        ocr.resolve_engine("vision")


def test_ocr_command_stores_text_and_engine(
    root: Path, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png")
    assert runner.invoke(app, ["ingest", "--root", str(root)]).exit_code == 0

    monkeypatch.setattr(ocr, "resolve_engine", lambda _: "vision")
    monkeypatch.setattr(ocr, "recognise", lambda *_: ("invoice total 249.00", "vision"))

    result = runner.invoke(app, ["ocr", "--root", str(root)])
    assert result.exit_code == 0, result.output

    conn = db.connect(root)
    row = conn.execute("SELECT ocr_text, ocr_engine, ocr_at FROM screenshots").fetchone()
    conn.close()
    assert row["ocr_text"] == "invoice total 249.00"
    assert row["ocr_engine"] == "vision"
    assert row["ocr_at"]


def test_empty_results_are_stored_as_empty_strings(
    root: Path, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png")
    runner.invoke(app, ["ingest", "--root", str(root)])

    monkeypatch.setattr(ocr, "resolve_engine", lambda _: "vision")
    monkeypatch.setattr(ocr, "recognise", lambda *_: ("   ", "vision"))
    runner.invoke(app, ["ocr", "--root", str(root)])

    conn = db.connect(root)
    row = conn.execute("SELECT ocr_text FROM screenshots").fetchone()
    conn.close()
    assert row["ocr_text"] == ""

    # An empty result is not retried, so a second run has nothing to do.
    result = runner.invoke(app, ["ocr", "--root", str(root)])
    assert "Nothing to OCR" in result.output


def test_unknown_engine_is_rejected(root: Path) -> None:
    result = runner.invoke(app, ["ocr", "--engine", "nope", "--root", str(root)])
    assert result.exit_code == 2


@pytest.mark.skipif(not ocr.vision_available(), reason="Apple Vision is not available")
def test_apple_vision_reads_real_text(tmp_path: Path) -> None:
    path = tmp_path / "vision.png"
    image = Image.new("RGB", (640, 200), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "INVOICE TOTAL", fill=(0, 0, 0))
    image = image.resize((1280, 400), Image.Resampling.LANCZOS)
    image.save(path)

    text, engine = ocr.recognise(path, "vision")
    assert engine == "vision"
    assert "INVOICE" in text.upper()


@pytest.mark.skipif(not ocr.vision_available(), reason="Apple Vision is not available")
def test_oversized_images_are_decoded_down_to_the_cap(tmp_path: Path) -> None:
    import Quartz  # noqa: PLC0415
    from Foundation import NSURL  # noqa: PLC0415

    path = tmp_path / "big.png"
    Image.new("RGB", (4000, 3000), (255, 255, 255)).save(path)

    source = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
    assert ocr._pixel_size(Quartz, source) == (4000, 3000)

    image = ocr._decode_for_recognition(Quartz, source)
    longest = max(Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image))
    assert longest == ocr.MAX_RECOGNITION_PIXELS


@pytest.mark.skipif(not ocr.vision_available(), reason="Apple Vision is not available")
def test_images_under_the_cap_are_not_resized(tmp_path: Path) -> None:
    import Quartz  # noqa: PLC0415
    from Foundation import NSURL  # noqa: PLC0415

    path = tmp_path / "small.png"
    Image.new("RGB", (800, 600), (255, 255, 255)).save(path)

    source = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
    image = ocr._decode_for_recognition(Quartz, source)
    assert (Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)) == (800, 600)


@pytest.mark.skipif(not ocr.vision_available(), reason="Apple Vision is not available")
def test_vision_still_reads_a_downscaled_image(tmp_path: Path) -> None:
    """Downscaling must not cost us the text on an oversized screenshot."""
    path = tmp_path / "big.png"
    image = Image.new("RGB", (1600, 500), (255, 255, 255))
    ImageDraw.Draw(image).text((20, 200), "QUARTERLY INVOICE", fill=(0, 0, 0))
    image.resize((4800, 1500), Image.Resampling.LANCZOS).save(path)

    text, _ = ocr.recognise(path, "vision")
    assert "INVOICE" in text.upper()


def test_each_row_is_committed_before_the_next_is_read(
    root: Path, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill part way through must not lose the rows already recognised."""
    for minute in ("50", "51", "52"):
        make_image(inbox / f"Screenshot 2026-07-20 at 14.03.{minute}.png", f"text {minute}")
    assert runner.invoke(app, ["ingest", "--root", str(root)]).exit_code == 0

    seen: list[str] = []

    def recognise_then_die(path: Path, _engine: str) -> tuple[str, str]:
        if len(seen) == 2:
            raise KeyboardInterrupt
        seen.append(str(path))
        return (f"recognised {len(seen)}", "vision")

    monkeypatch.setattr(ocr, "resolve_engine", lambda _: "vision")
    monkeypatch.setattr(ocr, "recognise", recognise_then_die)

    # Click turns an interrupt into an abort, so the run ends non-zero part way through.
    result = runner.invoke(app, ["ocr", "--root", str(root)])
    assert result.exit_code != 0
    assert len(seen) == 2

    # A brand new connection sees the work done before the interruption.
    conn = db.connect(root)
    done = conn.execute("SELECT COUNT(*) FROM screenshots WHERE ocr_text IS NOT NULL").fetchone()[0]
    conn.close()
    assert done == 2
