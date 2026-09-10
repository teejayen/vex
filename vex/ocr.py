"""Text recognition engines: Apple Vision on macOS, Tesseract elsewhere."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

RECOGNITION_LANGUAGES = ["en-AU", "en"]

# Vision does not need full resolution to read a screenshot, and decoding a large
# PNG at native size is the main driver of peak memory during a long run.
MAX_RECOGNITION_PIXELS = 2500


class OcrUnavailableError(RuntimeError):
    """Raised when no usable OCR engine is installed."""


def vision_available() -> bool:
    """True if the Apple Vision bindings can be imported on this machine."""
    if sys.platform != "darwin":
        return False
    try:
        import Quartz  # noqa: F401, PLC0415
        import Vision  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def tesseract_available() -> bool:
    """True if pytesseract is installed and the tesseract binary is on PATH."""
    try:
        import pytesseract  # noqa: F401, PLC0415
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def resolve_engine(engine: str) -> str:
    """Turn 'auto' into a concrete engine name, or raise if none is usable."""
    if engine == "vision":
        if not vision_available():
            raise OcrUnavailableError(
                "Apple Vision is not available. Install the mac extra with "
                "'uv sync --extra mac' on macOS."
            )
        return "vision"
    if engine == "tesseract":
        if not tesseract_available():
            raise OcrUnavailableError(
                "Tesseract is not available. Install the tesseract extra and the "
                "tesseract binary (brew install tesseract)."
            )
        return "tesseract"
    if vision_available():
        return "vision"
    if tesseract_available():
        return "tesseract"
    raise OcrUnavailableError(
        "No OCR engine available. On macOS run 'uv sync --extra mac'. "
        "Otherwise install Tesseract and run 'uv sync --extra tesseract'."
    )


def _pixel_size(quartz: object, source: object) -> tuple[int, int] | None:
    """Read an image's pixel dimensions from its metadata, without decoding it."""
    properties = quartz.CGImageSourceCopyPropertiesAtIndex(source, 0, None)
    if not properties:
        return None
    width = properties.get(quartz.kCGImagePropertyPixelWidth)
    height = properties.get(quartz.kCGImagePropertyPixelHeight)
    if width is None or height is None:
        return None
    return int(width), int(height)


def _decode_for_recognition(quartz: object, source: object) -> object:
    """Decode an image for Vision, downscaling anything oversized.

    Vision gains nothing from full resolution on a screenshot, and decoding a
    16 MB PNG at native size is what drives peak memory. Core Graphics can decode
    straight to a bounded size, so the full-resolution bitmap never exists.
    """
    size = _pixel_size(quartz, source)
    if size is None or max(size) > MAX_RECOGNITION_PIXELS:
        options = {
            quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
            quartz.kCGImageSourceThumbnailMaxPixelSize: MAX_RECOGNITION_PIXELS,
            quartz.kCGImageSourceCreateThumbnailWithTransform: True,
            quartz.kCGImageSourceShouldCache: False,
        }
        return quartz.CGImageSourceCreateThumbnailAtIndex(source, 0, options)
    return quartz.CGImageSourceCreateImageAtIndex(
        source, 0, {quartz.kCGImageSourceShouldCache: False}
    )


def recognise_vision(path: Path) -> str:
    """Run Apple Vision text recognition, returning lines ordered top to bottom.

    Everything Objective-C happens inside an autorelease pool and is released
    before returning. Without the pool, autoreleased Core Graphics and Vision
    objects accumulate for the life of the process, which over a long run holds
    hundreds of megabytes of decoded bitmaps that nothing is still using.
    """
    import objc  # noqa: PLC0415
    import Quartz  # noqa: PLC0415
    import Vision  # noqa: PLC0415
    from Foundation import NSURL  # noqa: PLC0415

    with objc.autorelease_pool():
        source = None
        image = None
        request = None
        handler = None
        try:
            url = NSURL.fileURLWithPath_(str(path))
            source = Quartz.CGImageSourceCreateWithURL(url, None)
            if source is None:
                raise OSError(f"Could not open image for Vision: {path}")

            image = _decode_for_recognition(Quartz, source)
            if image is None:
                raise OSError(f"Could not decode image for Vision: {path}")

            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            request.setRecognitionLanguages_(RECOGNITION_LANGUAGES)

            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
            success, error = handler.performRequests_error_([request], None)
            if not success:
                raise OSError(f"Vision request failed: {error}")

            lines: list[tuple[float, str]] = []
            for observation in request.results() or []:
                candidates = observation.topCandidates_(1)
                if not candidates:
                    continue
                text = candidates[0].string()
                if not text:
                    continue
                box = observation.boundingBox()
                # Vision's origin is bottom-left, so a higher y is nearer the top.
                lines.append((-box.origin.y, str(text)))

            lines.sort(key=lambda item: item[0])
            # Built from Python floats and strings, so it outlives the pool safely.
            recognised = "\n".join(text for _, text in lines)
        finally:
            # Drop the last strong references before the pool drains.
            del handler, request, image, source

    return recognised


def recognise_tesseract(path: Path) -> str:
    """Run Tesseract over the full-size image."""
    import pytesseract  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as image:
        return str(pytesseract.image_to_string(image)).strip()


def recognise(path: Path, engine: str = "auto") -> tuple[str, str]:
    """Recognise text in an image. Returns (text, engine used)."""
    resolved = resolve_engine(engine)
    if resolved == "vision":
        return recognise_vision(path), "vision"
    return recognise_tesseract(path), "tesseract"
