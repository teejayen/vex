"""Face detection with Apple Vision.

Only a count is stored. The catalogue never keeps a face's location, an
identifier, or anything else that would amount to recognising a person: the
number exists so the purge command can leave anything with a person in it alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

from vex import config


class FaceDetectionUnavailableError(RuntimeError):
    """Raised when Apple Vision face detection is not available on this machine."""


UNAVAILABLE_MESSAGE = (
    "Face detection needs Apple Vision, which is macOS only. "
    "On macOS install the mac extra with 'uv sync --extra mac'."
)


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


def require_vision() -> None:
    """Raise unless Apple Vision face detection can run here."""
    if not vision_available():
        raise FaceDetectionUnavailableError(UNAVAILABLE_MESSAGE)


def detect_faces(path: Path) -> int:
    """Count the faces Vision finds in an image.

    Run against a thumbnail rather than the original: 512 px is ample for face
    rectangles, and it keeps a long pass off the full-size decode path.

    Everything Objective-C happens inside an autorelease pool and is released
    before returning, the same shape as the OCR pass. Without the pool the
    autoreleased Core Graphics and Vision objects accumulate for the life of the
    process, and memory climbs steadily across thousands of rows.
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

            image = Quartz.CGImageSourceCreateImageAtIndex(
                source, 0, {Quartz.kCGImageSourceShouldCache: False}
            )
            if image is None:
                raise OSError(f"Could not decode image for Vision: {path}")

            request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
            success, error = handler.performRequests_error_([request], None)
            if not success:
                raise OSError(f"Vision face request failed: {error}")

            # A plain Python int, so it outlives the pool safely.
            count = len(request.results() or [])
        finally:
            # Drop the last strong references before the pool drains.
            del handler, request, image, source

    return count


def detect_on_thumbnail(root: Path, file_hash: str) -> int:
    """Count faces on a row's thumbnail. Raises if the thumbnail is not there."""
    thumb = config.thumb_path(root, file_hash)
    if not thumb.is_file():
        raise OSError(f"No thumbnail for {file_hash[:8]}")
    return detect_faces(thumb)


def detect_best_effort(root: Path, file_hash: str) -> int | None:
    """Count faces if that is possible right now, otherwise None.

    Used on the ingest path, where face detection is a bonus rather than the job:
    no Vision, no thumbnail, or a failure of any kind leaves ``faces`` null so a
    later ``screenshots faces`` run picks the row up.
    """
    if not vision_available():
        return None
    try:
        return detect_on_thumbnail(root, file_hash)
    except Exception:  # noqa: BLE001 - best effort; the row simply stays unchecked
        return None
