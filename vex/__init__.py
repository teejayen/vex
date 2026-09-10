"""vex: screenshot consolidation, cataloguing, OCR and classification."""

__version__ = "2.0.0"


def _register_heif() -> None:
    """Teach Pillow to open HEIC and HEIF, everywhere in this package.

    Photos flags plenty of HEIC images as screenshots, usually ones that have been
    edited or shared, and Pillow cannot open those on its own. Without this they
    fail on the way in with "could not read the image" and sit in the inbox
    forever, while their sidecar tells the next export they have already been
    fetched. Registering here rather than in each module means every use of
    Pillow gets it: hashing, dimensions, perceptual hashes and thumbnails alike.
    """
    try:
        from pillow_heif import register_heif_opener  # noqa: PLC0415
    except ImportError:  # pragma: no cover - the dependency is not optional
        return
    register_heif_opener()


_register_heif()
