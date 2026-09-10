"""macOS Photos library integration: exporting screenshots and deleting assets.

Everything here is macOS-only and needs the `mac` extra, which brings in
osxphotos. Nothing in this module is imported at start-up on other platforms.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from vex import naming

# Exporting an iCloud original can be slow on a cold connection.
EXPORT_TIMEOUT_SECONDS = 300

#: One PhotoKit request is one confirmation dialogue, so the default is a single
#: request for the whole run. At 200 per request, clearing this library meant
#: seventy-four dialogues. Apple documents no limit on how many assets a single
#: delete request may carry; --chunk exists for the day one turns up.
DEFAULT_DELETE_CHUNK = 0

#: PHPhotosError.userCancelled. Declining the dialogue is a decision, not a fault.
PHOTOS_USER_CANCELLED = 3072


class PhotosUnavailableError(RuntimeError):
    """Raised when the Photos integration cannot run on this machine."""


class DeletionDeclinedError(RuntimeError):
    """Raised when the confirmation dialogue was dismissed. Nothing was deleted."""


@dataclass(frozen=True)
class ExportedItem:
    """One screenshot exported out of Photos, with the sidecar written beside it."""

    uuid: str
    original_filename: str
    path: Path
    sidecar: Path


def photos_available() -> bool:
    """True if osxphotos can be imported on this machine."""
    if sys.platform != "darwin":
        return False
    try:
        import osxphotos  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def require_photos() -> None:
    """Raise a clear error when the Photos integration is not usable."""
    if sys.platform != "darwin":
        raise PhotosUnavailableError("The Photos commands only run on macOS.")
    if not photos_available():
        raise PhotosUnavailableError(
            "osxphotos is not installed. Run 'uv sync --extra mac' on macOS."
        )


#: Types the pipeline can actually read. Photos' screenshot flag is broader than
#: it sounds: this library holds 14385 PNG, 246 JPEG and 134 HEIC, all real
#: screenshots that have been edited or shared, alongside 26 Adobe raw images and
#: 12 movies, which are screen recordings and photographs rather than screenshots.
#: An allow-list rather than a deny-list, because an unknown type that cannot be
#: read is worse than one that is set aside and counted.
INGESTABLE_UTIS: frozenset[str] = frozenset(
    {
        "public.png",
        "public.jpeg",
        "public.heic",
        "public.heif",
        "public.tiff",
        "com.compuserve.gif",
        "org.webmproject.webp",
    }
)


def is_ingestable(photo: Any) -> bool:
    """True when an asset is a still image whose ORIGINAL this pipeline can read.

    Three checks, because the asset's own type is not the whole story. Photos
    records IMG_7376.DNG as ``public.jpeg`` while the original it hands over is a
    raw DNG, so the type alone waves it through. That one asset was exported,
    set aside and re-exported twenty times before anyone noticed. The original's
    own type and the original's filename are what settle it.
    """
    if getattr(photo, "ismovie", False):
        return False
    kinds = [str(getattr(photo, name, "") or "").lower() for name in ("uti", "uti_original")]
    if not any(kinds):
        return False
    if any(kind and kind not in INGESTABLE_UTIS for kind in kinds):
        return False
    original = getattr(photo, "original_filename", None)
    return not original or naming.is_image(Path(str(original)))


def split_by_type(found: list[Any]) -> tuple[list[Any], dict[str, int]]:
    """Split assets into the ones worth exporting and a tally of what was set aside.

    Exporting a screen recording or a raw image only puts a file in the inbox that
    nothing downstream can read, where it sits blocking its own UUID.
    """
    keep: list[Any] = []
    set_aside: dict[str, int] = {}
    for photo in found:
        if is_ingestable(photo):
            keep.append(photo)
            continue
        label = "movie" if getattr(photo, "ismovie", False) else str(photo.uti or "unknown")
        set_aside[label] = set_aside.get(label, 0) + 1
    return keep, set_aside


#: Names for the two ways of deciding what counts as a screenshot, so a run can
#: say which one it used.
SELECTOR_DETECTED = "Photos' own screenshot flag"
SELECTOR_OSXPHOTOS = "osxphotos' screenshot property"

#: Photos records its own judgement on every asset. osxphotos exposes a narrower
#: one: of 14766 assets this library flags, osxphotos agrees with 14465 and calls
#: the other 301 ordinary photographs. They are mostly JPEG and HEIC, and they are
#: real screenshots. Reading the flag directly is the difference between having
#: them and not. Trashed assets are excluded here as well as by osxphotos.
DETECTED_SCREENSHOT_SQL = (
    "SELECT ZUUID, ZUNIFORMTYPEIDENTIFIER FROM ZASSET "
    "WHERE ZISDETECTEDSCREENSHOT = 1 AND ZTRASHEDSTATE = 0"
)


def library_database(library: Path | None = None) -> Path:
    """The SQLite file inside a Photos library."""
    return Path(library or library_path()) / "database" / "Photos.sqlite"


def detected_screenshot_uuids(library: Path | None = None) -> set[str] | None:
    """UUIDs Photos itself flagged as screenshots, or None if it cannot be read.

    Opened read-only and immutable, because Photos is very likely running and
    holding its own locks. Nothing is written, and every failure is a None that
    sends the caller back to osxphotos rather than stopping the run.
    """
    path = library_database(library)
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error:
        return None
    try:
        rows = conn.execute(DETECTED_SCREENSHOT_SQL).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    # The recorded type is a first filter, not the last word: it is what Photos
    # thinks the asset is, which for a raw original can be the rendered JPEG.
    # is_ingestable still has to look at the original itself.
    return {str(row[0]) for row in rows if row[0] and str(row[1] or "").lower() in INGESTABLE_UTIS}


def load_screenshots(year: int | None = None) -> tuple[list[Any], str]:
    """Every asset Photos calls a screenshot, oldest first, and how they were chosen.

    Loading the Photos database costs several hundred megabytes, so this is
    called once per command rather than per item.
    """
    require_photos()
    import osxphotos  # noqa: PLC0415

    database = osxphotos.PhotosDB()
    flagged = detected_screenshot_uuids()
    if flagged is None:
        selector = SELECTOR_OSXPHOTOS
        found = [photo for photo in database.photos() if photo.screenshot]
    else:
        selector = SELECTOR_DETECTED
        found = [photo for photo in database.photos() if photo.uuid in flagged]
    if year is not None:
        found = [photo for photo in found if photo.date and photo.date.year == year]
    found.sort(key=lambda photo: (photo.date is None, photo.date))
    return found, selector


def device_of(photo: Any) -> str | None:
    """The capturing device, when Photos knows it."""
    exif = getattr(photo, "exif_info", None)
    model = getattr(exif, "camera_model", None) if exif else None
    return str(model).strip() or None if model else None


def sidecar_content(photo: Any, exported_at: str) -> dict[str, Any]:
    """The sidecar we write next to an exported file, for ingest to pick up.

    `date` is the key the capture-date reader already looks for, so the export
    date survives into the catalogue without any special casing.
    """
    return {
        "photos_uuid": photo.uuid,
        "original_filename": photo.original_filename,
        "date": photo.date.isoformat() if photo.date else None,
        "device": device_of(photo),
        "source": "photos",
        "exported_at": exported_at,
    }


def export_one(photo: Any, destination_dir: Path, exported_at: str) -> ExportedItem | None:
    """Export one screenshot and write its sidecar. Returns None if nothing came out."""
    import json  # noqa: PLC0415

    from osxphotos.photoexporter import ExportOptions, PhotoExporter  # noqa: PLC0415

    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = naming.safe_filename(photo.original_filename or f"{photo.uuid}.png")

    options = ExportOptions(
        download_missing=True,
        use_photokit=True,
        overwrite=False,
        increment=True,
        timeout=EXPORT_TIMEOUT_SECONDS,
    )
    results = PhotoExporter(photo).export(destination_dir, filename, options=options)

    written = list(results.exported) or list(results.new)
    if not written:
        return None

    path = Path(written[0])
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps(sidecar_content(photo, exported_at), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return ExportedItem(photo.uuid, photo.original_filename or path.name, path, sidecar)


def exported_uuids(inbox: Path) -> set[str]:
    """UUIDs of items already sitting in the inbox, read from their sidecars."""
    import json  # noqa: PLC0415

    found: set[str] = set()
    if not inbox.is_dir():
        return found
    for sidecar in inbox.glob("*.json"):
        if sidecar.name.startswith("."):
            continue  # an AppleDouble companion the drive left, not a sidecar
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("photos_uuid"):
            found.add(str(data["photos_uuid"]))
    return found


def delete_assets(uuids: list[str]) -> int:
    """Move the given assets to Photos' Recently Deleted. Returns how many went.

    This goes through PhotoKit, so macOS shows its own confirmation dialogue and
    the items land in Recently Deleted rather than disappearing outright.
    """
    require_photos()
    import Photos  # noqa: PLC0415
    from osxphotos.photokit import PhotoLibrary  # noqa: PLC0415

    if not uuids:
        return 0

    library = PhotoLibrary()
    assets = library.fetch_uuid_list(uuids)
    phassets = [asset.phasset for asset in assets if getattr(asset, "phasset", None)]
    if not phassets:
        return 0

    def changes() -> None:
        Photos.PHAssetChangeRequest.deleteAssets_(phassets)

    shared = Photos.PHPhotoLibrary.sharedPhotoLibrary()
    success, error = shared.performChangesAndWait_error_(changes, None)
    if not success:
        if error is not None and error.code() == PHOTOS_USER_CANCELLED:
            raise DeletionDeclinedError("The confirmation dialogue was dismissed.")
        raise PhotosUnavailableError(f"Photos refused the deletion: {error}")
    return len(phassets)


def chunked(items: list[str], size: int = DEFAULT_DELETE_CHUNK) -> list[list[str]]:
    """Split a list into chunks. A size of zero or less means one chunk of everything.

    Each chunk costs the person at the keyboard one confirmation dialogue, so the
    number of chunks is the number of times they are interrupted.
    """
    if not items:
        return []
    if size <= 0:
        return [list(items)]
    return [items[i : i + size] for i in range(0, len(items), size)]


def now_iso() -> str:
    """Seconds-precision local timestamp."""
    return datetime.now().replace(microsecond=0).isoformat()  # noqa: DTZ005


# ---------------------------------------------------------------------------
# The result line the export prints for the backlog loop to read
# ---------------------------------------------------------------------------

#: The export prints one machine-readable line so a caller can tell the two
#: reasons for an empty batch apart: a year that is genuinely drained, and a year
#: whose downloads all failed. They look identical from the number of files that
#: landed, and treating the second as the first abandons a year with thousands of
#: screenshots still in it.
RESULT_PREFIX = "export-result"
RESULT_FIELDS = ("detected", "new", "pending", "exported", "failed")


@dataclass(frozen=True, slots=True)
class ExportResult:
    """What one export call found and managed to write."""

    detected: int = 0
    new: int = 0
    pending: int = 0
    exported: int = 0
    failed: int = 0

    @property
    def drained(self) -> bool:
        """True when Photos has nothing left for this year, whatever was written."""
        return self.new == 0

    def format(self) -> str:
        """The one line a caller parses."""
        fields = " ".join(f"{name}={getattr(self, name)}" for name in RESULT_FIELDS)
        return f"{RESULT_PREFIX} {fields}"


def parse_result(text: str) -> ExportResult | None:
    """Read the result line out of an export's output, or None if it is not there."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith(RESULT_PREFIX):
            continue
        values: dict[str, int] = {}
        for token in stripped[len(RESULT_PREFIX) :].split():
            name, _, raw = token.partition("=")
            if name in RESULT_FIELDS:
                try:
                    values[name] = int(raw)
                except ValueError:
                    return None
        return ExportResult(**values)
    return None


# ---------------------------------------------------------------------------
# Disk space, and why Photos stops handing anything over
# ---------------------------------------------------------------------------

#: Photos will not fetch an iCloud original onto a volume that is nearly full. It
#: refuses instantly, in milliseconds, with CloudPhotoLibraryErrorDomain 1005,
#: "Disk space is very low", so a whole batch fails inside the same second and no
#: amount of waiting changes it.
#:
#: Diagnosed on a real library: 404 such refusals in one hour at 4.1 GB free of
#: 228 GB. The 2195 screenshots that had downloaded earlier and the 2384 that
#: would not were identical in every property osxphotos reports - all missing, all
#: in cloud, none shared or hidden, all PNG - so it was never about the items. The
#: gate sat near 5 GB free, and a batch would briefly succeed right after a purge
#: freed a little space. The floor is set well above that so the run stops and
#: says so rather than grinding all night.
CLOUD_LOW_DISK_CODE = 1005

#: Free space Photos needs on its own volume. The knife-edge sat near 5 GB, so
#: this is the floor that actually matters once the library moves to a drive.
PHOTOS_MIN_FREE_GB = 6.0

#: Free space the library root needs. Only really about the root sharing a volume
#: with Photos; a roomy external drive is not the constraint.
DEFAULT_MIN_FREE_GB = 10.0

#: A separate root volume with this much free is not worth checking against the
#: root floor at all: the pictures are not going there.
ROOT_FLOOR_SKIP_GB = 50.0

#: Within this much of a floor, a batch of failures is the disk rather than the
#: network, so backing off is the wrong answer.
DISK_HEADROOM_GB = 2.0

BYTES_PER_GB = 1024**3

DEFAULT_LIBRARY = Path.home() / "Pictures" / "Photos Library.photoslibrary"


def free_gb(path: Path) -> float:
    """Free space on the volume holding a path, in gigabytes."""
    return shutil.disk_usage(path).free / BYTES_PER_GB


def nearest_existing(path: Path) -> Path:
    """The given path, or the closest parent that exists, so it can be stat'd."""
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


@lru_cache(maxsize=1)
def library_path() -> Path:
    """Where the Photos library lives, as osxphotos reports it."""
    if sys.platform != "darwin":
        return DEFAULT_LIBRARY
    try:
        from osxphotos.utils import get_system_library_path  # noqa: PLC0415

        found = get_system_library_path()
    except Exception:  # noqa: BLE001 - any failure just means the usual location
        return DEFAULT_LIBRARY
    return Path(found) if found else DEFAULT_LIBRARY


def on_one_volume(first: Path, second: Path) -> bool:
    """True when two paths sit on the same filesystem."""
    try:
        return nearest_existing(first).stat().st_dev == nearest_existing(second).stat().st_dev
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class DiskCheck:
    """Free space on both volumes that matter, against the floor each one needs.

    Once the library moves to an external drive these come apart: the drive has a
    terabyte and Photos, still on the internal disk, has eight gigabytes. Only the
    second one decides whether anything downloads.
    """

    root_gb: float
    photos_gb: float
    same_volume: bool
    root_floor: float = DEFAULT_MIN_FREE_GB
    photos_floor: float = PHOTOS_MIN_FREE_GB

    @property
    def root_floor_applies(self) -> bool:
        """A roomy volume of its own is not a constraint worth enforcing."""
        return self.same_volume or self.root_gb <= ROOT_FLOOR_SKIP_GB

    @property
    def blocked(self) -> str | None:
        """Which volume is too full to work with, or None when both are fine."""
        if self.photos_gb < self.photos_floor:
            return "photos"
        if self.root_floor_applies and self.root_gb < self.root_floor:
            return "root"
        return None

    @property
    def near_floor(self) -> bool:
        """True when a volume is close enough that failures are about space."""
        if self.photos_gb <= self.photos_floor + DISK_HEADROOM_GB:
            return True
        return self.root_floor_applies and self.root_gb <= self.root_floor + DISK_HEADROOM_GB

    def summary(self) -> str:
        """Both figures, for a batch line."""
        if self.same_volume:
            return f"{self.root_gb:.1f} GB free"
        return (
            f"{self.root_gb:.1f} GB free on the library volume, "
            f"{self.photos_gb:.1f} GB on the Photos volume"
        )


def build_check(
    root: Path,
    *,
    photos_library: Path | None = None,
    free_bytes: Callable[[Path], int] | None = None,
    root_floor: float = DEFAULT_MIN_FREE_GB,
) -> DiskCheck:
    """Measure both volumes. ``free_bytes`` is injectable so the rules can be tested."""
    library = Path(photos_library) if photos_library else library_path()
    measured = nearest_existing(library)
    read = free_bytes or (lambda path: shutil.disk_usage(path).free)
    return DiskCheck(
        root_gb=read(Path(root)) / BYTES_PER_GB,
        photos_gb=read(measured) / BYTES_PER_GB,
        same_volume=on_one_volume(root, measured),
        root_floor=root_floor,
    )


def low_disk_message(check: DiskCheck) -> str:
    """Why nothing is downloading, and what to do about it."""
    needs = f"Photos needs {check.photos_floor:.0f} GB free on its own volume"
    if check.root_floor_applies:
        needs += f", and the library volume needs {check.root_floor:.0f} GB"
    return (
        f"{check.summary()}. {needs}. Photos refuses to download iCloud originals onto a "
        f"nearly full disk: it replies in milliseconds with CloudPhotoLibrary error "
        f"{CLOUD_LOW_DISK_CODE}, 'Disk space is very low', so every export fails in the same "
        "second. Waiting does not help, and nor does a smaller batch. Free up disk space on "
        "this Mac, then run it again."
    )
