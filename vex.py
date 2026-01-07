#!/usr/bin/env python3
"""
vex - Vision extraction for screenshots using Claude.

Batch-process screenshots through Claude's vision API to extract
structured, searchable content from images.
"""

import argparse
import base64
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

__version__ = "0.1.0"

load_dotenv()

# Supported image formats (PIL-compatible)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Model configurations (pricing per million tokens, input)
MODELS = {
    "claude-haiku-4-5-20251001": {"name": "Haiku 4.5", "cost_per_mtok": 1.0},
    "claude-sonnet-4-5-20250929": {"name": "Sonnet 4.5", "cost_per_mtok": 3.0},
    "claude-opus-4-5-20251101": {"name": "Opus 4.5", "cost_per_mtok": 5.0},
}
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# The extraction prompt - adapts to content type
EXTRACTION_PROMPT = """Analyse this screenshot and extract all useful information.

Determine what type of content this is and extract accordingly:

**RECIPE**: Title, full ingredients with quantities, complete method steps, servings, cook time, source.

**CODE/TECHNICAL**: Language/technology, what it does, extract visible code verbatim, any commands shown.

**ERROR MESSAGE**: Exact error text, platform/app, error codes, stack trace, likely cause if obvious.

**ARTICLE/TEXT**: Key points summary, notable quotes, author, source, date if visible.

**CHAT/SOCIAL**: Who said what, platform, context, any links or key info shared.

**DOCUMENT**: Document type, key content, dates, names, reference numbers, important details.

**MEME/IMAGE**: Describe the image, transcribe all text overlays, explain the joke/context if relevant.

**UI/APP**: App name, what screen/feature is shown, any settings or data displayed.

**SHOPPING/PRODUCT**: Item name, price, store/site, any specs or details, link if visible.

**MAP/LOCATION**: Place name, full address, coordinates if shown, directions, context for why saved.

**RECEIPT/TRANSACTION**: Merchant, amount, date, order/confirmation number, items if listed.

**BOOKING/EVENT**: Event name, date, time, venue/location, confirmation number, ticket details.

**CONTACT INFO**: Name, phone, email, company, role, address - extract all contact details.

**MUSIC/MEDIA**: Song title, artist, album, playlist name, platform.

**QUOTE/INSPIRATION**: The exact quote, attribution/author, source.

**HEALTH/FITNESS**: Metrics shown, values, dates, app name, what's being tracked.

**SETTINGS/CONFIG**: App name, what settings are shown, current values, why this might matter.

Respond in JSON format:
{
    "type": "recipe|code|error|article|chat|document|meme|photo|ui|shopping|map|receipt|booking|contact|music|quote|health|settings|other",
    "category": "food|tech|work|social|finance|health|travel|entertainment|reference|personal|shopping|other",
    "summary": "1-2 sentence description of what this screenshot contains",
    "extracted_content": "The main extracted content. Be thorough - this is the primary value.",
    "tags": ["up to 5 relevant tags for searching"],
    "source": "source website/app if identifiable, null otherwise"
}

Be thorough with extracted_content. For recipes, include everything to cook it. For code, include the actual code. For receipts, include all transaction details. Extract everything useful for searching later."""


@dataclass
class ProcessingResult:
    """Result of processing a single screenshot."""

    path: str
    type: str
    category: str
    summary: str
    extracted_content: str
    tags: list[str]
    source: str | None
    processed_at: str
    model: str
    tokens_used: int
    error: str | None = None


@dataclass
class ProcessingStats:
    """Statistics for a processing run."""

    total: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        return self.processed / self.elapsed * 60 if self.elapsed > 0 else 0


# =============================================================================
# Image Processing
# =============================================================================


def get_media_type(path: Path) -> str:
    """Get MIME type for an image file."""
    types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return types.get(path.suffix.lower(), "image/png")


def prepare_image(path: Path, max_dimension: int = 1568) -> tuple[str, str]:
    """
    Prepare an image for Claude API.

    Args:
        path: Path to image file
        max_dimension: Maximum width/height (Claude recommends 1568px)

    Returns:
        Tuple of (base64_data, media_type)
    """
    with Image.open(path) as img:
        # Resize if needed
        if img.width > max_dimension or img.height > max_dimension:
            ratio = min(max_dimension / img.width, max_dimension / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed (handles RGBA, P, LA, L modes)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Encode
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        return base64.standard_b64encode(buffer.read()).decode("utf-8"), "image/jpeg"


def find_images(directory: Path, recursive: bool = True) -> list[Path]:
    """Find all supported image files in a directory."""
    pattern = "**/*" if recursive else "*"
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(directory.glob(f"{pattern}{ext}"))
        files.extend(directory.glob(f"{pattern}{ext.upper()}"))
    return sorted(set(files))


# =============================================================================
# Response Parsing
# =============================================================================


def parse_response(content: str) -> dict:
    """Parse JSON from Claude's response, handling markdown code blocks."""
    text = content

    # Strip markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        data = json.loads(text.strip())
        # Ensure extracted_content is a string (Claude sometimes returns dicts)
        if isinstance(data.get("extracted_content"), dict):
            data["extracted_content"] = json.dumps(data["extracted_content"], indent=2)
        return data
    except json.JSONDecodeError:
        return {
            "type": "other",
            "category": "other",
            "summary": content[:200],
            "extracted_content": content,
            "tags": [],
            "source": None,
        }


# =============================================================================
# Progress Tracking
# =============================================================================


def load_progress(output_file: Path) -> set[str]:
    """Load already-processed file paths from output file."""
    processed = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed.add(result["path"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return processed


def estimate_cost(tokens: int, model: str) -> float:
    """Estimate cost in USD for token usage."""
    cost_per_mtok = MODELS.get(model, {}).get("cost_per_mtok", 3.0)
    return tokens * cost_per_mtok / 1_000_000


# =============================================================================
# Processing
# =============================================================================


def process_single(
    client: anthropic.Anthropic,
    path: Path,
    model: str,
) -> ProcessingResult:
    """Process a single image with Claude."""
    try:
        image_data, media_type = prepare_image(path)

        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }
            ],
        )

        data = parse_response(response.content[0].text)
        tokens = response.usage.input_tokens + response.usage.output_tokens

        return ProcessingResult(
            path=str(path),
            type=data.get("type", "other"),
            category=data.get("category", "other"),
            summary=data.get("summary", ""),
            extracted_content=data.get("extracted_content", ""),
            tags=data.get("tags", []),
            source=data.get("source"),
            processed_at=datetime.now().isoformat(),
            model=model,
            tokens_used=tokens,
        )

    except Exception as e:
        return ProcessingResult(
            path=str(path),
            type="error",
            category="error",
            summary="",
            extracted_content="",
            tags=[],
            source=None,
            processed_at=datetime.now().isoformat(),
            model=model,
            tokens_used=0,
            error=str(e),
        )


def process_realtime(
    directory: Path,
    output_file: Path,
    model: str,
    rate_limit: float,
    resume: bool,
    limit: int | None,
) -> ProcessingStats:
    """Process images one at a time with rate limiting."""
    client = anthropic.Anthropic()
    stats = ProcessingStats()

    images = find_images(directory)
    stats.total = len(images)

    if not images:
        print(f"No images found in {directory}")
        return stats

    # Filter already processed
    processed = load_progress(output_file) if resume else set()
    stats.skipped = len(processed)
    to_process = [p for p in images if str(p) not in processed]

    if limit:
        to_process = to_process[:limit]

    if not to_process:
        print("All images already processed")
        return stats

    print(f"Found {stats.total} images, {len(to_process)} to process")
    print(f"Output: {output_file}")
    print(f"Model: {MODELS[model]['name']}")
    print()

    with open(output_file, "a") as f:
        for path in tqdm(to_process, desc="Processing"):
            result = process_single(client, path, model)

            if result.error:
                stats.errors += 1
                tqdm.write(f"Error: {path.name}: {result.error}")
            else:
                stats.processed += 1
                stats.total_tokens += result.tokens_used

            f.write(json.dumps(asdict(result)) + "\n")
            f.flush()

            if rate_limit > 0:
                time.sleep(rate_limit)

    return stats


# =============================================================================
# Batch Processing
# =============================================================================


def create_batch_request(path: Path, custom_id: str, model: str) -> dict:
    """Create a batch API request for a single image."""
    image_data, media_type = prepare_image(path)
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }
            ],
        },
    }


def cmd_batch(args: argparse.Namespace) -> None:
    """Submit images for batch processing."""
    require_api_key()
    client = anthropic.Anthropic()

    images = find_images(args.directory)
    if not images:
        print(f"No images found in {args.directory}")
        sys.exit(1)

    # Filter already processed
    if not args.no_resume and args.output.exists():
        processed = load_progress(args.output)
        images = [p for p in images if str(p) not in processed]

    if args.limit:
        images = images[: args.limit]

    if not images:
        print("All images already processed")
        return

    print(f"Preparing {len(images)} images for batch processing...")

    # Build requests with custom_id -> path mapping
    requests = []
    path_mapping = {}
    failed = 0
    for idx, path in enumerate(tqdm(images, desc="Preparing")):
        try:
            # Use index-based custom_id (batch API requires ^[a-zA-Z0-9_-]{1,64}$)
            custom_id = f"img_{idx:06d}"
            path_mapping[custom_id] = str(path)
            requests.append(create_batch_request(path, custom_id, args.model))
        except Exception as e:
            failed += 1
            tqdm.write(f"Skipped {path.name}: {e}")

    if not requests:
        print("No valid images to process")
        sys.exit(1)

    if failed:
        print(f"Skipped {failed} images due to errors")

    print(f"Submitting batch of {len(requests)} requests...")

    batch = client.messages.batches.create(requests=requests)

    print(f"\nBatch submitted: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"\nCheck progress: vex status {batch.id}")
    print(f"Get results:    vex results {batch.id}")

    # Save batch info with path mapping
    info_file = Path(f".vex-batch-{batch.id}.json")
    with open(info_file, "w") as f:
        json.dump(
            {
                "batch_id": batch.id,
                "model": args.model,
                "count": len(requests),
                "submitted_at": datetime.now().isoformat(),
                "directory": str(args.directory),
                "output_file": str(args.output),
                "path_mapping": path_mapping,
            },
            f,
            indent=2,
        )
    print(f"Batch info saved: {info_file}")


def cmd_status(args: argparse.Namespace) -> None:
    """Check batch processing status."""
    require_api_key()
    client = anthropic.Anthropic()

    batch = client.messages.batches.retrieve(args.batch_id)

    print(f"Batch: {batch.id}")
    print(f"Status: {batch.processing_status}")

    if hasattr(batch, "request_counts") and batch.request_counts:
        c = batch.request_counts
        print(f"Progress: {c.succeeded} done, {c.processing} processing, {c.errored} errors")

    if batch.processing_status == "ended":
        print(f"\nReady! Run: vex results {batch.id}")


def cmd_results(args: argparse.Namespace) -> None:
    """Download batch results."""
    require_api_key()
    client = anthropic.Anthropic()

    batch = client.messages.batches.retrieve(args.batch_id)
    if batch.processing_status != "ended":
        print(f"Batch not complete. Status: {batch.processing_status}")
        sys.exit(1)

    # Load path mapping from batch info file
    info_file = Path(f".vex-batch-{args.batch_id}.json")
    path_mapping = {}
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
            path_mapping = info.get("path_mapping", {})
    else:
        print(f"Warning: Batch info file not found ({info_file})")
        print("Results will use custom_id instead of original paths")

    print(f"Downloading results to {args.output}...")

    count = 0
    errors = 0

    with open(args.output, "a") as f:
        for result in client.messages.batches.results(args.batch_id):
            count += 1
            custom_id = result.custom_id
            # Map custom_id back to original path
            path = path_mapping.get(custom_id, custom_id)

            if result.result.type == "succeeded":
                response = result.result.message
                data = parse_response(response.content[0].text)
                tokens = response.usage.input_tokens + response.usage.output_tokens

                record = ProcessingResult(
                    path=path,
                    type=data.get("type", "other"),
                    category=data.get("category", "other"),
                    summary=data.get("summary", ""),
                    extracted_content=data.get("extracted_content", ""),
                    tags=data.get("tags", []),
                    source=data.get("source"),
                    processed_at=datetime.now().isoformat(),
                    model=response.model,
                    tokens_used=tokens,
                )
            else:
                errors += 1
                error_msg = getattr(result.result, "error", "Unknown error")
                record = ProcessingResult(
                    path=path,
                    type="error",
                    category="error",
                    summary="",
                    extracted_content="",
                    tags=[],
                    source=None,
                    processed_at=datetime.now().isoformat(),
                    model="",
                    tokens_used=0,
                    error=str(error_msg),
                )

            f.write(json.dumps(asdict(record)) + "\n")

    print(f"Downloaded {count} results ({errors} errors)")
    print(f"Saved to {args.output}")


# =============================================================================
# Organise
# =============================================================================


def cmd_organise(args: argparse.Namespace) -> None:
    """Sort processed files into category folders."""
    if not args.results_file.exists():
        print(f"Results file not found: {args.results_file}")
        sys.exit(1)

    args.target.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0

    with open(args.results_file) as f:
        for line in f:
            try:
                result = json.loads(line)
                source = Path(result["path"])
                category = result.get("category", "other")

                if not source.exists():
                    skipped += 1
                    continue

                # Create category folder and determine destination
                cat_dir = args.target / category
                cat_dir.mkdir(exist_ok=True)
                dest = cat_dir / source.name

                # Handle duplicates
                if dest.exists():
                    stem, suffix = source.stem, source.suffix
                    counter = 1
                    while dest.exists():
                        dest = cat_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                if args.copy:
                    shutil.copy2(source, dest)
                else:
                    shutil.move(source, dest)
                moved += 1

            except (json.JSONDecodeError, KeyError):
                skipped += 1

    action = "Copied" if args.copy else "Moved"
    print(f"{action} {moved} files to {args.target}")
    if skipped:
        print(f"Skipped {skipped} (missing or invalid)")


# =============================================================================
# CLI
# =============================================================================


def require_api_key() -> None:
    """Exit if API key is not set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Set in .env file or: export ANTHROPIC_API_KEY=...")
        sys.exit(1)


def cmd_process(args: argparse.Namespace) -> None:
    """Process images (realtime or batch mode)."""
    require_api_key()

    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    # Dry run
    if args.dry_run:
        images = find_images(args.directory)
        print(f"Found {len(images)} images:")
        for img in images[:20]:
            print(f"  {img}")
        if len(images) > 20:
            print(f"  ... and {len(images) - 20} more")
        return

    # Batch mode
    if args.batch:
        cmd_batch(args)
        return

    # Realtime mode
    stats = process_realtime(
        directory=args.directory,
        output_file=args.output,
        model=args.model,
        rate_limit=args.rate,
        resume=not args.no_resume,
        limit=args.limit,
    )

    # Summary
    print()
    print("=" * 50)
    print(f"Processed: {stats.processed}")
    print(f"Errors: {stats.errors}")
    print(f"Skipped: {stats.skipped}")
    print(f"Tokens: {stats.total_tokens:,}")
    print(f"Time: {stats.elapsed:.1f}s ({stats.rate:.1f}/min)")

    if stats.total_tokens > 0:
        cost = estimate_cost(stats.total_tokens, args.model)
        print(f"Est. cost: ${cost:.2f} USD")


def main() -> None:
    """Main entry point."""
    # If first argument looks like a path (not a subcommand), insert 'process'
    subcommands = {"process", "status", "results", "organise", "-h", "--help", "--version"}
    if len(sys.argv) > 1 and sys.argv[1] not in subcommands:
        # Check if it could be a directory path
        if not sys.argv[1].startswith("-"):
            sys.argv.insert(1, "process")

    parser = argparse.ArgumentParser(
        prog="vex",
        description="Vision extraction - batch process screenshots with Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    # Process command (default)
    proc = subparsers.add_parser(
        "process",
        help="Process images in a directory",
        description="Extract content from screenshots using Claude's vision API",
    )
    proc.add_argument("directory", type=Path, help="Directory containing images")
    proc.add_argument("-o", "--output", type=Path, default=Path("vex-results.jsonl"))
    proc.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()))
    proc.add_argument("-r", "--rate", type=float, default=0.5, help="Seconds between requests")
    proc.add_argument("-l", "--limit", type=int, help="Max images to process")
    proc.add_argument("--batch", action="store_true", help="Use batch API (50%% off)")
    proc.add_argument("--no-resume", action="store_true", help="Ignore previous progress")
    proc.add_argument("--dry-run", action="store_true", help="List files only")
    proc.set_defaults(func=cmd_process)

    # Status command
    stat = subparsers.add_parser("status", help="Check batch status")
    stat.add_argument("batch_id", help="Batch ID")
    stat.set_defaults(func=cmd_status)

    # Results command
    res = subparsers.add_parser("results", help="Download batch results")
    res.add_argument("batch_id", help="Batch ID")
    res.add_argument("-o", "--output", type=Path, default=Path("vex-results.jsonl"))
    res.set_defaults(func=cmd_results)

    # Organise command
    org = subparsers.add_parser("organise", help="Sort files by category")
    org.add_argument("results_file", type=Path, help="Results JSONL file")
    org.add_argument("--target", type=Path, required=True, help="Target directory")
    org.add_argument("--copy", action="store_true", help="Copy instead of move")
    org.set_defaults(func=cmd_organise)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
