"""Compare candidate models over a sample of real screenshots."""

from __future__ import annotations

import json
import random
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vex import classify, config, db, openrouter, taxonomy
from vex.openrouter import OpenRouterClient

#: Shortlisted from the live OpenRouter model list. See docs/model-choice.md.
DEFAULT_CANDIDATES: tuple[str, ...] = (
    "google/gemini-2.5-flash-lite",
    "google/gemini-3.1-flash-lite",
    "mistralai/mistral-small-3.2-24b-instruct",
    "google/gemma-3-12b-it",
    "qwen/qwen3-vl-8b-instruct",
)

TIERS = ("text", "vision")

#: Projections printed alongside the measured cost per image.
PROJECTION_SIZES = (1_700, 15_000)


@dataclass(slots=True)
class Trial:
    """One model, one tier, one screenshot."""

    hash: str
    model: str
    tier: str
    category: str | None = None
    confidence: float | None = None
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the model answered with a usable classification."""
        return self.error is None and self.category is not None


@dataclass(slots=True)
class ModelReport:
    """Aggregated results for one model and tier."""

    model: str
    tier: str
    attempted: int = 0
    succeeded: int = 0
    agreement: float | None = None
    median_latency_s: float = 0.0
    mean_cost_usd: float = 0.0
    junk_rate: float = 0.0
    categories: dict[str, int] = field(default_factory=dict)

    def projection(self, images: int) -> float:
        """Estimated dollar cost of classifying this many images with this model and tier."""
        return self.mean_cost_usd * images


@dataclass(slots=True)
class BakeoffResult:
    """Everything a bakeoff run produced."""

    started_at: str
    sample: list[str]
    models: list[str]
    trials: list[Trial]
    reports: list[ModelReport]
    consensus: dict[str, str]
    output_path: Path | None = None


def sample_rows(
    conn: sqlite3.Connection, root: Path, *, size: int, seed: int | None = None
) -> list[classify.RowInput]:
    """A random sample of rows that have both OCR text and a thumbnail on disk."""
    sql = (
        "SELECT hash, original_name, source, width, height, ocr_text FROM screenshots "
        "WHERE ocr_text IS NOT NULL AND trim(ocr_text) != ''"
    )
    candidates = [
        classify.RowInput(
            hash=row["hash"],
            original_name=row["original_name"] or "",
            source=row["source"] or "",
            width=row["width"],
            height=row["height"],
            ocr_text=row["ocr_text"] or "",
            thumb=config.thumb_path(root, row["hash"]),
        )
        for row in conn.execute(sql).fetchall()
    ]
    candidates = [row for row in candidates if row.has_thumbnail()]
    rng = random.Random(seed)  # noqa: S311 - sampling, not security
    rng.shuffle(candidates)
    return candidates[:size]


def _run_trial(
    client: OpenRouterClient,
    row: classify.RowInput,
    model: str,
    tier: str,
) -> tuple[Trial, classify.CacheWrite | None]:
    trial = Trial(hash=row.hash, model=model, tier=tier)
    outcome = classify.RowOutcome(hash=row.hash, tier=tier)
    started = time.monotonic()
    try:
        result = classify.call_tier(client, row, mode=tier, model=model, cache={}, outcome=outcome)
    except (openrouter.OpenRouterError, OSError) as exc:
        trial.latency_s = time.monotonic() - started
        trial.error = str(exc)
        return trial, None

    trial.latency_s = time.monotonic() - started
    trial.category = result.category
    trial.confidence = result.confidence
    trial.caption = result.caption
    trial.tags = result.tags
    trial.cost_usd = outcome.cost_usd
    write = outcome.cache_writes[0] if outcome.cache_writes else None
    if write is not None:
        trial.input_tokens = write.input_tokens
        trial.output_tokens = write.output_tokens
    return trial, write


def build_reports(trials: list[Trial], consensus: dict[str, str]) -> list[ModelReport]:
    """Aggregate per model and tier, scoring agreement against the consensus category."""
    grouped: dict[tuple[str, str], list[Trial]] = defaultdict(list)
    for trial in trials:
        grouped[(trial.model, trial.tier)].append(trial)

    reports: list[ModelReport] = []
    for (model, tier), items in sorted(grouped.items()):
        good = [item for item in items if item.ok]
        agreed = [item for item in good if consensus.get(item.hash) == item.category]
        latencies = [item.latency_s for item in good if item.latency_s > 0]
        reports.append(
            ModelReport(
                model=model,
                tier=tier,
                attempted=len(items),
                succeeded=len(good),
                agreement=(len(agreed) / len(good)) if good else None,
                median_latency_s=statistics.median(latencies) if latencies else 0.0,
                mean_cost_usd=(sum(item.cost_usd for item in good) / len(good)) if good else 0.0,
                junk_rate=(
                    sum(1 for item in good if item.category == "junk") / len(good) if good else 0.0
                ),
                categories=dict(Counter(item.category for item in good if item.category)),
            )
        )
    return reports


def build_consensus(trials: list[Trial]) -> dict[str, str]:
    """The most commonly chosen category per screenshot, used as the yardstick."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for trial in trials:
        if trial.ok and trial.category:
            votes[trial.hash][trial.category] += 1
    return {file_hash: counter.most_common(1)[0][0] for file_hash, counter in votes.items()}


def usable_tiers(client: OpenRouterClient, model: str, tiers: tuple[str, ...]) -> list[str]:
    """Which tiers this model can actually serve, from its advertised modalities."""
    pricing = client.pricing_for(model)
    chosen = []
    for tier in tiers:
        if tier == "vision" and pricing.model and not pricing.supports_image_input:
            continue
        chosen.append(tier)
    return chosen


def run(
    conn: sqlite3.Connection,
    rows: list[classify.RowInput],
    *,
    models: tuple[str, ...],
    client: OpenRouterClient,
    tiers: tuple[str, ...] = TIERS,
    concurrency: int = 4,
    progress: Any = None,
) -> BakeoffResult:
    """Run every model over every sampled screenshot in every tier it supports."""
    jobs: list[tuple[classify.RowInput, str, str]] = []
    for model in models:
        for tier in usable_tiers(client, model, tiers):
            jobs.extend((row, model, tier) for row in rows)

    trials: list[Trial] = []
    writes: list[classify.CacheWrite] = []

    def work(job: tuple[classify.RowInput, str, str]) -> tuple[Trial, classify.CacheWrite | None]:
        row, model, tier = job
        return _run_trial(client, row, model, tier)

    workers = max(1, min(concurrency, len(jobs) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (trial, write) in enumerate(pool.map(work, jobs), start=1):
            trials.append(trial)
            if write is not None:
                writes.append(write)
            if progress is not None:
                progress(index, len(jobs), trial)

    for write in writes:
        db.cache_classification(
            conn,
            write.hash,
            write.model,
            write.mode,
            write.response_json,
            input_tokens=write.input_tokens,
            output_tokens=write.output_tokens,
            cost_usd=write.cost_usd,
            created_at=write.created_at,
        )

    consensus = build_consensus(trials)
    return BakeoffResult(
        started_at=datetime.now().replace(microsecond=0).isoformat(),  # noqa: DTZ005
        sample=[row.hash for row in rows],
        models=list(models),
        trials=trials,
        reports=build_reports(trials, consensus),
        consensus=consensus,
    )


def write_dump(root: Path, result: BakeoffResult) -> Path:
    """Save the whole run under <root>/bakeoff/ so the numbers can be revisited."""
    directory = Path(root) / "bakeoff"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.replace(":", "").replace("-", "")
    path = directory / f"bakeoff-{stamp}.json"
    payload = {
        "started_at": result.started_at,
        "models": result.models,
        "sample": result.sample,
        "taxonomy": list(taxonomy.active().names),
        "consensus": result.consensus,
        "reports": [asdict(report) for report in result.reports],
        "trials": [asdict(trial) for trial in result.trials],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    result.output_path = path
    return path
