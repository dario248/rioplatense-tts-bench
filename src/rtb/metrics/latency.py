"""Latency aggregation over manifest timings (SPEC 7.3).

Excludes warm-up and error rows, which are not latency measurements. Median and p95, never the
mean.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile. Deliberately simple: no interpolation, no numpy."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class LatencySummary:
    """Latency for one model, or for one slice of it."""

    label: str
    n: int
    ttfb_ms_median: float | None = None
    ttfb_ms_p95: float | None = None
    total_ms_median: float | None = None
    total_ms_p95: float | None = None
    rtf_median: float | None = None
    rtf_p95: float | None = None
    audio_seconds: float = 0.0
    # El hardware va SIEMPRE junto al numero: un RTF sin la placa que lo produjo no dice nada.
    hardware: set[str] = field(default_factory=set)
    streaming: set[bool] = field(default_factory=set)

    @property
    def comparable(self) -> bool:
        """Whether these rows can be aggregated into one figure at all.

        Two devices, or streaming mixed with whole-mode, are different measurements (SPEC 6.3).
        """
        return len(self.hardware) <= 1 and len(self.streaming) <= 1


def measurable(rows: Iterable[GenerationRow]) -> list[GenerationRow]:
    """Rows whose timings mean something: no warm-up, no failures, no real audio."""
    return [
        row
        for row in rows
        if not row.warmup and row.error is None and row.model != GROUND_TRUTH_MODEL
    ]


def summarize(rows: Iterable[GenerationRow], label: str = "all") -> LatencySummary:
    usable = measurable(rows)
    summary = LatencySummary(label=label, n=len(usable))
    if not usable:
        return summary

    ttfb = [row.ttfb_ms for row in usable if row.ttfb_ms is not None]
    total = [row.total_ms for row in usable if row.total_ms is not None]
    rtf = [row.rtf for row in usable if row.rtf is not None]

    if ttfb:
        summary.ttfb_ms_median = statistics.median(ttfb)
        summary.ttfb_ms_p95 = percentile(ttfb, 0.95)
    if total:
        summary.total_ms_median = statistics.median(total)
        summary.total_ms_p95 = percentile(total, 0.95)
    if rtf:
        summary.rtf_median = statistics.median(rtf)
        summary.rtf_p95 = percentile(rtf, 0.95)

    summary.audio_seconds = sum(row.duration_s or 0.0 for row in usable)
    summary.hardware = {f"{row.device}:{row.gpu or 'n/a'}" for row in usable}
    summary.streaming = {row.streaming for row in usable}
    return summary


def summarize_by(
    rows: Iterable[GenerationRow],
    key: str,
) -> list[LatencySummary]:
    """One summary per distinct value of a row attribute (`speaker_id`, `model`, ...)."""
    grouped: dict[str, list[GenerationRow]] = {}
    for row in rows:
        grouped.setdefault(str(getattr(row, key)), []).append(row)
    return [summarize(group, label) for label, group in sorted(grouped.items())]


def summarize_by_category(
    rows: Iterable[GenerationRow],
    category_of: dict[str, str],
) -> list[LatencySummary]:
    """One summary per phrase category, because `total_ms` is dominated by phrase length."""
    grouped: dict[str, list[GenerationRow]] = {}
    for row in rows:
        category = category_of.get(row.phrase_id, "unknown")
        grouped.setdefault(category, []).append(row)
    return [summarize(group, label) for label, group in sorted(grouped.items())]
