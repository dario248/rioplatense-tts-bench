"""The four segments of a clip, cut at the last transcribed word (SPEC 7.5).

    total = lead silence + speech + trailing audio + tail silence

Derived at report time from `audio_shape.jsonl` plus `transcripts.jsonl`, never a field in
`audio_shape`: that pass must not start depending on whether the ASR ran (SPEC 3).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from rtb.metrics.latency import percentile
from rtb.schema import GenerationRow, MetricRow

METRIC_NAME = "anatomy"


@dataclass
class Anatomy:
    """One clip cut in four, in seconds. The parts add up to `total_s`.

    `speech_s` is NOT `AudioShape.speech_s`: this one stops at the last transcribed word, that one
    at the last audible frame, and the difference is `trailing_audio_s`.
    """

    total_s: float
    lead_silence_s: float
    speech_s: float
    trailing_audio_s: float
    tail_silence_s: float

    @property
    def silence_s(self) -> float:
        """Silence around the clip: the padding figure, with the tail no longer inside it."""
        return self.lead_silence_s + self.tail_silence_s

    @property
    def non_speech_s(self) -> float:
        """Everything that is not the speech that was asked for."""
        return self.total_s - self.speech_s


def measure(shape: Mapping[str, object], last_word_end_s: float | None) -> Anatomy | None:
    """Cut one clip at its last transcribed word, or None when nothing says where that is."""
    if last_word_end_s is None:
        return None
    total = shape.get("total_s")
    start = shape.get("speech_start_s")
    end = shape.get("speech_end_s")
    if not (
        isinstance(total, (int, float))
        and isinstance(start, (int, float))
        and isinstance(end, (int, float))
    ):
        return None
    total_s, start_s, end_s = float(total), float(start), float(end)

    # Acotado al tramo con sonido: los timestamps de palabra tienen error de decenas de ms y sin
    # esto producen colas o habla negativas.
    cut = min(max(float(last_word_end_s), start_s), end_s)
    return Anatomy(
        total_s=total_s,
        lead_silence_s=start_s,
        speech_s=cut - start_s,
        trailing_audio_s=end_s - cut,
        tail_silence_s=max(0.0, total_s - end_s),
    )


def from_rows(
    shapes: Mapping[str, MetricRow],
    transcripts: Mapping[str, MetricRow],
) -> dict[str, Anatomy]:
    """The anatomy of every clip both passes could measure, by `generation_id`."""
    result = {}
    for generation_id, shape in shapes.items():
        transcript = transcripts.get(generation_id)
        if shape.error or transcript is None or transcript.error:
            continue
        last_word = transcript.values.get("last_word_end_s")
        anatomy = measure(
            shape.values, float(last_word) if isinstance(last_word, (int, float)) else None
        )
        if anatomy is not None:
            result[generation_id] = anatomy
    return result


@dataclass
class AnatomySummary:
    """The anatomy of one model, or of one slice of it."""

    label: str
    n: int
    threshold_s: float
    lead_silence_p50: float | None = None
    speech_p50: float | None = None
    trailing_p50: float | None = None
    trailing_p95: float | None = None
    tail_silence_p50: float | None = None
    non_speech_p50: float | None = None
    over_threshold: int = 0

    @property
    def percent_over(self) -> float:
        """Share of clips whose trailing audio is over the threshold `failures` flags at."""
        return 100.0 * self.over_threshold / self.n if self.n else 0.0


def summarize(
    anatomies: Iterable[Anatomy],
    threshold_s: float,
    label: str = "all",
) -> AnatomySummary:
    items = list(anatomies)
    summary = AnatomySummary(label=label, n=len(items), threshold_s=threshold_s)
    if not items:
        return summary
    trailing = [item.trailing_audio_s for item in items]
    summary.lead_silence_p50 = statistics.median(item.lead_silence_s for item in items)
    summary.speech_p50 = statistics.median(item.speech_s for item in items)
    summary.trailing_p50 = statistics.median(trailing)
    summary.trailing_p95 = percentile(trailing, 0.95)
    summary.tail_silence_p50 = statistics.median(item.tail_silence_s for item in items)
    summary.non_speech_p50 = statistics.median(item.non_speech_s for item in items)
    summary.over_threshold = sum(1 for value in trailing if value > threshold_s)
    return summary


def summarize_by_category(
    anatomies: Mapping[str, Anatomy],
    rows: Iterable[GenerationRow],
    category_of: Mapping[str, str],
    threshold_s: float,
) -> list[AnatomySummary]:
    """One summary per phrase category, for the rows of one manifest."""
    grouped: dict[str, list[Anatomy]] = {}
    for row in rows:
        anatomy = anatomies.get(row.generation_id)
        if row.warmup or anatomy is None:
            continue
        grouped.setdefault(category_of.get(row.phrase_id, "unknown"), []).append(anatomy)
    return [summarize(group, threshold_s, label) for label, group in sorted(grouped.items())]


def speech_seconds(anatomies: Mapping[str, Anatomy]) -> dict[str, float]:
    """Speech by `generation_id` — the quantity the duration screen judges instead of the file."""
    return {generation_id: item.speech_s for generation_id, item in anatomies.items()}
