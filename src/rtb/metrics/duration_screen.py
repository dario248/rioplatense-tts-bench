"""Duration triage, not the published failure rate (SPEC 7.5).

Judges the speech that was asked for when `anatomy` can say where it ends, and the whole file
otherwise; every row declares which quantity it judged. Its JSONL is not committed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from rtb.config import DurationThresholds, Phrase, Speaker, Thresholds
from rtb.metrics.sensitivity import SensitivityPoint, knob_label
from rtb.metrics.speaking_rate import speaking_rates
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, MetricRow

METRIC_NAME = "duration_screen"

# Barridos con los que se reporta la sensibilidad. No son umbrales candidatos: es un rango amplio a
# proposito, para que se vea si la tasa esta en una meseta o si depende del knob.
SENSITIVITY_ALLOWANCES_S = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5)
SENSITIVITY_RATIOS = (2.0, 2.5, 3.0, 3.5, 4.0)

# Que cantidad se juzga. Viaja en cada fila porque las dos tasas no son comparables entre si, y una
# tabla que no dice cual de las dos produjo el numero es una tabla que se lee mal.
BY_DURATION = "duration_s"
BY_SPEECH = "speech_s"


@dataclass
class DurationVerdict:
    expected_s: float
    ratio: float | None
    too_short: bool
    too_long: bool
    # "ratio" para frases normales, "absolute" para enunciados de pocas palabras: la tolerancia
    # multiplicativa crece con el texto y deja pasar un enunciado corto tres veces mas largo.
    rule: str


def judge_duration(
    measured_s: float,
    n_spoken_words: int,
    rate_s_per_word: float,
    thresholds: DurationThresholds,
) -> DurationVerdict:
    """Judge one measured length against what the real speakers take to say the same words."""
    expected = rate_s_per_word * n_spoken_words
    if n_spoken_words <= thresholds.short_utterance_max_words:
        ceiling = expected + thresholds.short_utterance_allowance_s
        return DurationVerdict(
            expected_s=expected,
            ratio=measured_s / expected if expected else None,
            too_short=measured_s < expected * thresholds.too_short_ratio,
            too_long=measured_s > ceiling,
            rule="absolute",
        )
    ratio = measured_s / expected if expected else None
    return DurationVerdict(
        expected_s=expected,
        ratio=ratio,
        too_short=ratio is not None and ratio < thresholds.too_short_ratio,
        too_long=ratio is not None and ratio > thresholds.too_long_ratio,
        rule="ratio",
    )


def screen_row(
    row: GenerationRow,
    phrase: Phrase | None,
    rate_s_per_word: float | None,
    thresholds: Thresholds,
    speech_s: float | None = None,
    measured: str = BY_DURATION,
) -> MetricRow:
    """One row per synthesis, saying what could be measured and what could not.

    Asking for speech on a row with no transcript leaves it unjudged rather than falling back to
    the duration: a rate that mixes the two quantities is a rate of nothing.
    """
    values: dict[str, object] = {
        "hard_error": row.error is not None,
        "too_short": False,
        "too_long": False,
        "measured": measured,
    }

    if row.error is not None:
        # Una falla dura no tiene audio que medir: se cuenta y se sigue.
        values["reason"] = "hard_error"
        return MetricRow(
            generation_id=row.generation_id,
            metric=METRIC_NAME,
            metric_version=thresholds.version,
            values=values,
        )

    measured_s = row.duration_s if measured == BY_DURATION else speech_s
    if phrase is None or rate_s_per_word is None or measured_s is None:
        missing = (
            "phrase not in the corpus"
            if phrase is None
            else "no speaking rate for this speaker"
            if rate_s_per_word is None
            else "row has no duration"
            if measured == BY_DURATION
            else "no transcript, so the speech duration is unknown"
        )
        # No se inventa un veredicto: la fila queda con error y no se cuenta como exito ni fallo.
        return MetricRow(
            generation_id=row.generation_id,
            metric=METRIC_NAME,
            metric_version=thresholds.version,
            error=f"cannot judge duration: {missing}",
        )

    verdict = judge_duration(
        measured_s, phrase.n_spoken_words, rate_s_per_word, thresholds.duration
    )
    values.update(
        {
            "measured_s": round(measured_s, 3),
            "duration_s": None if row.duration_s is None else round(row.duration_s, 3),
            "expected_duration_s": round(verdict.expected_s, 3),
            "ratio": None if verdict.ratio is None else round(verdict.ratio, 3),
            "n_spoken_words": phrase.n_spoken_words,
            "category": phrase.category.value,
            "rule": verdict.rule,
            "too_short": verdict.too_short,
            "too_long": verdict.too_long,
        }
    )
    return MetricRow(
        generation_id=row.generation_id,
        metric=METRIC_NAME,
        metric_version=thresholds.version,
        values=values,
    )


def screen(
    rows: Iterable[GenerationRow],
    phrases: Iterable[Phrase],
    speakers: Iterable[Speaker],
    thresholds: Thresholds,
    speech_s: Mapping[str, float] | None = None,
) -> list[MetricRow]:
    """Screen every non-warm-up row of a manifest.

    With `speech_s` the judged quantity is the speech; without it, the file, and the pass stays
    manifest-only.
    """
    by_id = {phrase.id: phrase for phrase in phrases}
    rates_by_speaker = speaking_rates(speakers)
    measured = BY_DURATION if speech_s is None else BY_SPEECH
    return [
        screen_row(
            row,
            by_id.get(row.phrase_id),
            rates_by_speaker.get(row.speaker_id),
            thresholds,
            speech_s=None if speech_s is None else speech_s.get(row.generation_id),
            measured=measured,
        )
        for row in rows
        # El warm-up es descartable por definicion y el ground truth es audio real: ninguno de los
        # dos entra en la tasa de un modelo.
        if not row.warmup and row.model != GROUND_TRUTH_MODEL
    ]


@dataclass
class ScreenRates:
    """Counts and percentages over the screened rows."""

    total: int = 0
    hard_error: int = 0
    too_short: int = 0
    too_long: int = 0
    unjudged: int = 0

    @property
    def any_flag(self) -> int:
        return self.hard_error + self.too_short + self.too_long

    def percent(self, count: int) -> float:
        return 100.0 * count / self.total if self.total else 0.0


def rates(metric_rows: Iterable[MetricRow]) -> ScreenRates:
    result = ScreenRates()
    for row in metric_rows:
        result.total += 1
        if row.error is not None:
            result.unjudged += 1
            continue
        if row.values.get("hard_error"):
            result.hard_error += 1
        if row.values.get("too_short"):
            result.too_short += 1
        if row.values.get("too_long"):
            result.too_long += 1
    return result


def rates_by_category(metric_rows: Iterable[MetricRow]) -> dict[str, ScreenRates]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in metric_rows:
        category = str(row.values.get("category", "unknown"))
        grouped.setdefault(category, []).append(row)
    return {category: rates(rows) for category, rows in sorted(grouped.items())}


def sensitivity(
    rows: Iterable[GenerationRow],
    phrases: Iterable[Phrase],
    speakers: Iterable[Speaker],
    thresholds: Thresholds,
    speech_s: Mapping[str, float] | None = None,
) -> list[SensitivityPoint]:
    """How much each rate moves when its threshold moves (SPEC 7.5)."""
    by_id = {phrase.id: phrase for phrase in phrases}
    rates_by_speaker = speaking_rates(speakers)

    def measured_of(row: GenerationRow) -> float | None:
        # La misma cantidad que se reporta: barrer la duracion del archivo mientras la tabla
        # publica el habla daria una sensibilidad de otra tasa.
        return row.duration_s if speech_s is None else speech_s.get(row.generation_id)

    judged = [
        (measured, by_id[row.phrase_id], rates_by_speaker[row.speaker_id])
        for row in rows
        if not row.warmup
        and row.error is None
        and row.phrase_id in by_id
        and row.speaker_id in rates_by_speaker
        and (measured := measured_of(row)) is not None
    ]
    short = [
        item
        for item in judged
        if item[1].n_spoken_words <= thresholds.duration.short_utterance_max_words
    ]
    regular = [
        item
        for item in judged
        if item[1].n_spoken_words > thresholds.duration.short_utterance_max_words
    ]

    points: list[SensitivityPoint] = []

    def count(items: list[tuple[float, Phrase, float]], knobs: DurationThresholds) -> int:
        return sum(
            judge_duration(measured, phrase.n_spoken_words, rate, knobs).too_long
            for measured, phrase, rate in items
        )

    in_use_allowance = thresholds.duration.short_utterance_allowance_s
    for allowance in sorted({*SENSITIVITY_ALLOWANCES_S, in_use_allowance}):
        knobs = thresholds.duration.model_copy(
            update={"short_utterance_allowance_s": allowance}
        )
        label = knob_label("allowance_s", allowance, in_use_allowance)
        points.append(
            SensitivityPoint("absolute", label, allowance, count(short, knobs), len(short))
        )

    in_use_ratio = thresholds.duration.too_long_ratio
    for ratio in sorted({*SENSITIVITY_RATIOS, in_use_ratio}):
        knobs = thresholds.duration.model_copy(update={"too_long_ratio": ratio})
        label = knob_label("too_long_ratio", ratio, in_use_ratio)
        points.append(SensitivityPoint("ratio", label, ratio, count(regular, knobs), len(regular)))
    return points
