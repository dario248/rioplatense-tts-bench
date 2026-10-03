"""Markdown tables, ready to paste into the README.

Presentation only: everything here renders values computed elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rtb.metrics.anatomy import AnatomySummary
from rtb.metrics.duration_screen import ScreenRates
from rtb.metrics.failures import FailureRates
from rtb.metrics.latency import LatencySummary
from rtb.metrics.sensitivity import SensitivityPoint
from rtb.metrics.speaker_sim import PAIR_DESCRIPTIONS, Scale, SimSummary


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    body = list(rows)
    widths = [len(header) for header in headers]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(headers), separator, *(line(row) for row in body)])


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def latency_table(summaries: Iterable[LatencySummary]) -> str:
    rows = []
    for summary in summaries:
        hardware = ", ".join(sorted(summary.hardware)) or "—"
        if not summary.comparable:
            # Mezclar hardware o modos de streaming en una sola cifra la vuelve un numero sin
            # significado: se marca en la tabla, no se esconde.
            hardware += " ⚠ mixed"
        rows.append(
            [
                summary.label,
                str(summary.n),
                _ms(summary.total_ms_median),
                _ms(summary.total_ms_p95),
                _ms(summary.ttfb_ms_median),
                _ratio(summary.rtf_median),
                _ratio(summary.rtf_p95),
                hardware,
            ]
        )
    return markdown_table(
        ["slice", "n", "total ms p50", "total ms p95", "ttfb ms p50", "rtf p50", "rtf p95", "hw"],
        rows,
    )


def anatomy_table(summaries: Iterable[AnatomySummary]) -> str:
    """The clip cut in four, with the trailing audio as seconds instead of a boolean.

    Median and p95 both, because the median hides this defect (ANALYSIS A13).
    """
    body = list(summaries)
    threshold = body[0].threshold_s if body else 0.0
    rows = [
        [
            summary.label,
            str(summary.n),
            _seconds(summary.lead_silence_p50),
            _seconds(summary.speech_p50),
            _seconds(summary.trailing_p50),
            _seconds(summary.trailing_p95),
            f"{summary.percent_over:.1f}",
            _seconds(summary.tail_silence_p50),
            _seconds(summary.non_speech_p50),
        ]
        for summary in body
    ]
    return markdown_table(
        [
            "slice",
            "n",
            "lead p50",
            "speech p50",
            "trailing p50",
            "trailing p95",
            f"trailing > {threshold:g}s %",
            "end sil p50",
            "non-speech p50",
        ],
        rows,
    )


def screen_table(rates: dict[str, ScreenRates], overall: ScreenRates | None = None) -> str:
    rows = []
    for label, value in rates.items():
        rows.append(
            [
                label,
                str(value.total),
                f"{value.percent(value.hard_error):.1f}",
                f"{value.percent(value.too_short):.1f}",
                f"{value.percent(value.too_long):.1f}",
                f"{value.percent(value.any_flag):.1f}",
            ]
        )
    if overall is not None:
        rows.append(
            [
                "**all**",
                str(overall.total),
                f"{overall.percent(overall.hard_error):.1f}",
                f"{overall.percent(overall.too_short):.1f}",
                f"{overall.percent(overall.too_long):.1f}",
                f"{overall.percent(overall.any_flag):.1f}",
            ]
        )
    return markdown_table(
        ["category", "n", "hard error %", "too short %", "too long %", "any %"],
        rows,
    )


def intelligibility_table(
    rates: dict[str, FailureRates],
    ground_truth: FailureRates | None = None,
) -> str:
    """WER and CER by category, with real audio in the SAME table and not in a footnote.

    A WER of 6 % means nothing without the floor this pipeline scores on real speech (SPEC 7.1).
    """
    rows = []

    def line(label: str, value: FailureRates) -> list[str]:
        wers = sorted(value.wers)
        cers = sorted(value.cers)
        if not wers:
            return [label, str(value.scored), "—", "—", "—", str(value.unscorable)]
        median = len(wers) // 2
        p95 = min(len(wers) - 1, int(len(wers) * 0.95))
        return [
            label,
            str(value.scored),
            f"{wers[median]:.3f}",
            f"{wers[p95]:.3f}",
            f"{cers[median]:.3f}",
            str(value.unscorable),
        ]

    if ground_truth is not None:
        rows.append(line("**real audio**", ground_truth))
    for label, value in rates.items():
        rows.append(line(label, value))
    return markdown_table(
        ["slice", "n", "WER p50", "WER p95", "CER p50", "unscorable"],
        rows,
    )


def failure_table(
    rates: dict[str, FailureRates],
    overall: FailureRates | None = None,
    ground_truth: FailureRates | None = None,
) -> str:
    """The failure taxonomy, as percentages of the rows that could be classified."""
    rows = []

    def line(label: str, value: FailureRates) -> list[str]:
        return [
            label,
            str(value.scored),
            f"{value.percent(value.hard_error):.1f}",
            f"{value.percent(value.silent):.1f}",
            f"{value.percent(value.truncated):.1f}",
            f"{value.percent(value.looped):.1f}",
            f"{value.percent(value.hallucinated):.1f}",
            f"{value.percent(value.trailing_audio):.1f}",
            f"{value.percent(value.any_failure):.1f}",
            f"{value.percent(value.boundary_disagreement):.1f}",
            f"{value.percent(value.trailing_disagreement):.1f}",
        ]

    if ground_truth is not None:
        rows.append(line("**real audio**", ground_truth))
    for label, value in rates.items():
        rows.append(line(label, value))
    if overall is not None:
        rows.append(line("**all**", overall))
    return markdown_table(
        # Las dos ultimas van DESPUES de `any` y separadas: no son fallas, son las filas que
        # salen del ranking a escucha. Entre las otras leerian como una categoria de fallo mas.
        [
            "slice",
            "n",
            "hard",
            "silent",
            "trunc",
            "loop",
            "halluc",
            "trailing",
            "any",
            "boundary",
            "trail-dis",
        ],
        rows,
    )


def speaker_scale_table(scale: Scale) -> str:
    """The ruler itself: the two ends, and the diagnostics that say the ruler is sound.

    Printed BEFORE the similarity numbers: a cosine column with no scale above it is the failure
    mode SPEC 7.2 was written to prevent.
    """
    rows = []
    # Orden canonico y no el del archivo: se lee de arriba abajo como una regla.
    for kind, description in PAIR_DESCRIPTIONS.items():
        end = scale.end(kind)
        if not end.n:
            continue
        rows.append(
            [
                kind,
                description,
                str(end.n),
                _ratio(end.mean),
                _ratio(end.low),
                _ratio(end.high),
            ]
        )
    return markdown_table(["end", "what it compares", "pairs", "mean", "min", "max"], rows)


def speaker_sim_table(summaries: Iterable[SimSummary]) -> str:
    """Cosine by slice, with its position on the floor-to-ceiling scale next to it.

    `cos p05` and not p95: in similarity the bad tail is the low one.
    """
    rows = [
        [
            summary.label,
            str(summary.n),
            _ratio(summary.median),
            _ratio(summary.p05),
            _ratio(summary.normalized_median),
            summary.ceiling_kind,
        ]
        for summary in summaries
    ]
    return markdown_table(["slice", "n", "cos p50", "cos p05", "scaled p50", "vs"], rows)


def sensitivity_table(points: Iterable[SensitivityPoint]) -> str:
    rows = [
        [
            point.rule,
            point.knob,
            f"{point.value:g}",
            f"{point.flagged}/{point.total}",
            f"{point.percent:.0f}",
        ]
        for point in points
    ]
    return markdown_table(["rule", "knob", "value", "flagged", "%"], rows)
