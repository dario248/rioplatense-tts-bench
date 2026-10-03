"""The failure taxonomy, now that there is a transcript to classify with (SPEC 7.5).

Almost nothing here is a threshold: the edit distance decomposes and the decomposition is the
classification. Only two numbers are chosen, and `--sensitivity` sweeps both. Rows whose alignment
is untrustworthy are set aside rather than averaged in.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from rtb.config import FailureThresholds, Phrase, Thresholds
from rtb.metrics.intelligibility import Score, max_ngram_repeats, score
from rtb.metrics.sensitivity import SensitivityPoint, knob_label
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, MetricRow

METRIC_NAME = "failures"

# La taxonomia no es solo umbrales: la mayor parte es codigo, y cambiar una regla mueve la tasa
# publicada sin tocar `thresholds.yaml`. Paso dos veces el 2026-09-16 —la tercera condicion del
# loop y la precedencia de `trailing_disagreement`— y las filas no lo declaraban. Se sube a mano
# cuando `classify` cambia de conducta, y `test_intelligibility` falla si cambia sin subirla.
TAXONOMY_VERSION = "1"

# Rango amplio a proposito: el valor elegido decide cuantas filas entran de cada modelo.
SENSITIVITY_TAILS_S = (0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)

# Idem para el desacuerdo de frontera: el corte decide cuantas filas salen de `hallucinated`.
SENSITIVITY_GLUED_RATES = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30)

# Idem para el desborde del loop: el corte decide cuantas filas quedan como loop y cuantas caen al
# resto de la cadena.
SENSITIVITY_LENGTH_RATIOS = (1.05, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5)


@dataclass
class Verdict:
    """What went wrong with one synthesis, and the evidence for saying so."""

    hard_error: bool = False
    silent: bool = False
    truncated: bool = False
    looped: bool = False
    hallucinated: bool = False
    # Ni fallo ni exito: el WER vio inserciones y el CER pegado las desmiente. Queda FUERA de
    # any_failure a proposito — la fila va a escucha, no al ranking (SPEC 7.5).
    boundary_disagreement: bool = False
    trailing_audio: bool = False
    # Los segundos, no solo el booleano: es la magnitud que se reporta en la anatomia del audio.
    trailing_audio_s: float | None = None
    # El par del anterior, y tampoco es una falla: faltan palabras al final Y hay sonido ahi, asi
    # que truncamiento y cola se contradicen. Va a escucha, no al ranking (SPEC 7.5).
    trailing_disagreement: bool = False
    unscorable: bool = False
    reason: str | None = None

    @property
    def any_failure(self) -> bool:
        return (
            self.hard_error
            or self.silent
            or self.truncated
            or self.looped
            or self.hallucinated
            or self.trailing_audio
        )


def overruns(hypothesis_words: int, reference_words: int, thresholds: FailureThresholds) -> bool:
    """Whether the hypothesis is long enough to count as running past the text it was given."""
    if reference_words <= 0:
        return False
    return hypothesis_words / reference_words >= thresholds.loop_min_length_ratio


def classify(
    scored: Score,
    shape: Mapping[str, object] | None,
    thresholds: FailureThresholds,
    last_word_end_s: float | None = None,
) -> Verdict:
    """Turn one alignment plus one envelope into a verdict."""
    if scored.formatting_conflict:
        # Ni fallo ni exito: la medicion no sirve, y contarla como cualquiera de los dos seria
        # inventar un dato.
        return Verdict(unscorable=True, reason="asr wrote digits where the reference has words")

    verdict = Verdict()
    hypothesis_words = scored.hypothesis.split()

    verdict.hallucinated = scored.words.insertions >= thresholds.insertions_min
    # Partir una palabra CONSUME la adyacente ademas de insertar otra; una alucinacion solo
    # inserta. Esa sustitucion es el mecanismo, y el texto pegado dice que no hay material nuevo.
    verdict.boundary_disagreement = (
        verdict.hallucinated
        and scored.words.substitutions >= 1
        and scored.glued_rate <= thresholds.boundary_max_glued_rate
    )
    if verdict.boundary_disagreement:
        verdict.hallucinated = False
    verdict.truncated = scored.words.trailing_deletions >= thresholds.trailing_deletions_min
    # Las tres condiciones son una sola idea: un decodificador en loop repite, agrega, y el
    # resultado se pasa de largo. Sin la tercera, decir "uno uno" por "once" en una frase de 21
    # palabras alcanza para marcar loop, y eso no es un loop (SPEC 7.5).
    verdict.looped = (
        max_ngram_repeats(hypothesis_words) >= thresholds.ngram_repeats_min
        and scored.words.insertions >= thresholds.insertions_min
        and overruns(len(hypothesis_words), scored.words.reference_length, thresholds)
    )

    if shape:
        verdict.silent = bool(shape.get("silent"))
        speech_end = shape.get("speech_end_s")
        # Sonido despues de la ultima palabra que Whisper oyo: contenido que el modelo invento.
        # El umbral es generoso porque los timestamps de palabra tienen error de decenas de ms.
        if isinstance(speech_end, (int, float)) and last_word_end_s is not None:
            tail = max(0.0, float(speech_end) - last_word_end_s)
            verdict.trailing_audio_s = tail
            verdict.trailing_audio = tail > thresholds.tail_after_last_word_s

    # Faltar palabras al final y haber sonido justo ahi no pueden ser verdad a la vez: o el ASR no
    # las oyo o el modelo las dijo mal. El piso es 1 y no `trailing_deletions_min` porque la
    # palabra final suelta es justamente el caso que ese umbral evita llamar truncamiento.
    verdict.trailing_disagreement = scored.words.trailing_deletions >= 1 and verdict.trailing_audio
    if verdict.trailing_disagreement:
        verdict.truncated = False
        verdict.trailing_audio = False
        # Y tambien la alucinacion, por la misma razon que `boundary_disagreement`: a un transcript
        # al que le faltan palabras del final no se le puede contar material de mas. La insercion
        # que ve el alineador no es contenido que el modelo agrego, es el acomodo de lo que el ASR
        # no escribio. Un veredicto de revision no deja etiquetas de falla puestas.
        verdict.hallucinated = False

    if verdict.silent:
        verdict.reason = "silent"
    elif verdict.looped:
        verdict.reason = "loop"
    elif verdict.hallucinated:
        verdict.reason = "hallucination"
    elif verdict.boundary_disagreement:
        verdict.reason = "boundary disagreement"
    elif verdict.trailing_disagreement:
        verdict.reason = "trailing disagreement"
    elif verdict.truncated:
        verdict.reason = "truncation"
    elif verdict.trailing_audio:
        verdict.reason = "trailing audio"
    return verdict


def reference_forms(row: GenerationRow, phrase: Phrase | None) -> list[str] | None:
    """What the clip was supposed to say: `reference_text`, else the phrase and its alternatives
    (SPEC 5.5)."""
    if row.reference_text:
        return [row.reference_text]
    if phrase is None:
        return None
    return [phrase.spoken_reference, *phrase.accepted_alternatives]


def evaluate_row(
    row: GenerationRow,
    phrase: Phrase | None,
    transcript: MetricRow | None,
    shape: MetricRow | None,
    thresholds: Thresholds,
    asr_version: str,
) -> MetricRow:
    version = f"taxonomy{TAXONOMY_VERSION}/thresholds{thresholds.version}/{asr_version}"

    if row.error is not None:
        return MetricRow(
            generation_id=row.generation_id,
            metric=METRIC_NAME,
            metric_version=version,
            values={"hard_error": True, "any_failure": True, "reason": "hard_error"},
        )

    forms = reference_forms(row, phrase)
    if forms is None or transcript is None or transcript.error:
        missing = (
            "phrase not in the corpus"
            if forms is None
            else "no transcript for this row"
            if transcript is None
            else f"transcription failed: {transcript.error}"
        )
        return MetricRow(
            generation_id=row.generation_id,
            metric=METRIC_NAME,
            metric_version=version,
            error=f"cannot classify: {missing}",
        )

    scored = score(forms, str(transcript.values.get("text", "")))
    if scored is None:  # solo si forms viniera vacio, que reference_forms no permite
        return MetricRow(
            generation_id=row.generation_id,
            metric=METRIC_NAME,
            metric_version=version,
            error="cannot classify: no reference form",
        )
    last_word_end = transcript.values.get("last_word_end_s")
    shape_values = shape.values if shape and not shape.error else None
    verdict = classify(
        scored,
        shape_values,
        thresholds.failures,
        last_word_end if isinstance(last_word_end, (int, float)) else None,
    )

    return MetricRow(
        generation_id=row.generation_id,
        metric=METRIC_NAME,
        metric_version=version,
        values={
            "hard_error": False,
            "silent": verdict.silent,
            "truncated": verdict.truncated,
            "looped": verdict.looped,
            "hallucinated": verdict.hallucinated,
            "boundary_disagreement": verdict.boundary_disagreement,
            "trailing_audio": verdict.trailing_audio,
            "trailing_disagreement": verdict.trailing_disagreement,
            "trailing_audio_s": (
                None
                if verdict.trailing_audio_s is None
                else round(verdict.trailing_audio_s, 3)
            ),
            "unscorable": verdict.unscorable,
            "any_failure": verdict.any_failure,
            "reason": verdict.reason,
            # La evidencia viaja con el veredicto: sin esto, revisar una clasificacion obliga a
            # recalcular todo.
            "wer": round(scored.wer, 4),
            "cer": round(scored.cer, 4),
            "substitutions": scored.words.substitutions,
            "insertions": scored.words.insertions,
            "deletions": scored.words.deletions,
            "trailing_deletions": scored.words.trailing_deletions,
            "insertion_rate": round(scored.words.insertion_rate, 4),
            "glued_rate": round(scored.glued_rate, 4),
            "ngram_repeats": max_ngram_repeats(scored.hypothesis.split()),
            "reference_words": scored.reference_words,
            # El largo relativo viaja con la fila para que el barrido del loop no tenga que volver
            # a alinear, igual que `trailing_audio_s` y `glued_rate`.
            "length_ratio": (
                round(len(scored.hypothesis.split()) / scored.reference_words, 4)
                if scored.reference_words
                else None
            ),
            "matched_form_index": scored.matched_index,
            "category": phrase.category.value if phrase else None,
        },
    )


def evaluate(
    rows: Iterable[GenerationRow],
    phrases: Iterable[Phrase],
    transcripts: Mapping[str, MetricRow],
    shapes: Mapping[str, MetricRow],
    thresholds: Thresholds,
    asr_version: str,
) -> list[MetricRow]:
    by_id = {phrase.id: phrase for phrase in phrases}
    return [
        evaluate_row(
            row,
            by_id.get(row.phrase_id),
            transcripts.get(row.generation_id),
            shapes.get(row.generation_id),
            thresholds,
            asr_version,
        )
        for row in rows
        if not row.warmup
    ]


@dataclass
class FailureRates:
    """The failure-rate column of the results table."""

    total: int = 0
    hard_error: int = 0
    silent: int = 0
    truncated: int = 0
    looped: int = 0
    hallucinated: int = 0
    boundary_disagreement: int = 0
    trailing_audio: int = 0
    trailing_disagreement: int = 0
    any_failure: int = 0
    unscorable: int = 0
    unclassified: int = 0
    wers: list[float] = field(default_factory=list)
    cers: list[float] = field(default_factory=list)

    @property
    def scored(self) -> int:
        """Rows the taxonomy could actually judge."""
        return self.total - self.unscorable - self.unclassified

    def percent(self, count: int) -> float:
        return 100.0 * count / self.scored if self.scored else 0.0


def rates(metric_rows: Iterable[MetricRow]) -> FailureRates:
    result = FailureRates()
    for row in metric_rows:
        result.total += 1
        if row.error is not None:
            result.unclassified += 1
            continue
        values = row.values
        if values.get("unscorable"):
            result.unscorable += 1
            continue
        for name in (
            "hard_error",
            "silent",
            "truncated",
            "looped",
            "hallucinated",
            "boundary_disagreement",
            "trailing_audio",
            "trailing_disagreement",
            "any_failure",
        ):
            if values.get(name):
                setattr(result, name, getattr(result, name) + 1)
        if isinstance(values.get("wer"), (int, float)):
            result.wers.append(float(values["wer"]))
            result.cers.append(float(values["cer"]))
    return result


def rates_by_category(metric_rows: Iterable[MetricRow]) -> dict[str, FailureRates]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in metric_rows:
        category = str(row.values.get("category") or "unknown")
        grouped.setdefault(category, []).append(row)
    return {category: rates(rows) for category, rows in sorted(grouped.items())}


def ground_truth_rates(
    rows: Iterable[GenerationRow], metric_rows: Mapping[str, MetricRow]
) -> FailureRates:
    """The same taxonomy over real audio — the scale everything else is read against."""
    return rates(
        metric_rows[row.generation_id]
        for row in rows
        if row.model == GROUND_TRUTH_MODEL and row.generation_id in metric_rows
    )


def boundary_sensitivity(
    metric_rows: Iterable[MetricRow],
    thresholds: FailureThresholds,
) -> list[SensitivityPoint]:
    """How many rows leave `hallucinated` as the glued-text cutoff moves.

    The denominator is the rows where the mechanism holds at all — an insertion together with a
    substitution — because everywhere else the cutoff has nothing to decide.
    """
    rates_seen = [
        float(row.values["glued_rate"])
        for row in metric_rows
        if not row.error
        and isinstance(row.values.get("glued_rate"), (int, float))
        and int(row.values.get("insertions") or 0) >= thresholds.insertions_min
        and int(row.values.get("substitutions") or 0) >= 1
    ]
    in_use = thresholds.boundary_max_glued_rate
    return [
        SensitivityPoint(
            rule="boundary_disagreement",
            knob=knob_label("boundary_max_glued_rate", value, in_use),
            value=value,
            flagged=sum(1 for rate in rates_seen if rate <= value),
            total=len(rates_seen),
        )
        for value in sorted({*SENSITIVITY_GLUED_RATES, in_use})
    ]


def loop_sensitivity(
    metric_rows: Iterable[MetricRow],
    thresholds: FailureThresholds,
) -> list[SensitivityPoint]:
    """How many rows stay `looped` as the overrun cutoff moves.

    The denominator is the rows where the mechanism holds at all — a repeated n-gram together with
    an insertion — because everywhere else the cutoff has nothing to decide.
    """
    ratios = [
        float(row.values["length_ratio"])
        for row in metric_rows
        if not row.error
        and isinstance(row.values.get("length_ratio"), (int, float))
        and int(row.values.get("ngram_repeats") or 0) >= thresholds.ngram_repeats_min
        and int(row.values.get("insertions") or 0) >= thresholds.insertions_min
    ]
    in_use = thresholds.loop_min_length_ratio
    return [
        SensitivityPoint(
            rule="loop",
            knob=knob_label("loop_min_length_ratio", value, in_use),
            value=value,
            flagged=sum(1 for ratio in ratios if ratio >= value),
            total=len(ratios),
        )
        for value in sorted({*SENSITIVITY_LENGTH_RATIOS, in_use})
    ]


def tail_sensitivity(
    metric_rows: Iterable[MetricRow],
    thresholds: FailureThresholds,
) -> list[SensitivityPoint]:
    """How much the trailing-audio rate moves when its threshold moves.

    Re-thresholds the `trailing_audio_s` every classified row carries. The denominator is the rows
    where the tail could be measured at all.
    """
    tails = [
        float(row.values["trailing_audio_s"])
        for row in metric_rows
        if not row.error and isinstance(row.values.get("trailing_audio_s"), (int, float))
    ]
    in_use = thresholds.tail_after_last_word_s
    return [
        SensitivityPoint(
            rule="trailing_audio",
            knob=knob_label("tail_after_last_word_s", value, in_use),
            value=value,
            flagged=sum(1 for tail in tails if tail > value),
            total=len(tails),
        )
        for value in sorted({*SENSITIVITY_TAILS_S, in_use})
    ]
