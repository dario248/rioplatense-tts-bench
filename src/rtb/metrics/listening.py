"""Listening worklists: the rows the instrument cannot decide on its own (ANALYSIS A13).

    python -m rtb.metrics.listening --verdict trailing_audio

One TSV row per clip with an empty `veredicto` column and everything needed to judge it.
Regenerating preserves the verdicts already written — the one thing here that is not recomputable.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rtb.config import (
    DEFAULT_PHRASES_PATH,
    DEFAULT_THRESHOLDS_PATH,
    ConfigError,
    load_phrases,
    load_thresholds,
)
from rtb.logconf import event, setup_logging
from rtb.manifest import index_by_generation_id, read_rows
from rtb.metrics import anatomy as anatomy_metric
from rtb.metrics.asr import METRIC_NAME as TRANSCRIPTS_METRIC
from rtb.metrics.audio_shape import METRIC_NAME as SHAPE_METRIC
from rtb.metrics.audio_shape import rms_envelope
from rtb.metrics.failures import METRIC_NAME as FAILURES_METRIC
from rtb.schema import GenerationRow, MetricRow

logger = logging.getLogger("rtb.metrics.listening")

DEFAULT_OUT_DIR = Path("data/listening")
VERDICT_COLUMN = "veredicto"

# Vocabulario cerrado, uno por veredicto. Sin esto, la mitad de las filas dice "ruido" y la otra
# "noise, poco" y no se puede contar nada.
VERDICTS = {
    "trailing_audio": {
        "habla": "la cola tiene palabras que el modelo invento",
        "ruido": "la cola suena pero no son palabras",
        "ambas": "las dos cosas en el mismo clip",
        "nada": "no se escucha nada: la marca es un falso positivo",
        "corte": "la cola es la frase cortandose, no material extra",
    },
    # Faltan palabras al final y hay sonido ahi. La pregunta es quien se comio el final, y la
    # regla no puede contestarla: el ASR que no lo oyo y el modelo que lo dijo mal se ven igual
    # desde el texto (ANALYSIS A2).
    "trailing_disagreement": {
        "dijo_bien": "el modelo dijo el final entero; el ASR no lo transcribio",
        "dijo_mal": "el modelo pronuncio mal el final, y el ASR transcribio lo que pudo",
        "no_lo_dijo": "el modelo se corto: el truncamiento es real",
        "nada": "no se oye nada despues de la ultima palabra: la marca es un falso positivo",
    },
    # La pregunta acá es de quien es la particion, y la regla no puede contestarla: el mismo
    # patron lo produce un rehilamiento fuerte y una aproximante (ANALYSIS A1).
    "boundary_disagreement": {
        "dijo_bien": "el modelo dijo la palabra entera; el ASR la partio",
        "dijo_mal": "el modelo pronuncio algo distinto, y la particion refleja el audio",
        "dialectal": "lo que cambia es la realizacion rioplatense: rehilamiento o su ausencia",
        "nada": "no se oye diferencia: la marca es un falso positivo",
    },
}

COLUMNS = (
    "generation_id",
    "modelo",
    "hablante",
    "phrase_id",
    "categoria",
    "wav",
    "cola_s",
    "nivel_db",
    "habla_s",
    "relleno_s",
    "total_s",
    "wer",
    "frase",
    "transcript",
    VERDICT_COLUMN,
)


def db_relative(value: float, peak: float) -> float:
    """Level in dB relative to the clip's own peak — what decides whether a tail is audible."""
    if peak <= 0 or value <= 0:
        return -120.0
    return 20 * math.log10(value / peak)


@dataclass
class TailRow:
    """One clip to listen to, with everything needed to judge it."""

    generation_id: str
    model: str
    speaker: str
    phrase_id: str
    category: str
    wav: str
    tail_s: float
    level_db: float
    speech_s: float
    padding_s: float
    total_s: float
    wer: float | None
    phrase: str
    transcript: str
    verdict: str = ""

    def cells(self) -> list[str]:
        def num(value: float | None, digits: int = 2) -> str:
            return "" if value is None else f"{value:.{digits}f}"

        return [
            self.generation_id,
            self.model,
            self.speaker,
            self.phrase_id,
            self.category,
            self.wav,
            num(self.tail_s),
            num(self.level_db, 1),
            num(self.speech_s),
            num(self.padding_s),
            num(self.total_s),
            num(self.wer, 3),
            flatten(self.phrase),
            flatten(self.transcript),
            self.verdict,
        ]


def flatten(text: str) -> str:
    """One line, no tabs: a TSV cell with a newline in it silently shifts every later column."""
    return " ".join((text or "").split())


@dataclass
class Worklist:
    rows: list[TailRow] = field(default_factory=list)
    carried: int = 0

    def to_tsv(self) -> str:
        lines = ["\t".join(COLUMNS)]
        lines.extend("\t".join(row.cells()) for row in self.rows)
        return "\n".join(lines) + "\n"


def read_verdicts(path: Path) -> dict[str, str]:
    """Annotations already written, by `generation_id`.

    Lenient on purpose: a stray column in the spreadsheet must not throw away an afternoon.
    """
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    header = lines[0].split("\t")
    try:
        id_at = header.index("generation_id")
        verdict_at = header.index(VERDICT_COLUMN)
    except ValueError:
        logger.warning("%s has no generation_id/%s columns; ignoring it", path, VERDICT_COLUMN)
        return {}
    verdicts = {}
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) > max(id_at, verdict_at) and cells[verdict_at].strip():
            verdicts[cells[id_at]] = cells[verdict_at].strip()
    return verdicts


def build(
    manifests: Sequence[Path],
    metrics_dir: Path,
    phrases_by_id: dict[str, tuple[str, str]],
    frame_ms: int,
    verdict_field: str,
    existing: dict[str, str],
    root: Path,
) -> Worklist:
    """Every row flagged with `verdict_field`, with its tail measured."""
    transcripts = {
        row.generation_id: row
        for row in read_rows(metrics_dir / f"{TRANSCRIPTS_METRIC}.jsonl", MetricRow, strict=False)
    }
    shapes = {
        row.generation_id: row
        for row in read_rows(metrics_dir / f"{SHAPE_METRIC}.jsonl", MetricRow, strict=False)
    }
    flagged = {
        row.generation_id: row
        for row in read_rows(metrics_dir / f"{FAILURES_METRIC}.jsonl", MetricRow, strict=False)
        if not row.error and row.values.get(verdict_field)
    }
    anatomies = anatomy_metric.from_rows(shapes, transcripts)

    generations: dict[str, GenerationRow] = {}
    for manifest in manifests:
        path = manifest if manifest.is_absolute() else root / manifest
        generations.update(index_by_generation_id(read_rows(path, GenerationRow)))

    worklist = Worklist()
    for generation_id, verdict_row in flagged.items():
        generation = generations.get(generation_id)
        transcript = transcripts.get(generation_id)
        anatomy = anatomies.get(generation_id)
        if generation is None or generation.wav_path is None or anatomy is None:
            continue

        last_word = (transcript.values.get("last_word_end_s") if transcript else None) or 0.0
        declared = Path(generation.wav_path)
        wav = declared if declared.is_absolute() else root / declared
        try:
            rms, frame_s = rms_envelope(wav, frame_ms)
        except Exception as err:  # noqa: BLE001 — un wav roto no aborta la lista
            logger.warning("cannot read %s: %s", wav, err)
            continue
        peak = max(rms, default=0.0)
        tail = rms[int(last_word / frame_s) :] if frame_s else []
        # El pico de la cola y no su mediana: lo que decide si se escucha algo es el momento mas
        # fuerte, no el promedio de una cola que puede ser casi toda silencio.
        level = db_relative(max(tail, default=0.0), peak)

        phrase_text, category = phrases_by_id.get(
            generation.phrase_id or "", (generation.reference_text or "", "?")
        )
        verdict = existing.get(generation_id, "")
        if verdict:
            worklist.carried += 1
        worklist.rows.append(
            TailRow(
                generation_id=generation_id,
                model=generation.model,
                speaker=generation.speaker_id,
                phrase_id=generation.phrase_id or "",
                category=category,
                wav=str(generation.wav_path),
                tail_s=anatomy.trailing_audio_s,
                level_db=level,
                speech_s=anatomy.speech_s,
                padding_s=anatomy.silence_s,
                total_s=anatomy.total_s,
                wer=verdict_row.values.get("wer"),
                phrase=phrase_text,
                transcript=(transcript.values.get("text", "") if transcript else ""),
                verdict=verdict,
            )
        )

    # Peor primero: si la escucha se corta a la mitad, lo escuchado es lo que mas informa.
    worklist.rows.sort(key=lambda row: (-row.tail_s, row.generation_id))
    return worklist


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtb.metrics.listening",
        description="Build a TSV worklist of the clips a human has to listen to.",
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--metrics", type=Path, default=Path("data/metrics"))
    parser.add_argument("--phrases", type=Path, default=DEFAULT_PHRASES_PATH)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--verdict",
        default="trailing_audio",
        help="which failure field to build the worklist from",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--plain-logs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level, json_lines=not args.plain_logs)
    root = Path(args.root).resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        thresholds = load_thresholds(resolve(args.thresholds))
        phrases_by_id = {
            phrase.id: (phrase.text, phrase.category.value)
            for phrase in load_phrases(resolve(args.phrases))
        }
    except ConfigError as err:
        event(logger, "listening_aborted", level=logging.CRITICAL, error=str(err))
        return 1

    out_path = resolve(args.out) / f"{args.verdict}.tsv"
    existing = read_verdicts(out_path)
    worklist = build(
        args.manifest,
        resolve(args.metrics),
        phrases_by_id,
        thresholds.audio_shape.frame_ms,
        args.verdict,
        existing,
        root,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(worklist.to_tsv(), encoding="utf-8")

    by_model: dict[str, int] = {}
    for row in worklist.rows:
        by_model[row.model] = by_model.get(row.model, 0) + 1
    event(
        logger,
        "worklist_written",
        path=str(out_path),
        rows=len(worklist.rows),
        by_model=by_model,
        verdicts_carried=worklist.carried,
        pending=len(worklist.rows) - worklist.carried,
    )
    print(f"\n[{len(worklist.rows)} clips para escuchar en {out_path}]")
    vocabulary = VERDICTS.get(args.verdict, {})
    if vocabulary:
        print("Vocabulario del veredicto:")
        for name, meaning in vocabulary.items():
            print(f"  {name:10s} {meaning}")
    else:
        print(f"Sin vocabulario cerrado para `{args.verdict}`: definirlo antes de etiquetar.")
    if worklist.carried:
        print(f"\n[{worklist.carried} veredictos ya escritos, preservados]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
