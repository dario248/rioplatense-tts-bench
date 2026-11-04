"""Build the worklist of clips to transcribe by ear: one TSV row per distinct audio.

    python -m rtb.metrics.corrections --category normalizacion --model sintetica

`dicho` comes pre-loaded with the ASR transcript, to correct instead of retyping, and `revisado`
is what marks a row as done. Rebuilding preserves both.

Rows are keyed by `audio_sha` and not by `generation_id`, which is a grid coordinate and survives
a regeneration: keyed by it, a transcript would silently attach to audio nobody heard.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rtb.config import DEFAULT_PHRASES_PATH, ConfigError, load_phrases
from rtb.logconf import event, setup_logging
from rtb.manifest import index_by_generation_id, read_rows
from rtb.metrics.asr import METRIC_NAME as TRANSCRIPTS_METRIC
from rtb.schema import GenerationRow, MetricRow

logger = logging.getLogger("rtb.metrics.corrections")


class Dropped(Exception):
    """Rebuilding would drop transcripts already written: the one thing that cannot be redone."""

    def __init__(self, shas: set[str]) -> None:
        super().__init__(f"{len(shas)} transcripciones quedarian afuera")
        self.shas = shas


DEFAULT_OUT_DIR = Path("data/corrections")
DONE_COLUMN = "revisado"
SAID_COLUMN = "dicho"

# La forma hablada esperada NO va: a la vista se transcribe lo que el modelo deberia haber dicho.
COLUMNS = (
    "generation_id",
    "audio_sha",
    "modelo",
    "hablante",
    "phrase_id",
    "frase",
    "n",
    "wav",
    "asr",
    SAID_COLUMN,
    DONE_COLUMN,
)


def flatten(text: str) -> str:
    """One line, no tabs: a TSV cell with a newline in it silently shifts every later column."""
    return " ".join((text or "").split())


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@dataclass
class Clip:
    """One distinct audio to transcribe, and the generations that share it."""

    generation_id: str
    audio_sha: str
    model: str
    speaker: str
    phrase_id: str
    phrase: str
    n: int
    wav: str
    asr: str
    said: str = ""
    done: str = ""

    def cells(self) -> list[str]:
        return [
            self.generation_id,
            self.audio_sha,
            self.model,
            self.speaker,
            self.phrase_id,
            flatten(self.phrase),
            str(self.n),
            self.wav,
            flatten(self.asr),
            flatten(self.said),
            self.done,
        ]

    def unrewritten_digits(self) -> bool:
        """Marked as reviewed but still carrying digits: the number was never written out."""
        return bool(self.done.strip()) and any(c.isdigit() for c in self.said)


@dataclass
class Worklist:
    rows: list[Clip] = field(default_factory=list)
    carried: int = 0

    def to_tsv(self) -> str:
        lines = ["\t".join(COLUMNS)]
        lines.extend("\t".join(row.cells()) for row in self.rows)
        return "\n".join(lines) + "\n"


def read_done(path: Path) -> dict[str, tuple[str, str]]:
    """Transcripts already written, by `audio_sha`.

    Lenient on purpose: a stray column in the spreadsheet must not throw away an afternoon.
    """
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    header = lines[0].split("\t")
    try:
        sha_at = header.index("audio_sha")
        said_at = header.index(SAID_COLUMN)
        done_at = header.index(DONE_COLUMN)
    except ValueError:
        logger.warning(
            "%s has no audio_sha/%s/%s columns; ignoring it", path, SAID_COLUMN, DONE_COLUMN
        )
        return {}
    out = {}
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) > max(sha_at, said_at, done_at) and cells[done_at].strip():
            out[cells[sha_at]] = (cells[said_at], cells[done_at].strip())
    return out


def models_of(path: Path, shas: set[str]) -> set[str]:
    """Which models those rows belong to, to name them when a rebuild would drop their work."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if not lines:
        return set()
    header = lines[0].split("\t")
    try:
        sha_at, model_at = header.index("audio_sha"), header.index("modelo")
    except ValueError:
        return set()
    found = set()
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) > max(sha_at, model_at) and cells[sha_at] in shas:
            found.add(cells[model_at])
    return found


def build(
    manifests: Sequence[Path],
    metrics_dir: Path,
    category: str,
    models: Sequence[str],
    phrases_by_id: dict[str, tuple[str, str]],
    existing: dict[str, tuple[str, str]],
    root: Path,
    one_seed: Sequence[str] = (),
) -> Worklist:
    """One row per distinct audio in `category`, with the ASR text as the draft to correct."""
    transcripts = {
        row.generation_id: row.values.get("text", "")
        for row in read_rows(metrics_dir / f"{TRANSCRIPTS_METRIC}.jsonl", MetricRow, strict=False)
    }

    # El manifest es append-only: una corrida repetida deja la fila dos veces y gana la ultima.
    generations: dict[str, GenerationRow] = {}
    for manifest in manifests:
        path = manifest if manifest.is_absolute() else root / manifest
        generations.update(index_by_generation_id(read_rows(path, GenerationRow)))

    groups: dict[tuple[str, str], list[GenerationRow]] = {}
    for row in generations.values():
        if row.error or not row.wav_path or (models and row.model not in models):
            continue
        # Solo la rejilla publicada: el warm-up y las filas `_ref1` no entran en ningun numero.
        if row.warmup or row.ref_clip_idx != 0:
            continue
        phrase = phrases_by_id.get(row.phrase_id or "")
        if phrase is None or phrase[1] != category:
            continue
        declared = Path(row.wav_path)
        wav = declared if declared.is_absolute() else root / declared
        if not wav.exists():
            logger.warning("%s is not on disk", wav)
            continue
        groups.setdefault((row.model, sha_of(wav)), []).append(row)

    if one_seed:
        # Por modelo y no global: uno sondeado con una semilla por celda convive en el mismo TSV
        # con otros transcriptos enteros. El descarte va DESPUES de agrupar, porque si las tres
        # semillas son el mismo audio la celda ya es una fila sola y `n` tiene que seguir en 3.
        groups = keep_one_audio_per_cell(groups, set(one_seed))

    worklist = Worklist()
    for (model, sha), rows in groups.items():
        rows.sort(key=lambda r: r.generation_id)
        head = rows[0]
        said, done = existing.get(sha, (transcripts.get(head.generation_id, ""), ""))
        if done:
            worklist.carried += 1
        worklist.rows.append(
            Clip(
                generation_id=head.generation_id,
                audio_sha=sha,
                model=model,
                speaker=head.speaker_id,
                phrase_id=head.phrase_id or "",
                phrase=phrases_by_id[head.phrase_id or ""][0],
                n=len(rows),
                wav=str(head.wav_path),
                asr=transcripts.get(head.generation_id, ""),
                said=said,
                done=done,
            )
        )

    worklist.rows.sort(key=lambda row: (row.model, row.phrase_id, row.speaker))

    # Reconstruir con menos `--model` del que tiene el archivo borraria transcripciones hechas.
    perdidas = set(existing) - {row.audio_sha for row in worklist.rows}
    if perdidas:
        raise Dropped(perdidas)
    return worklist


def keep_one_audio_per_cell(
    groups: dict[tuple[str, str], list[GenerationRow]],
    models: set[str],
) -> dict[tuple[str, str], list[GenerationRow]]:
    """For `models`, one group per (model, speaker, phrase): the lowest `generation_id`."""
    best: dict[tuple[str, str, str], tuple[str, tuple[str, str]]] = {}
    kept = {}
    for key, rows in groups.items():
        if rows[0].model not in models:
            kept[key] = rows
            continue
        head = min(row.generation_id for row in rows)
        cell = (rows[0].model, rows[0].speaker_id, rows[0].phrase_id or "")
        if cell not in best or head < best[cell][0]:
            best[cell] = (head, key)
    kept.update({key: groups[key] for _, key in best.values()})
    return kept


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtb.metrics.corrections",
        description="Build a TSV of clips to transcribe by ear, one per distinct audio.",
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--category", default="normalizacion")
    parser.add_argument(
        "--one-seed",
        action="append",
        default=[],
        metavar="MODEL",
        help="probe this model with one seed per (speaker, phrase) instead of all three; repeat",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="limit to these models; repeat the flag. Default: every model in the manifests",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--metrics", type=Path, default=Path("data/metrics"))
    parser.add_argument("--phrases", type=Path, default=DEFAULT_PHRASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
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
        phrases_by_id = {
            phrase.id: (phrase.text, phrase.category.value)
            for phrase in load_phrases(resolve(args.phrases))
        }
    except ConfigError as err:
        event(logger, "corrections_aborted", level=logging.CRITICAL, error=str(err))
        return 1

    out_path = resolve(args.out) / f"{args.category}.tsv"
    try:
        worklist = build(
            args.manifest,
            resolve(args.metrics),
            args.category,
            args.model,
            phrases_by_id,
            read_done(out_path),
            root,
            args.one_seed,
        )
    except Dropped as err:
        faltan = models_of(out_path, err.shas)
        event(logger, "corrections_aborted", level=logging.CRITICAL, error=str(err))
        print(f"\nABORTADO: {err}, y el TSV no se toco.")
        print(f"Falta pasar --model y --manifest de: {', '.join(sorted(faltan)) or '?'}")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(worklist.to_tsv(), encoding="utf-8")

    covered = sum(row.n for row in worklist.rows)
    event(
        logger,
        "corrections_written",
        path=str(out_path),
        clips=len(worklist.rows),
        covers_rows=covered,
        done=worklist.carried,
    )
    print(f"\n[{len(worklist.rows)} clips para transcribir en {out_path}]")
    print(f"Cubren {covered} filas del grid: las semillas con audio identico comparten transcript.")
    print(f"Corregir la columna `{SAID_COLUMN}` escuchando, y marcar `{DONE_COLUMN}` al terminar.")
    print("Los numeros van en palabras, que es como esta la referencia.")
    if worklist.carried:
        print(f"\n[{worklist.carried} ya transcriptos, preservados]")

    sospechosas = [row for row in worklist.rows if row.unrewritten_digits()]
    if sospechosas:
        print(f"\n[{len(sospechosas)} filas marcadas pero con digitos sin escribir en palabras]")
        for row in sospechosas[:10]:
            print(f"  {row.generation_id}: {flatten(row.said)[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
