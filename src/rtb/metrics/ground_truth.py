"""Manifest rows for the real audio (SPEC 5.5).

    python -m rtb.metrics.ground_truth

Same schema and join key as a synthesis, so every metric downstream treats them as one more column.
Needs the core installed: hand-writing manifest rows is how invalid manifests happen.
"""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from collections.abc import Sequence
from pathlib import Path

from rtb.config import DEFAULT_SPEAKERS_PATH, ConfigError, RefClip, Speaker, load_speakers
from rtb.logconf import event, setup_logging
from rtb.manifest import JsonlWriter
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, make_ground_truth_id, utcnow

logger = logging.getLogger("rtb.metrics.ground_truth")

DEFAULT_OUT = Path("data/manifests/ground_truth.jsonl")
# Tolerancia al comparar la duracion declarada en speakers.yaml contra el header del wav.
DURATION_TOLERANCE_S = 0.02


class GroundTruthError(Exception):
    pass


def probe_wav(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        frames = fh.getnframes()
    if rate <= 0 or frames <= 0:
        raise GroundTruthError(f"{path}: no audio")
    return rate, frames / float(rate)


def row_for_clip(speaker: Speaker, clip: RefClip, root: Path) -> GenerationRow:
    """One ground-truth row, with duration read from the file rather than trusted."""
    wav_path = clip.path if clip.path.is_absolute() else root / clip.path
    if not wav_path.exists():
        raise GroundTruthError(
            f"{clip.path} is missing; build the clips first "
            "(scripts/build_reference_clips.py)"
        )
    if not clip.transcript:
        raise GroundTruthError(f"{clip.path} has no transcript, so it cannot be a reference point")

    sample_rate, duration_s = probe_wav(wav_path)
    if clip.duration_s and abs(clip.duration_s - duration_s) > DURATION_TOLERANCE_S:
        # Si el YAML y el disco no coinciden, la tasa de habla que sale del YAML tampoco vale, y
        # eso mueve en silencio toda la deteccion de duracion.
        raise GroundTruthError(
            f"{clip.path}: speakers.yaml declares {clip.duration_s:.3f}s but the file is "
            f"{duration_s:.3f}s — the clips on disk are not the ones the config describes"
        )

    return GenerationRow(
        generation_id=make_ground_truth_id(speaker.id, clip.path.stem),
        model=GROUND_TRUTH_MODEL,
        speaker_id=speaker.id,
        phrase_id=clip.path.stem,
        # El texto viaja EN la fila: estos clips no salen de phrases.yaml, asi que no hay
        # phrase_id que consultar (SPEC 5.3).
        reference_text=clip.transcript,
        wav_path=str(clip.path),
        sample_rate=sample_rate,
        duration_s=duration_s,
        streaming=False,
        warmup=False,
        created_at=utcnow(),
    )


def build_rows(speakers: Sequence[Speaker], root: Path) -> list[GenerationRow]:
    rows = []
    for speaker in speakers:
        for clip in [*speaker.control_clips, *speaker.calibration_clips]:
            rows.append(row_for_clip(speaker, clip, root))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtb.metrics.ground_truth",
        description="Write manifest rows for the real control and calibration clips.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--speakers", type=Path, default=DEFAULT_SPEAKERS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
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
        speakers = load_speakers(resolve(args.speakers))
        rows = build_rows(speakers, root)
    except (ConfigError, GroundTruthError) as err:
        event(logger, "ground_truth_aborted", level=logging.CRITICAL, error=str(err))
        return 1

    out_path = resolve(args.out)
    # Se reescribe entero: es una funcion determinista de speakers.yaml, y dos filas con el mismo
    # generation_id serian ambiguas.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    with JsonlWriter(out_path) as writer:
        for row in rows:
            writer.write(row)

    event(
        logger,
        "ground_truth_written",
        path=str(out_path),
        rows=len(rows),
        speakers=len(speakers),
        audio_seconds=round(sum(row.duration_s or 0.0 for row in rows), 1),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
