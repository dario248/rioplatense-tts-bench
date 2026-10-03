"""The audio pass: transcribe every clip and measure its shape (SPEC 7.1, 7.5).

    .venv-eval/bin/python -m rtb.metrics.transcribe \\
        --manifest data/manifests/xtts_v2.jsonl \\
        --manifest data/manifests/ground_truth.jsonl

One pass, two output files — they go stale for different reasons. Idempotent by `metric_version`:
a row transcribed with the same ASR identity is skipped, one with a different model is redone.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rtb.config import DEFAULT_THRESHOLDS_PATH, ConfigError, load_thresholds
from rtb.logconf import event, setup_logging
from rtb.manifest import JsonlWriter, index_by_generation_id, iter_rows, read_rows
from rtb.metrics import audio_shape as shape_metric
from rtb.metrics.asr import METRIC_NAME as TRANSCRIPTS_METRIC
from rtb.metrics.asr import AsrConfig, Transcriber
from rtb.schema import GenerationRow, MetricRow

logger = logging.getLogger("rtb.metrics.transcribe")

DEFAULT_OUT_DIR = Path("data/metrics")


def already_done(path: Path, version: str) -> set[str]:
    """Ids already measured with this exact version.

    Lenient on purpose: a torn tail must not block a resume of the expensive step.
    """
    return {
        row.generation_id
        for row in iter_rows(path, MetricRow, strict=False)
        if row.metric_version == version
    }


def pending(
    rows: Sequence[GenerationRow],
    transcribed: set[str],
    shaped: set[str],
) -> tuple[list[GenerationRow], int]:
    """Rows that still need either output, plus how many were skipped entirely."""
    todo = [
        row
        for row in rows
        # Sin wav no hay nada que transcribir: las filas con error se cuentan en la tasa de fallo,
        # no acá. El warm-up es descartable por definicion.
        if row.wav_path
        and not row.warmup
        and (row.generation_id not in transcribed or row.generation_id not in shaped)
    ]
    measurable = [row for row in rows if row.wav_path and not row.warmup]
    return todo, len(measurable) - len(todo)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtb.metrics.transcribe",
        description="Transcribe the benchmark audio and measure where speech starts and stops.",
    )
    parser.add_argument("--manifest", required=True, type=Path, action="append")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    # Todo lo que cambia una transcripcion entra por CLI y queda registrado en cada fila.
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="es")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--limit", type=int, help="stop after N clips, for a smoke test")
    parser.add_argument("--dry-run", action="store_true", help="report what would run and exit")
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
    except ConfigError as err:
        event(logger, "transcribe_aborted", level=logging.CRITICAL, error=str(err))
        return 1

    asr_config = AsrConfig(
        model=args.asr_model,
        compute_type=args.compute_type,
        device=args.device,
        language=args.language,
        beam_size=args.beam_size,
    )
    transcriber = Transcriber(asr_config)

    rows: list[GenerationRow] = []
    for manifest_path in args.manifest:
        # Append-only: vale la ultima fila de cada generation_id.
        rows.extend(index_by_generation_id(read_rows(resolve(manifest_path))).values())

    transcripts_path = resolve(args.out) / f"{TRANSCRIPTS_METRIC}.jsonl"
    shapes_path = resolve(args.out) / f"{shape_metric.METRIC_NAME}.jsonl"

    todo, skipped = pending(
        rows,
        already_done(transcripts_path, transcriber.metric_version),
        already_done(shapes_path, thresholds.audio_shape.version),
    )
    if args.limit:
        todo = todo[: args.limit]

    event(
        logger,
        "transcribe_plan",
        manifests=len(args.manifest),
        rows=len(rows),
        pending=len(todo),
        skipped=skipped,
        asr=transcriber.metric_version,
        shape_version=thresholds.audio_shape.version,
    )
    if args.dry_run or not todo:
        return 0

    event(logger, "loading_asr_model", model=asr_config.model, compute_type=args.compute_type)
    try:
        transcriber.load()
    except ImportError as err:
        event(
            logger,
            "transcribe_aborted",
            level=logging.CRITICAL,
            error=f"{err}. The ASR stack lives in its own environment — see envs/eval.txt",
        )
        return 1

    failed = 0
    with (
        JsonlWriter(transcripts_path) as transcripts,
        JsonlWriter(shapes_path) as shapes,
    ):
        for position, row in enumerate(todo, start=1):
            wav_path = resolve(Path(row.wav_path or ""))
            transcript = transcriber.transcript_row(row.generation_id, wav_path)
            shape = shape_metric.shape_row(
                row.generation_id, wav_path, thresholds.audio_shape, thresholds.audio_shape.version
            )
            transcripts.write(transcript)
            shapes.write(shape)
            if transcript.error or shape.error:
                failed += 1
            event(
                logger,
                "transcribed",
                level=logging.DEBUG,
                generation_id=row.generation_id,
                progress=f"{position}/{len(todo)}",
                text=transcript.values.get("text"),
                speech_s=shape.values.get("speech_s"),
                tail_silence_s=shape.values.get("tail_silence_s"),
                error=transcript.error or shape.error,
            )

    event(logger, "transcribe_finished", written=len(todo), failed=failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
