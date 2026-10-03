"""Stage 1 driver: walk the grid, synthesize what is missing, append one row per attempt.

Idempotent by (wav on disk, row in the manifest); warm-up runs outside the grid; a failing
synthesis becomes a row with an error instead of aborting the run (SPEC 6.2).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rtb.config import (
    DEFAULT_GRID_PATH,
    DEFAULT_PHRASES_PATH,
    DEFAULT_SPEAKERS_PATH,
    BenchConfig,
    ConfigError,
    Phrase,
    Speaker,
    categories_covered,
    load_bench_config,
)
from rtb.generate import resources
from rtb.generate.base import AdapterError, SynthesisResult, TTSAdapter, build_adapter
from rtb.generate.seeding import seed_everything
from rtb.logconf import event, setup_logging
from rtb.manifest import JsonlWriter, index_by_generation_id, iter_rows
from rtb.schema import GenerationRow, make_generation_id, make_warmup_id, utcnow

logger = logging.getLogger("rtb.generate")


@dataclass(frozen=True)
class GridItem:
    """One cell of the experimental grid: what to synthesize and where it goes."""

    speaker: Speaker
    phrase: Phrase
    seed: int
    ref_clip_idx: int
    generation_id: str
    wav_rel_path: Path


@dataclass
class GridFilter:
    """Subsetting for short sessions and for debugging. Never used for published numbers."""

    phrase_ids: set[str] | None = None
    speaker_ids: set[str] | None = None
    categories: set[str] | None = None
    seeds: set[int] | None = None
    ref_clip_indices: set[int] | None = None
    limit: int | None = None


@dataclass
class RunStats:
    total_grid: int = 0
    selected: int = 0
    skipped: int = 0
    generated: int = 0
    failed: int = 0
    warmup: int = 0
    errors: list[str] = field(default_factory=list)

    def as_fields(self) -> dict[str, int]:
        return {
            "total_grid": self.total_grid,
            "selected": self.selected,
            "skipped": self.skipped,
            "generated": self.generated,
            "failed": self.failed,
            "warmup": self.warmup,
        }


def wav_rel_path(
    audio_root: Path,
    model: str,
    speaker_id: str,
    phrase_id: str,
    seed: int,
    ref_clip_idx: int,
) -> Path:
    """`audio/<model>/<speaker>/<phrase>_s<seed>_ref<idx>.wav` (SPEC section 5.3)."""
    return (
        audio_root / model / speaker_id / f"{phrase_id}_s{seed}_ref{ref_clip_idx}.wav"
    )


def build_grid(
    config: BenchConfig,
    ref_clip_indices: Sequence[int] | None = None,
) -> list[GridItem]:
    """Every combination, in a deterministic order: speaker → phrase → reference clip → seed.

    Stable across runs, so an interrupted session resumes where it stopped and `--limit` always
    means the same prefix.

    `ref_clip_indices` OVERRIDES the grid's own factor instead of filtering it. El clip de
    referencia es un factor (`configs/grid.yaml`), asi que pedir uno que la grilla no trae tiene que
    extenderla: filtrarlo no seleccionaba nada y el chequeo de sensibilidad de A10 era un no-op.
    """
    items: list[GridItem] = []
    for speaker in config.speakers:
        for phrase in config.phrases:
            for ref_clip_idx in ref_clip_indices or config.grid.ref_clip_indices:
                if ref_clip_idx >= len(speaker.ref_clips):
                    # El 2do clip de referencia es un chequeo de sensibilidad y no todos los
                    # hablantes tienen que tenerlo; se saltea en vez de abortar la corrida.
                    event(
                        logger,
                        "ref_clip_missing",
                        level=logging.WARNING,
                        speaker_id=speaker.id,
                        ref_clip_idx=ref_clip_idx,
                        available=len(speaker.ref_clips),
                    )
                    continue
                for seed in config.grid.seeds:
                    items.append(
                        GridItem(
                            speaker=speaker,
                            phrase=phrase,
                            seed=seed,
                            ref_clip_idx=ref_clip_idx,
                            generation_id=make_generation_id(
                                config.model.name, speaker.id, phrase.id, seed, ref_clip_idx
                            ),
                            wav_rel_path=wav_rel_path(
                                config.model.audio_root,
                                config.model.name,
                                speaker.id,
                                phrase.id,
                                seed,
                                ref_clip_idx,
                            ),
                        )
                    )
    return items


def filter_grid(items: Iterable[GridItem], selection: GridFilter) -> list[GridItem]:
    result = []
    for item in items:
        if selection.phrase_ids and item.phrase.id not in selection.phrase_ids:
            continue
        if selection.speaker_ids and item.speaker.id not in selection.speaker_ids:
            continue
        if selection.categories and item.phrase.category.value not in selection.categories:
            continue
        if selection.seeds and item.seed not in selection.seeds:
            continue
        if selection.ref_clip_indices and item.ref_clip_idx not in selection.ref_clip_indices:
            continue
        result.append(item)
        if selection.limit is not None and len(result) >= selection.limit:
            break
    return result


def pending_items(
    items: Sequence[GridItem],
    existing: dict[str, GenerationRow],
    root: Path,
    *,
    retry_errors: bool = False,
) -> tuple[list[GridItem], int]:
    """Split the selection into what still has to run and how much is already done.

    A row alone is not proof: the wav has to be on disk too, or the cell is pending.
    """
    todo: list[GridItem] = []
    skipped = 0
    for item in items:
        row = existing.get(item.generation_id)
        if row is None:
            todo.append(item)
            continue
        if row.error is not None:
            if retry_errors:
                todo.append(item)
            else:
                skipped += 1
            continue
        wav = root / row.wav_path if row.wav_path else None
        if wav is not None and wav.exists():
            skipped += 1
            continue
        event(
            logger,
            "wav_missing_regenerating",
            level=logging.WARNING,
            generation_id=item.generation_id,
            wav_path=row.wav_path,
        )
        todo.append(item)
    return todo, skipped


def _finish_row(
    config: BenchConfig,
    adapter: TTSAdapter,
    described: dict[str, str | None],
    *,
    generation_id: str,
    speaker: Speaker,
    phrase_id: str,
    seed: int,
    ref_clip_idx: int,
    wav_rel: Path,
    result: SynthesisResult,
    wall_ms: float,
    warmup: bool,
) -> GenerationRow:
    """Turn a `SynthesisResult` into a manifest row, filling in what the adapter left out."""
    total_ms = result.total_ms if result.total_ms is not None else wall_ms
    duration_s = result.duration_s
    rtf = None
    if result.ok and duration_s:
        rtf = (total_ms / 1000.0) / duration_s

    ref_seconds_used = result.ref_seconds_used
    # El fallback estima desde el clip declarado, y eso vale SOLO para los que clonan. Un modelo
    # que elige la voz por id no consume referencia: ahi `None` es el dato, no un hueco que
    # rellenar, y rellenarlo haria que la fila declare 6 s que nunca escucho (SPEC 6.5).
    if ref_seconds_used is None and not adapter.selects_voice_by_id:
        declared = speaker.ref_clip(ref_clip_idx).duration_s
        ref_seconds_used = (
            min(declared, adapter.max_ref_seconds) if declared else adapter.max_ref_seconds
        )

    return GenerationRow(
        generation_id=generation_id,
        model=config.model.name,
        model_revision=described.get("model_revision") or config.model.model_revision,
        lib_version=described.get("lib_version"),
        torch_version=described.get("torch_version"),
        device=described.get("device") or config.model.device,
        gpu=described.get("gpu"),
        speaker_id=speaker.id,
        ref_clip_idx=ref_clip_idx,
        ref_seconds_used=ref_seconds_used if result.ok else None,
        # Lo dice el adapter, no el config: `requires_ref_text` es la intencion declarada y esto es
        # lo que efectivamente recibio el modelo en esta fila.
        ref_text_provided=(
            result.ref_text_provided
            if result.ref_text_provided is not None
            else adapter.requires_ref_text
        ),
        phrase_id=phrase_id,
        seed=seed,
        wav_path=str(wav_rel) if result.ok else None,
        sample_rate=result.sample_rate,
        duration_s=duration_s,
        ttfb_ms=result.ttfb_ms,
        total_ms=total_ms,
        rtf=rtf,
        streaming=(
            result.streaming if result.streaming is not None else adapter.supports_streaming
        ),
        warmup=warmup,
        error=result.error,
        created_at=utcnow(),
    )


def _synthesize_one(
    adapter: TTSAdapter,
    text: str,
    ref_wav: Path,
    seed: int,
    out_path: Path,
    ref_text: str | None = None,
    speaker_id: str | None = None,
) -> tuple[SynthesisResult, float]:
    """Call the adapter, converting any exception into an error result.

    Success with no file on disk is a failure, not a silent hole in the dataset.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        result = adapter.synthesize(
            text, ref_wav, seed, out_path, ref_text=ref_text, speaker_id=speaker_id
        )
    except Exception as err:  # noqa: BLE001 — un modelo que explota es un dato, no un abort
        logger.exception("synthesis raised", extra={"fields": {"out_path": str(out_path)}})
        result = SynthesisResult(error=f"{type(err).__name__}: {err}")
    wall_ms = (time.perf_counter() - started) * 1000.0

    if result.ok and not out_path.exists():
        result.error = "adapter reported success but wrote no file"
    if not result.ok and out_path.exists():
        # Un wav parcial haria que la proxima corrida crea que esta celda ya esta hecha.
        out_path.unlink()
    return result, wall_ms


def run_warmup(
    config: BenchConfig,
    adapter: TTSAdapter,
    described: dict[str, str | None],
    seed_item: GridItem,
    writer: JsonlWriter,
    *,
    count: int,
    start_index: int,
) -> int:
    """Generate throwaway syntheses, recorded but marked, before any timing is trusted."""
    text = config.grid.warmup_text or seed_item.phrase.text
    ref_clip = seed_item.speaker.ref_clip(seed_item.ref_clip_idx)
    ref_wav = config.resolve(ref_clip.path)
    done = 0
    for offset in range(count):
        index = start_index + offset
        generation_id = make_warmup_id(config.model.name, index)
        wav_rel = config.model.audio_root / config.model.name / "_warmup" / f"warmup{index}.wav"
        seed_everything(seed_item.seed)
        result, wall_ms = _synthesize_one(
            adapter,
            text,
            ref_wav,
            seed_item.seed,
            config.resolve(wav_rel),
            ref_text=ref_clip.transcript,
            speaker_id=seed_item.speaker.id,
        )
        writer.write(
            _finish_row(
                config,
                adapter,
                described,
                generation_id=generation_id,
                speaker=seed_item.speaker,
                phrase_id=seed_item.phrase.id,
                seed=seed_item.seed,
                ref_clip_idx=seed_item.ref_clip_idx,
                wav_rel=wav_rel,
                result=result,
                wall_ms=wall_ms,
                warmup=True,
            )
        )
        done += 1
        event(
            logger,
            "warmup_done",
            generation_id=generation_id,
            ok=result.ok,
            total_ms=round(result.total_ms or wall_ms, 1),
        )
    return done


def count_warmup_rows(rows: Iterable[GenerationRow]) -> int:
    return sum(1 for row in rows if row.warmup)


def describe_grid(config: BenchConfig, items: Sequence[GridItem], skipped: int) -> None:
    """Log what a run would do. This is all `--dry-run` does."""
    event(
        logger,
        "grid_summary",
        model=config.model.name,
        adapter=config.model.adapter,
        speakers=len(config.speakers),
        phrases=len(config.phrases),
        seeds=config.grid.seeds,
        # Los indices REALES de la seleccion, no los del config: con el override de CLI los dos
        # difieren, y un resumen que informa el del config describe una corrida que no es esta.
        ref_clip_indices=sorted({item.ref_clip_idx for item in items}),
        selected=len(items),
        already_done=skipped,
        pending=len(items) - skipped,
        categories=categories_covered(config.phrases),
    )
    for item in items[:10]:
        event(
            logger,
            "grid_item",
            level=logging.DEBUG,
            generation_id=item.generation_id,
            wav_path=str(item.wav_rel_path),
            category=item.phrase.category.value,
        )


def run(
    config: BenchConfig,
    *,
    selection: GridFilter | None = None,
    dry_run: bool = False,
    retry_errors: bool = False,
    warmup: bool = True,
    manifest_path: Path | None = None,
) -> RunStats:
    """Execute (or just plan) one model's pass over the grid."""
    stats = RunStats()
    chosen = (selection or GridFilter()).ref_clip_indices
    # Ordenado y no el set: el orden de la grilla tiene que ser determinista (SPEC 6.2).
    grid = build_grid(config, sorted(chosen) if chosen else None)
    stats.total_grid = len(grid)
    items = filter_grid(grid, selection or GridFilter())
    stats.selected = len(items)

    manifest = manifest_path or config.resolve(config.model.manifest_path)
    existing_rows = list(iter_rows(manifest, GenerationRow, strict=False))
    existing = index_by_generation_id(existing_rows)
    todo, skipped = pending_items(items, existing, config.root, retry_errors=retry_errors)
    stats.skipped = skipped

    if dry_run:
        describe_grid(config, items, skipped)
        return stats

    if not todo:
        event(logger, "nothing_to_do", **stats.as_fields())
        return stats

    adapter = build_adapter(config.model)
    load_started = time.perf_counter()
    try:
        adapter.load()
    except ImportError as err:
        # El error mas frecuente de todos: se corrio el modelo desde el venv equivocado. Merece un
        # mensaje que diga que hacer, no un traceback sobre un import que falta.
        raise AdapterError(
            f"{config.model.name}: could not load the model ({err}). Its dependencies live in a "
            f"separate environment on purpose — see envs/ and check which venv is active"
        ) from err
    described = adapter.describe()
    provenance: dict[str, Any] = {
        key: value for key, value in described.items() if value is not None
    }
    event(
        logger,
        "model_loaded",
        model=config.model.name,
        load_ms=round((time.perf_counter() - load_started) * 1000.0, 1),
        pending=len(todo),
        **provenance,
    )

    try:
        with JsonlWriter(manifest) as writer:
            if warmup and config.grid.warmup_syntheses:
                stats.warmup = run_warmup(
                    config,
                    adapter,
                    described,
                    todo[0],
                    writer,
                    count=config.grid.warmup_syntheses,
                    start_index=count_warmup_rows(existing_rows),
                )

            for position, item in enumerate(todo, start=1):
                seeded = seed_everything(item.seed)
                ref_clip = item.speaker.ref_clip(item.ref_clip_idx)
                result, wall_ms = _synthesize_one(
                    adapter,
                    item.phrase.text,
                    config.resolve(ref_clip.path),
                    item.seed,
                    config.resolve(item.wav_rel_path),
                    ref_text=ref_clip.transcript,
                    speaker_id=item.speaker.id,
                )
                writer.write(
                    _finish_row(
                        config,
                        adapter,
                        described,
                        generation_id=item.generation_id,
                        speaker=item.speaker,
                        phrase_id=item.phrase.id,
                        seed=item.seed,
                        ref_clip_idx=item.ref_clip_idx,
                        wav_rel=item.wav_rel_path,
                        result=result,
                        wall_ms=wall_ms,
                        warmup=False,
                    )
                )
                if result.ok:
                    stats.generated += 1
                else:
                    stats.failed += 1
                    stats.errors.append(f"{item.generation_id}: {result.error}")
                event(
                    logger,
                    "synthesis_done" if result.ok else "synthesis_failed",
                    level=logging.INFO if result.ok else logging.ERROR,
                    generation_id=item.generation_id,
                    progress=f"{position}/{len(todo)}",
                    category=item.phrase.category.value,
                    seeded=seeded,
                    duration_s=result.duration_s,
                    ttfb_ms=None if result.ttfb_ms is None else round(result.ttfb_ms, 1),
                    total_ms=round(result.total_ms or wall_ms, 1),
                    error=result.error,
                )
    finally:
        adapter.unload()

    # Lo que la corrida costo a la maquina, al lado de lo que produjo (SPEC 7.3). Va aca y no en
    # `finally`: una corrida abortada no deja un pico que describa el grid.
    written = resources.write(
        manifest, device=config.model.device, generated=stats.generated, warmup=stats.warmup
    )
    event(
        logger,
        "run_finished",
        model=config.model.name,
        resources=str(written),
        **resources.snapshot(),
        **stats.as_fields(),
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtb.generate",
        description="Generate the benchmark grid for one TTS model.",
    )
    parser.add_argument("--config", required=True, type=Path, help="path to configs/models/*.yaml")
    parser.add_argument(
        "--model",
        help="expected model name; fails if it does not match the config, to catch copy-paste",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--phrases", type=Path, default=DEFAULT_PHRASES_PATH)
    parser.add_argument("--speakers", type=Path, default=DEFAULT_SPEAKERS_PATH)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--manifest", type=Path, help="override the manifest path")
    parser.add_argument("--dry-run", action="store_true", help="print the grid and exit")

    parser.add_argument("--phrase-ids", help="comma-separated subset of phrase ids")
    parser.add_argument("--speaker-ids", help="comma-separated subset of speaker ids")
    parser.add_argument("--categories", help="comma-separated subset of phrase categories")
    parser.add_argument("--seeds", help="comma-separated subset of seeds")
    parser.add_argument("--ref-clip-indices", help="comma-separated subset of reference clips")
    parser.add_argument("--limit", type=int, help="stop after N grid items")

    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="re-run grid cells whose manifest row has an error",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip warm-up syntheses; timings from this run are not comparable",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--plain-logs", action="store_true", help="human-readable logs, not JSON")
    return parser


def _split_str(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _split_int(value: str | None) -> set[int] | None:
    parts = _split_str(value)
    return None if parts is None else {int(part) for part in parts}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level, json_lines=not args.plain_logs)

    try:
        config = load_bench_config(
            args.config,
            root=args.root,
            phrases=args.phrases,
            speakers=args.speakers,
            grid=args.grid,
            # En seco no hace falta el corpus; en una corrida real, un clip de referencia que
            # falta tiene que abortar antes de sintetizar 400 wavs contra la voz equivocada.
            check_paths=not args.dry_run,
        )
        if args.model and args.model != config.model.name:
            raise ConfigError(
                f"--model={args.model!r} does not match config name {config.model.name!r}"
            )
        selection = GridFilter(
            phrase_ids=_split_str(args.phrase_ids),
            speaker_ids=_split_str(args.speaker_ids),
            categories=_split_str(args.categories),
            seeds=_split_int(args.seeds),
            ref_clip_indices=_split_int(args.ref_clip_indices),
            limit=args.limit,
        )
        stats = run(
            config,
            selection=selection,
            dry_run=args.dry_run,
            retry_errors=args.retry_errors,
            warmup=not args.no_warmup,
            manifest_path=args.manifest,
        )
    except (ConfigError, AdapterError) as err:
        event(logger, "run_aborted", level=logging.CRITICAL, error=str(err))
        return 1

    # Las sintesis fallidas son un resultado y quedan en el manifest: no cambian el exit code.
    if stats.failed:
        event(logger, "failures_recorded", level=logging.WARNING, failed=stats.failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
