"""Every metric that needs no audio, and the tables for them (SPEC 7).

    python -m rtb.metrics --manifest data/manifests/xtts_v2.jsonl

Writes `duration_screen.jsonl` (triage, gitignored) and `failures.jsonl` (a result, committed).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rtb.config import (
    DEFAULT_PHRASES_PATH,
    DEFAULT_SPEAKERS_PATH,
    DEFAULT_THRESHOLDS_PATH,
    ConfigError,
    Phrase,
    Speaker,
    Thresholds,
    load_phrases,
    load_speakers,
    load_thresholds,
)
from rtb.logconf import event, setup_logging
from rtb.manifest import JsonlWriter, index_by_generation_id, read_rows
from rtb.metrics import anatomy as anatomy_metric
from rtb.metrics import duration_screen as screen_metric
from rtb.metrics import failures as failures_metric
from rtb.metrics import latency as latency_metric
from rtb.metrics import speaker_sim as sim_metric
from rtb.metrics import speaking_rate
from rtb.metrics.asr import METRIC_NAME as TRANSCRIPTS_METRIC
from rtb.metrics.asr import installed_version
from rtb.metrics.audio_shape import METRIC_NAME as SHAPE_METRIC
from rtb.report import tables
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, MetricRow

logger = logging.getLogger("rtb.metrics")


def model_of(generation_id: str) -> str:
    """The model that produced a row, read off its `generation_id`.

    Ids are `<model>__<speaker>__<phrase>__s<seed>__ref<idx>`, warm-ups are `<model>__warmupN`, and
    real audio is `gt__...` — so the first token identifies the source in every case.
    """
    return generation_id.split("__", 1)[0]


def dropped_models(out_path: Path, rows: Sequence[MetricRow]) -> set[str]:
    """Models present in the results file on disk that this run did not recompute.

    The results files are rewritten whole, so a run that omits a manifest deletes that model's rows
    and nothing downstream looks wrong. It happened once, with the real-audio floor.
    """
    if not out_path.exists():
        return set()
    on_disk = {
        model_of(row.generation_id)
        for row in read_rows(out_path, MetricRow, strict=False)
    }
    return on_disk - {model_of(row.generation_id) for row in rows}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtb.metrics",
        description="Compute every metric that does not need audio.",
    )
    parser.add_argument("--manifest", required=True, type=Path, action="append")
    # No es un --manifest mas que haya que acordarse de pasar: es la escala de todas las tablas,
    # asi que se carga solo desde su ruta por defecto.
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/manifests/ground_truth.jsonl"),
        help="real-audio manifest; loaded automatically when present",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="write results with no real-audio floor. Only for a checkout without the corpus",
    )
    parser.add_argument(
        "--drop-missing-models",
        action="store_true",
        help="allow rewriting a results file without models it already contains, deleting them",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--phrases", type=Path, default=DEFAULT_PHRASES_PATH)
    parser.add_argument("--speakers", type=Path, default=DEFAULT_SPEAKERS_PATH)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/metrics"))
    parser.add_argument("--dry-run", action="store_true", help="print tables, write nothing")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="also show how much each rate moves when the thresholds move",
    )
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--plain-logs", action="store_true")
    return parser


def report_model(
    model: str,
    rows: list[GenerationRow],
    ground_truth: list[GenerationRow],
    phrases: list[Phrase],
    speakers: list[Speaker],
    thresholds: Thresholds,
    transcripts: dict[str, MetricRow],
    shapes: dict[str, MetricRow],
    sims: dict[str, MetricRow],
    anatomies: dict[str, anatomy_metric.Anatomy],
    scale: sim_metric.Scale,
    asr_version: str,
    show_sensitivity: bool,
) -> tuple[list[MetricRow], list[MetricRow]]:
    """Print every table for one model. Returns (screen rows, failure rows)."""
    categories = {phrase.id: phrase.category.value for phrase in phrases}

    print(f"\n## {model}\n")
    print("### Latencia\n")
    summaries = [
        latency_metric.summarize(rows, label=model),
        *latency_metric.summarize_by_category(latency_metric.measurable(rows), categories),
    ]
    print(tables.latency_table(summaries))
    measured = speaking_rate.speaking_rates(speakers)
    print(
        "\nTasa de habla de referencia, medida sobre los clips reales: "
        + ", ".join(f"{key} {value:.3f} s/palabra" for key, value in sorted(measured.items()))
    )

    model_anatomies = {
        row.generation_id: anatomies[row.generation_id]
        for row in rows
        if not row.warmup and row.generation_id in anatomies
    }
    tail_threshold = thresholds.failures.tail_after_last_word_s
    if model_anatomies:
        print("\n### Anatomia del audio\n")
        print(
            tables.anatomy_table(
                [
                    anatomy_metric.summarize(model_anatomies.values(), tail_threshold, model),
                    *anatomy_metric.summarize_by_category(
                        model_anatomies, rows, categories, tail_threshold
                    ),
                ]
            )
        )
        print(
            "\nEl clip cortado en cuatro: silencio inicial + habla + cola + silencio final. La "
            "cola es sonido DESPUES de la ultima palabra transcripta, y el corte la saca de "
            "`speech` —donde entraba por estar arriba del umbral de silencio— para contarla "
            "aparte. `trailing p95` importa mas que la mediana: es latencia que el RTF no ve, un "
            "turno que no cierra."
        )

    speech_seconds = anatomy_metric.speech_seconds(model_anatomies) if model_anatomies else None
    screen_rows = screen_metric.screen(rows, phrases, speakers, thresholds, speech_seconds)
    screen_overall = screen_metric.rates(screen_rows)
    print("\n### Screening de duracion (triage, NO la tasa de fallo publicada)\n")
    print(tables.screen_table(screen_metric.rates_by_category(screen_rows), screen_overall))
    if speech_seconds is None:
        print(
            f"\nSin juzgar: {screen_overall.unjudged}. Umbrales version {thresholds.version}. "
            "Juzga la duracion del ARCHIVO, que incluye el relleno y la cola: sin transcripts no "
            "hay forma de separarlos. Corré el pase de ASR y esta tabla pasa a juzgar el habla."
        )
    else:
        # La comparacion es el hallazgo: la version sobre el archivo marcaba colas y las llamaba
        # duraciones equivocadas, o sea que el mismo defecto se contaba dos veces con dos nombres.
        on_duration = screen_metric.rates(
            screen_metric.screen(rows, phrases, speakers, thresholds)
        )
        before = on_duration.percent(on_duration.too_long)
        after = screen_overall.percent(screen_overall.too_long)
        print(
            f"\nSin juzgar: {screen_overall.unjudged}. Umbrales version {thresholds.version}. "
            "Juzga el HABLA PEDIDA —hasta la ultima palabra transcripta—, no la duracion del "
            f"archivo. Sobre el archivo la misma regla marcaba {before:.1f} % de largos, contra "
            f"{after:.1f} % ahora: la diferencia era cola, que la taxonomia de fallas ya cuenta "
            "como `trailing`."
        )

    failure_rows: list[MetricRow] = []
    if not transcripts:
        print(
            "\n[sin transcripts todavia: corré `python -m rtb.metrics.transcribe` en el entorno "
            "de evaluacion para inteligibilidad y tasa de fallo]"
        )
    else:
        failure_rows = failures_metric.evaluate(
            rows, phrases, transcripts, shapes, thresholds, asr_version
        )
        gt_rows = failures_metric.evaluate(
            ground_truth, phrases, transcripts, shapes, thresholds, asr_version
        )
        gt_rates = failures_metric.rates(gt_rows) if gt_rows else None
        by_category = failures_metric.rates_by_category(failure_rows)

        print("\n### Inteligibilidad\n")
        print(tables.intelligibility_table(by_category, gt_rates))
        print(
            "\nEl audio real va en la MISMA tabla a proposito: un WER sin ese piso no significa "
            "nada. `unscorable` son las filas donde el ASR escribio digitos donde la referencia "
            "tiene palabras, asi que su alineamiento mide la convencion del ASR y no al modelo."
        )
        print("\n### Tasa de fallo\n")
        print(tables.failure_table(by_category, failures_metric.rates(failure_rows), gt_rates))
        print(
            "\nPorcentajes sobre las filas clasificables. `halluc` son palabras de mas, `trunc` "
            "palabras faltantes al final, `loop` repeticion CON material extra, `tail` sonido "
            "despues de la ultima palabra. Casi nada de eso es un umbral: sale del alineamiento."
        )

    model_sims = [sims[row.generation_id] for row in rows if row.generation_id in sims]
    if not model_sims:
        print(
            "\n[sin similitud de hablante todavia: corré `python -m rtb.metrics.speaker_sim` en "
            ".venv-simil — ver envs/speaker_sim.txt]"
        )
    else:
        print("\n### Similitud de hablante\n")
        print(tables.speaker_scale_table(scale))
        print(f"\n{scale.diagnostic_verdict()}\n")
        print(
            tables.speaker_sim_table(
                [
                    sim_metric.summarize(model_sims, scale, label=model),
                    # La duracion va PRIMERO porque es el factor de confusion: leer las
                    # categorias sin verla lleva a atribuirle al clonado lo que es largo de clip.
                    *sim_metric.summarize_by_duration(model_sims, shapes, scale),
                    *sim_metric.summarize_by(model_sims, scale, "category"),
                    *sim_metric.summarize_by(model_sims, scale, "speaker_id"),
                    *sim_metric.summarize_by(model_sims, scale, "ref_clip_idx"),
                ]
            )
        )
        print(
            "\n`scaled p50` es donde cae el coseno entre el piso (0) y el techo (1); `vs` dice "
            "CONTRA CUAL techo, porque hay dos: un clip de menos de un segundo no alcanza el "
            "techo largo ni cuando el audio es real, y para eso esta `short_ceiling`. El coseno "
            "crudo depende del checkpoint, asi que la columna escalada es la unica comparable "
            "entre benchmarks."
        )
        # Las cifras de esta nota se MIDEN sobre las filas de este modelo: hardcodeadas, una
        # salvedad correcta se imprime debajo de otro modelo y pasa a ser falsa.
        correlation = sim_metric.duration_correlation(model_sims, shapes)
        band = sim_metric.category_spread_in_band(model_sims, shapes)
        sim_by_category = sim_metric.summarize_by(model_sims, scale, "category")
        category_medians = [s.median for s in sim_by_category if s.median is not None]
        note = "\nLAS FILAS POR CATEGORIA ESTAN CONFUNDIDAS CON LA DURACION y no se leen solas"
        if correlation is not None:
            note += f": la correlacion entre segundos de habla y coseno es {correlation:.2f}"
        if band is not None and category_medians:
            note += (
                f", y dentro de la banda de {band.low:g} a {band.high:g} s las {band.categories} "
                f"categorias que conviven ahi caen entre {band.lowest:.2f} y {band.highest:.2f} "
                f"—{band.spread:.2f} de dispersion donde la tabla por categoria muestra "
                f"{max(category_medians) - min(category_medians):.2f}—"
            )
        print(note + ". El orden por categoria es, en su mayor parte, largo de clip.")

        short = next(
            (s for s in sim_metric.summarize_by_duration(model_sims, shapes, scale)
             if s.ceiling_kind == sim_metric.SHORT_CEILING),
            None,
        )
        overall = sim_metric.summarize(model_sims, scale)
        if short and short.normalized_median and overall.normalized_median:
            verdict = (
                "igual queda"
                if short.normalized_median < overall.normalized_median
                else "queda"
            )
            print(
                f"\nCorregido por duracion —contra el techo corto, medido en audio real de la "
                f"misma duracion— el tramo de menos de un segundo {verdict} en "
                f"{short.normalized_median:.3f} contra {overall.normalized_median:.3f} del "
                "conjunto. O sea que sobre el sesgo de duracion hay un deficit de clonado propio "
                "de los enunciados cortos. Los tramos de 1 a 4 s no tienen techo pareado, asi que "
                "su columna escalada es una cota inferior."
            )

    if show_sensitivity:
        print("\n### Sensibilidad de los umbrales\n")
        points = screen_metric.sensitivity(rows, phrases, speakers, thresholds, speech_seconds)
        if failure_rows:
            points.extend(failures_metric.tail_sensitivity(failure_rows, thresholds.failures))
            points.extend(failures_metric.boundary_sensitivity(failure_rows, thresholds.failures))
            points.extend(failures_metric.loop_sensitivity(failure_rows, thresholds.failures))
        print(tables.sensitivity_table(points))
        print(
            "\nUna tasa que se mueve del 85 % al 15 % barriendo umbrales plausibles reporta el "
            "umbral y no el fenomeno. La fila marcada `(actual)` es la que produjo los numeros de "
            "arriba."
        )
    return screen_rows, failure_rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level, json_lines=not args.plain_logs)
    root = Path(args.root).resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        phrases = load_phrases(resolve(args.phrases))
        speakers = load_speakers(resolve(args.speakers))
        thresholds = load_thresholds(resolve(args.thresholds))
    except ConfigError as err:
        event(logger, "metrics_aborted", level=logging.CRITICAL, error=str(err))
        return 1

    metrics_dir = resolve(args.out)
    # Opcionales: si el pase de audio no corrio, se reporta lo que se puede y se dice que falta.
    transcripts = {
        row.generation_id: row
        for row in read_rows(metrics_dir / f"{TRANSCRIPTS_METRIC}.jsonl", MetricRow, strict=False)
    }
    shapes = {
        row.generation_id: row
        for row in read_rows(metrics_dir / f"{SHAPE_METRIC}.jsonl", MetricRow, strict=False)
    }
    sims = {
        row.generation_id: row
        for row in read_rows(
            metrics_dir / f"{sim_metric.METRIC_NAME}.jsonl", MetricRow, strict=False
        )
    }
    # Derivada al reportar, uniendo la envolvente con el ASR: `audio_shape` es ciego al transcript a
    # proposito, asi que el cuarto tramo —la cola— no puede vivir en su archivo (SPEC 7.5).
    anatomies = anatomy_metric.from_rows(shapes, transcripts)
    # La escala se reconstruye del archivo, no se recalcula: leerla no necesita audio ni torch, que
    # es lo que permite reportar en cualquier maquina.
    scale = sim_metric.scale_from_rows(
        read_rows(metrics_dir / f"{sim_metric.SCALE_METRIC_NAME}.jsonl", MetricRow, strict=False)
    )
    asr_version = next(
        (str(row.metric_version) for row in transcripts.values()), installed_version()
    )

    # Los manifests se leen TODOS antes de reportar: el ground truth es la escala de las tablas de
    # cada modelo, asi que no puede depender del orden en que se pasaron los --manifest.
    by_model: dict[str, list[GenerationRow]] = {}
    ground_truth: list[GenerationRow] = []
    # La ruta por defecto se agrega sola, y de ultima, para que pasarla explicitamente por
    # --manifest no la duplique.
    manifests = list(args.manifest)
    default_gt = resolve(args.ground_truth)
    already_passed = any(resolve(path) == default_gt for path in manifests)
    if not args.no_ground_truth and default_gt.exists() and not already_passed:
        manifests.append(args.ground_truth)
    for manifest_path in manifests:
        path = resolve(manifest_path)
        # Append-only: vale la ULTIMA fila de cada generation_id, no los reintentos.
        rows = list(index_by_generation_id(read_rows(path, GenerationRow)).values())
        if not rows:
            print(f"\n## {path.name}: sin filas\n")
            continue
        if rows[0].model == GROUND_TRUTH_MODEL:
            ground_truth.extend(rows)
            seconds = sum(row.duration_s or 0.0 for row in rows)
            print(
                f"\n## {path.name}: {len(rows)} filas de audio real ({seconds:.1f} s). "
                "Sin latencia ni screening: son la escala de las otras tablas.\n"
            )
            continue
        by_model.setdefault(rows[0].model, []).extend(rows)

    all_screen: list[MetricRow] = []
    all_failures: list[MetricRow] = []
    for model, rows in by_model.items():
        screen_rows, failure_rows = report_model(
            model,
            rows,
            ground_truth,
            phrases,
            speakers,
            thresholds,
            transcripts,
            shapes,
            sims,
            anatomies,
            scale,
            asr_version,
            args.sensitivity,
        )
        all_screen.extend(screen_rows)
        all_failures.extend(failure_rows)

    if args.dry_run:
        print("\n[dry-run: no se escribio ningun archivo de metricas]")
        return 0

    if all_failures and not ground_truth and not args.no_ground_truth:
        # Se aborta ANTES de escribir: el archivo se reescribe entero, asi que sin ground truth no
        # queda incompleto, BORRA el piso de audio real que ya estaba commiteado (SPEC 7.1).
        event(
            logger,
            "metrics_aborted",
            level=logging.CRITICAL,
            error=(
                f"no real-audio rows: {default_gt} is missing. Build it with "
                "`python -m rtb.metrics.ground_truth`, or pass --no-ground-truth to write results "
                "with no floor on purpose"
            ),
        )
        print(
            f"\n[ABORTADO sin escribir nada: no hay filas de audio real. {default_gt} no existe. "
            "Sobrescribir failures.jsonl sin el piso de audio real borraria la escala de toda la "
            "tabla. Generalo con `python -m rtb.metrics.ground_truth`, o forzá con "
            "--no-ground-truth]"
        )
        return 1

    # Las filas de ground truth tambien van al archivo de fallas: son parte del resultado, y sin
    # ellas nadie puede recalcular la escala sin volver a correr esto.
    if all_failures and ground_truth:
        all_failures.extend(
            failures_metric.evaluate(
                ground_truth, phrases, transcripts, shapes, thresholds, asr_version
            )
        )

    pending_writes = [
        (metrics_dir / f"{name}.jsonl", metric_rows)
        for name, metric_rows in [
            (screen_metric.METRIC_NAME, all_screen),
            (failures_metric.METRIC_NAME, all_failures),
        ]
        if metric_rows
    ]

    # TODOS los archivos se chequean antes de escribir NINGUNO: abortar a mitad de camino dejaria
    # un failures.jsonl nuevo junto a un duration_screen.jsonl viejo, que es peor que no correr.
    if not args.drop_missing_models:
        for out_path, metric_rows in pending_writes:
            dropped = dropped_models(out_path, metric_rows)
            if not dropped:
                continue
            missing = ", ".join(f"data/manifests/{model}.jsonl" for model in sorted(dropped))
            event(
                logger,
                "metrics_aborted",
                level=logging.CRITICAL,
                error=f"{out_path.name} would lose models {sorted(dropped)}; missing {missing}",
            )
            print(
                f"\n[ABORTADO sin escribir nada: {out_path.name} ya tiene resultados de "
                f"{', '.join(sorted(dropped))} y esta corrida no los recalculo. El archivo se "
                f"reescribe ENTERO, asi que escribirlo ahora los borraria. Agregá --manifest "
                f"{missing}, o pasá --drop-missing-models si de verdad los querés sacar]"
            )
            return 1

    for out_path, metric_rows in pending_writes:
        # Se reescribe entero: es un recalculo determinista, y dos veredictos para el mismo
        # generation_id serian ambiguos.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        with JsonlWriter(out_path) as writer:
            for metric_row in metric_rows:
                writer.write(metric_row)
        print(f"\n[escrito {out_path} con {len(metric_rows)} filas]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
