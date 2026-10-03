"""Latency aggregation and the duration screen.

Both run on the manifest alone, so they are fully testable here — no audio, no models, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rtb.config import Phrase, Speaker, load_speakers, load_thresholds
from rtb.manifest import read_rows
from rtb.metrics import duration_screen, latency, speaking_rate
from rtb.report import tables
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, MetricRow

from .conftest import REPO_ROOT, make_thresholds
from .test_schema import ok_row

THRESHOLDS = make_thresholds()


def phrase(**overrides: object) -> Phrase:
    base: dict[str, object] = {
        "id": "vos_001",
        "text": "Vos tenés razón, pero no me lo dijiste antes.",
        "category": "voseo",
        "failure_hypothesis": "x",
    }
    base.update(overrides)
    return Phrase.model_validate(base)


def speaker(**overrides: object) -> Speaker:
    base: dict[str, object] = {
        "id": "slr61_f_01",
        "gender": "F",
        "ref_clips": [{"path": "r.wav"}],
        "control_clips": [
            # 10 palabras en 3.0 s de habla, mas un gap de 250 ms por la concatenacion.
            {
                "path": "c1.wav",
                "duration_s": 3.25,
                "transcript": "una dos tres cuatro cinco seis siete ocho nueve diez",
                "source_utterances": ["a", "b"],
            },
            {"path": "c2.wav", "duration_s": 1.0, "transcript": "once doce trece cuatorce"},
        ],
    }
    base.update(overrides)
    return Speaker.model_validate(base)


# ── tasa de habla ────────────────────────────────────────────────────────────────────────────


def test_speaking_rate_discounts_the_concatenation_gaps() -> None:
    # 3.25 s con un gap de 0.25 son 3.0 s de habla en 10 palabras; mas 1.0 s en 4 palabras.
    # Sin descontar el gap la tasa saldria inflada y todo pareceria durar de menos.
    rate = speaking_rate.speaking_rate(speaker())
    assert rate == pytest.approx((3.0 + 1.0) / 14)


def test_speaking_rate_ignores_clips_without_a_transcript() -> None:
    quiet = speaker(
        control_clips=[
            {"path": "c1.wav", "duration_s": 3.0, "transcript": "una dos tres"},
            {"path": "c2.wav", "duration_s": 9.0},
        ]
    )
    assert speaking_rate.speaking_rate(quiet) == pytest.approx(1.0)


def test_speaking_rate_is_none_without_usable_clips() -> None:
    assert (
        speaking_rate.speaking_rate(speaker(control_clips=[{"path": "a.wav"}, {"path": "b.wav"}]))
        is None
    )


def test_committed_speaking_rates_are_plausible(repo_root: Path) -> None:
    # Un cambio en speakers.yaml que rompa la tasa mueve TODA la tasa de fallo, en silencio.
    rates = speaking_rate.speaking_rates(load_speakers(repo_root / "configs" / "speakers.yaml"))
    assert len(rates) == 4
    for speaker_id, rate in rates.items():
        assert 0.15 < rate < 0.45, f"{speaker_id}: {rate:.3f} s/palabra es implausible"


# ── veredicto de duracion ────────────────────────────────────────────────────────────────────


def test_a_normal_duration_is_not_flagged() -> None:
    verdict = duration_screen.judge_duration(2.7, 9, 0.29, THRESHOLDS.duration)
    assert (verdict.too_short, verdict.too_long, verdict.rule) == (False, False, "ratio")


def test_far_too_short_is_truncation() -> None:
    verdict = duration_screen.judge_duration(0.9, 9, 0.29, THRESHOLDS.duration)
    assert verdict.too_short and not verdict.too_long


def test_far_too_long_is_flagged_without_naming_the_cause() -> None:
    # Desde el manifest solo se ve que se paso de largo. Loop o alucinacion se distinguen con la
    # transcripcion, asi que el veredicto no elige.
    verdict = duration_screen.judge_duration(9.0, 9, 0.29, THRESHOLDS.duration)
    assert verdict.too_long
    assert not hasattr(verdict, "loop")


def test_short_utterances_use_an_absolute_ceiling() -> None:
    # "Dale." es 1 palabra: la tolerancia tiene que ser un overhead fijo, no una proporcion.
    short = phrase(id="cor_001", text="Dale.", category="enunciados_cortos")
    verdict = duration_screen.judge_duration(0.8, short.n_spoken_words, 0.29, THRESHOLDS.duration)
    assert verdict.rule == "absolute"
    assert not verdict.too_long  # 0.29 + 0.6 = 0.89 s de techo
    assert duration_screen.judge_duration(1.4, 1, 0.29, THRESHOLDS.duration).too_long


def test_the_ratio_rule_is_too_permissive_for_a_short_utterance() -> None:
    # Justificacion medible de la excepcion: la tolerancia multiplicativa CRECE con el texto. Para
    # 3 palabras a 0.29 s/palabra, 3x son 2.6 s de holgura sobre 0.87 s esperados, asi que una
    # sintesis de 2.5 s —tres veces mas larga de lo que corresponde— pasa el filtro. La tolerancia
    # aditiva se queda en 0.6 s pase lo que pase, y la caza.
    ratio_only = THRESHOLDS.duration.model_copy(update={"short_utterance_max_words": 0})
    assert not duration_screen.judge_duration(2.5, 3, 0.29, ratio_only).too_long
    assert duration_screen.judge_duration(2.5, 3, 0.29, THRESHOLDS.duration).too_long


# ── filas de metricas ────────────────────────────────────────────────────────────────────────


def test_evaluate_skips_warmup_and_ground_truth() -> None:
    rows = [
        ok_row(generation_id="m__s__vos_001__s1__ref0", phrase_id="vos_001", duration_s=2.7),
        ok_row(generation_id="m__warmup0", phrase_id="vos_001", duration_s=2.7, warmup=True),
        GenerationRow(
            generation_id="gt__x",
            model=GROUND_TRUTH_MODEL,
            speaker_id="slr61_f_01",
            ref_clip_idx=0,
            phrase_id="ctrl_00",
            wav_path="a.wav",
            sample_rate=16000,
            duration_s=3.0,
        ),
    ]
    result = duration_screen.screen(rows, [phrase()], [speaker()], THRESHOLDS)
    assert [row.generation_id for row in result] == ["m__s__vos_001__s1__ref0"]


def test_a_hard_error_is_counted_and_not_judged_on_duration() -> None:
    row = GenerationRow(
        generation_id="m__s__vos_001__s1__ref0",
        model="m",
        model_revision="abc",
        lib_version="x==1",
        device="cuda",
        speaker_id="slr61_f_01",
        ref_clip_idx=0,
        phrase_id="vos_001",
        seed=1,
        error="CUDA out of memory",
    )
    metric = duration_screen.screen([row], [phrase()], [speaker()], THRESHOLDS)[0]
    assert metric.values["hard_error"] is True
    assert "ratio" not in metric.values


def test_an_unknown_phrase_yields_an_error_row_not_a_verdict() -> None:
    # Inventar un veredicto sobre una frase que no esta en el corpus contaminaria la tasa.
    row = ok_row(generation_id="m__s__zzz_999__s1__ref0", phrase_id="zzz_999", duration_s=2.0)
    metric = duration_screen.screen([row], [phrase()], [speaker()], THRESHOLDS)[0]
    assert metric.error and "phrase not in the corpus" in metric.error


# ── la cantidad que se juzga ─────────────────────────────────────────────────────────────────


def test_the_screen_judges_the_speech_when_the_anatomy_is_available() -> None:
    # 9 s de archivo con 2.7 s de habla es la firma de una cola larga, no de una duracion
    # equivocada, y la cola ya la cuenta la taxonomia de fallas. Contarla de los dos lados era
    # contar el mismo defecto dos veces.
    row = ok_row(generation_id="m__s__vos_001__s1__ref0", phrase_id="vos_001", duration_s=9.0)
    on_duration = duration_screen.screen([row], [phrase()], [speaker()], THRESHOLDS)[0]
    assert on_duration.values["too_long"] is True

    speech = {row.generation_id: 2.7}
    on_speech = duration_screen.screen([row], [phrase()], [speaker()], THRESHOLDS, speech)[0]
    assert on_speech.values["too_long"] is False
    assert on_speech.values["measured"] == duration_screen.BY_SPEECH
    assert on_speech.values["measured_s"] == 2.7
    # La duracion del archivo sigue en la fila: es contexto, no la vara.
    assert on_speech.values["duration_s"] == 9.0


def test_a_row_with_no_transcript_is_left_unjudged_and_not_measured_on_the_file() -> None:
    # Una tasa que mezcla las dos cantidades no es una tasa de nada.
    row = ok_row(generation_id="m__s__vos_001__s1__ref0", phrase_id="vos_001", duration_s=9.0)
    metric = duration_screen.screen([row], [phrase()], [speaker()], THRESHOLDS, {})[0]
    assert metric.error and "no transcript" in metric.error


def test_metric_rows_carry_the_threshold_version() -> None:
    row = ok_row(generation_id="m__s__vos_001__s1__ref0", phrase_id="vos_001", duration_s=2.7)
    metric = duration_screen.screen([row], [phrase()], [speaker()], THRESHOLDS)[0]
    assert metric.metric_version == "test"
    assert metric.metric == "duration_screen"


def test_rates_are_percentages_of_the_total() -> None:
    rows = [
        MetricRow(
            generation_id="a", metric="failures", metric_version="1", values={"too_long": True}
        ),
        MetricRow(
            generation_id="b", metric="failures", metric_version="1", values={"too_long": False}
        ),
        MetricRow(
            generation_id="c", metric="failures", metric_version="1", values={"hard_error": True}
        ),
        MetricRow(generation_id="d", metric="failures", metric_version="1", error="unjudged"),
    ]
    result = duration_screen.rates(rows)
    assert (result.total, result.too_long, result.hard_error, result.unjudged) == (4, 1, 1, 1)
    assert result.percent(result.too_long) == pytest.approx(25.0)


# ── latencia ─────────────────────────────────────────────────────────────────────────────────


def latency_rows() -> list[GenerationRow]:
    return [
        ok_row(generation_id=f"m__s__p{i}__s1__ref0", total_ms=ms, ttfb_ms=ms, rtf=ms / 1000)
        for i, ms in enumerate([100, 200, 300, 400, 5000])
    ]


def test_median_and_p95_not_the_mean() -> None:
    # La mediana ignora el outlier de 5 s; una media lo dejaria dominar y describiria un turno
    # que no existe.
    summary = latency.summarize(latency_rows())
    assert summary.total_ms_median == 300
    assert summary.total_ms_p95 == 5000
    assert summary.n == 5


def test_warmup_and_errors_are_excluded() -> None:
    rows = [
        *latency_rows(),
        ok_row(generation_id="m__warmup0", total_ms=99000, warmup=True),
        GenerationRow(
            generation_id="m__s__x__s1__ref0",
            model="m",
            model_revision="a",
            lib_version="x==1",
            device="cuda",
            speaker_id="s",
            ref_clip_idx=0,
            phrase_id="p",
            seed=1,
            error="boom",
        ),
    ]
    assert latency.summarize(rows).n == 5


def test_mixed_hardware_is_reported_as_not_comparable() -> None:
    rows = [
        ok_row(generation_id="m__a", gpu="Tesla T4", total_ms=100),
        ok_row(generation_id="m__b", gpu="RTX 4050", total_ms=120),
    ]
    summary = latency.summarize(rows)
    assert not summary.comparable
    assert "mixed" in tables.latency_table([summary])


def test_mixing_streaming_modes_is_not_comparable() -> None:
    rows = [
        ok_row(generation_id="m__a", total_ms=100, streaming=True),
        ok_row(generation_id="m__b", total_ms=120, streaming=False),
    ]
    assert not latency.summarize(rows).comparable


def test_percentile_handles_a_single_value_and_an_empty_list() -> None:
    assert latency.percentile([42.0], 0.95) == 42.0
    assert latency.percentile([], 0.5) is None


def test_summarize_by_category_splits_the_corpus() -> None:
    rows = [
        ok_row(generation_id="m__a", phrase_id="cor_001", total_ms=300),
        ok_row(generation_id="m__b", phrase_id="lar_001", total_ms=3000),
    ]
    summaries = latency.summarize_by_category(
        rows, {"cor_001": "enunciados_cortos", "lar_001": "frase_larga"}
    )
    assert [(s.label, s.total_ms_median) for s in summaries] == [
        ("enunciados_cortos", 300),
        ("frase_larga", 3000),
    ]


# ── umbrales commiteados y sensibilidad ──────────────────────────────────────────────────────


def test_committed_thresholds_load(repo_root: Path) -> None:
    thresholds = load_thresholds(repo_root / "configs" / "thresholds.yaml")
    assert thresholds.version
    assert thresholds.duration.too_short_ratio < 1 < thresholds.duration.too_long_ratio


def test_sensitivity_includes_the_threshold_in_use() -> None:
    rows = [
        ok_row(generation_id=f"m__{i}", phrase_id="vos_001", duration_s=d)
        for i, d in enumerate([2.7, 9.0])
    ]
    points = duration_screen.sensitivity(rows, [phrase()], [speaker()], THRESHOLDS)
    in_use = [point for point in points if point.in_use]
    assert len(in_use) == 2  # uno por regla
    assert any(point.rule == "ratio" and point.flagged for point in points)


def test_sensitivity_sweeps_the_same_quantity_the_table_reports() -> None:
    # Barrer la duracion del archivo mientras la tabla publica el habla daria la sensibilidad de
    # otra tasa, que es peor que no barrer.
    rows = [
        ok_row(generation_id=f"m__{i}", phrase_id="vos_001", duration_s=9.0) for i in range(2)
    ]
    speech = {row.generation_id: 2.7 for row in rows}
    points = duration_screen.sensitivity(rows, [phrase()], [speaker()], THRESHOLDS, speech)
    assert all(point.flagged == 0 for point in points if point.rule == "ratio")
    on_duration = duration_screen.sensitivity(rows, [phrase()], [speaker()], THRESHOLDS)
    assert any(point.flagged == 2 for point in on_duration if point.rule == "ratio")


def test_markdown_table_aligns_columns() -> None:
    rendered = tables.markdown_table(["a", "bbb"], [["1", "2"]])
    assert rendered.splitlines()[0] == "| a | bbb |"


# ── el reporte no borra el piso de audio real ────────────────────────────────────────────────────


def write_metrics_repo(tmp_path: Path, *, with_ground_truth: bool) -> Path:
    """A checkout with a model manifest, a phrases file, and optionally the real-audio manifest."""
    from .conftest import PHRASES_YAML, SPEAKERS_YAML, write_wav

    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "phrases.yaml").write_text(PHRASES_YAML, encoding="utf-8")
    (tmp_path / "configs" / "speakers.yaml").write_text(SPEAKERS_YAML, encoding="utf-8")
    thresholds = (REPO_ROOT / "configs" / "thresholds.yaml").read_text(encoding="utf-8")
    (tmp_path / "configs" / "thresholds.yaml").write_text(thresholds, encoding="utf-8")
    for name in ("f01_ref0", "f01_ref1", "f01_ctrl0", "f01_ctrl1"):
        write_wav(tmp_path / "data" / "refs" / f"{name}.wav", seconds=5.0)

    manifest = tmp_path / "data" / "manifests" / "model.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    row = ok_row(generation_id="m__slr61_f_01__vos_001__s11__ref0", phrase_id="vos_001")
    manifest.write_text(row.model_dump_json() + "\n", encoding="utf-8")

    # Sin transcripts no se calcula ninguna fila de falla y no se escribe failures.jsonl, asi que
    # el guard que estos tests ejercitan no llegaria a correr.
    transcripts = tmp_path / "data" / "metrics" / "transcripts.jsonl"
    transcripts.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        MetricRow(
            generation_id=generation_id,
            metric="transcripts",
            metric_version="large-v3/int8_float16/faster-whisper==1.2.1",
            values={"text": text, "last_word_end_s": 1.0},
        ).model_dump_json()
        for generation_id, text in [
            ("m__slr61_f_01__vos_001__s11__ref0", "Vos tenés razón, pero no me lo dijiste antes."),
            ("gt__slr61_f_01__ctrl0", "una dos tres"),
        ]
    ]
    transcripts.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if with_ground_truth:
        gt = GenerationRow(
            generation_id="gt__slr61_f_01__ctrl0",
            model=GROUND_TRUTH_MODEL,
            speaker_id="slr61_f_01",
            phrase_id="ctrl0",
            reference_text="una dos tres",
            wav_path="data/refs/f01_ctrl0.wav",
            sample_rate=16000,
            duration_s=5.0,
        )
        path = tmp_path / "data" / "manifests" / "ground_truth.jsonl"
        path.write_text(gt.model_dump_json() + "\n", encoding="utf-8")
    return manifest


def run_metrics(tmp_path: Path, manifest: Path, *extra: str) -> int:
    from rtb.metrics.__main__ import main

    return main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--phrases",
            "configs/phrases.yaml",
            "--speakers",
            "configs/speakers.yaml",
            "--thresholds",
            "configs/thresholds.yaml",
            *extra,
        ]
    )


def test_the_real_audio_manifest_is_loaded_without_being_asked_for(tmp_path: Path) -> None:
    # Se perdio una vez de verdad: una corrida con solo --manifest del modelo reescribio
    # failures.jsonl sin las filas de audio real y borro la escala de la tabla commiteada.
    manifest = write_metrics_repo(tmp_path, with_ground_truth=True)
    assert run_metrics(tmp_path, manifest) == 0
    rows = read_rows(tmp_path / "data" / "metrics" / "failures.jsonl", MetricRow)
    assert any(row.generation_id.startswith("gt__") for row in rows)


def test_writing_results_without_a_real_audio_floor_aborts(tmp_path: Path) -> None:
    manifest = write_metrics_repo(tmp_path, with_ground_truth=False)
    assert run_metrics(tmp_path, manifest) == 1
    assert not (tmp_path / "data" / "metrics" / "failures.jsonl").exists()


def test_an_existing_results_file_survives_the_abort(tmp_path: Path) -> None:
    # Lo que importa no es que aborte, sino que NO haya tocado el archivo antes de abortar.
    manifest = write_metrics_repo(tmp_path, with_ground_truth=False)
    failures = tmp_path / "data" / "metrics" / "failures.jsonl"
    failures.parent.mkdir(parents=True, exist_ok=True)
    failures.write_text("previous results\n", encoding="utf-8")
    assert run_metrics(tmp_path, manifest) == 1
    assert failures.read_text(encoding="utf-8") == "previous results\n"


def test_the_floor_can_be_waived_on_purpose(tmp_path: Path) -> None:
    manifest = write_metrics_repo(tmp_path, with_ground_truth=False)
    assert run_metrics(tmp_path, manifest, "--no-ground-truth") == 0
    assert (tmp_path / "data" / "metrics" / "failures.jsonl").exists()


def add_second_model(tmp_path: Path) -> Path:
    """A second model manifest, so the results file ends up holding two models."""
    manifest = tmp_path / "data" / "manifests" / "other.jsonl"
    row = ok_row(
        generation_id="other__slr61_f_01__vos_001__s11__ref0",
        model="other",
        phrase_id="vos_001",
    )
    manifest.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    transcripts = tmp_path / "data" / "metrics" / "transcripts.jsonl"
    extra = MetricRow(
        generation_id="other__slr61_f_01__vos_001__s11__ref0",
        metric="transcripts",
        metric_version="large-v3/int8_float16/faster-whisper==1.2.1",
        values={"text": "Vos tenés razón, pero no me lo dijiste antes.", "last_word_end_s": 1.0},
    )
    with transcripts.open("a", encoding="utf-8") as fh:
        fh.write(extra.model_dump_json() + "\n")
    return manifest


def test_rewriting_results_without_a_models_manifest_aborts(tmp_path: Path) -> None:
    # El mismo agujero que borro el ground truth, del lado de los modelos: calcular metricas de un
    # modelo recien generado sin pasar el manifest del anterior lo borraria de la tabla, y nada
    # aguas abajo se veria mal — habria una fila menos y ya.
    first = write_metrics_repo(tmp_path, with_ground_truth=True)
    second = add_second_model(tmp_path)
    assert run_metrics(tmp_path, first, "--manifest", str(second)) == 0

    failures = tmp_path / "data" / "metrics" / "failures.jsonl"
    before = failures.read_text(encoding="utf-8")
    models = {row.generation_id.split("__", 1)[0] for row in read_rows(failures, MetricRow)}
    assert models == {"m", "other", "gt"}

    # Ahora, la corrida que se olvida del primer modelo.
    assert run_metrics(tmp_path, second) == 1
    assert failures.read_text(encoding="utf-8") == before


def test_the_abort_leaves_no_file_half_rewritten(tmp_path: Path) -> None:
    # Se chequean TODOS los archivos antes de escribir NINGUNO: un failures.jsonl nuevo junto a un
    # duration_screen.jsonl viejo seria peor que no haber corrido.
    first = write_metrics_repo(tmp_path, with_ground_truth=True)
    second = add_second_model(tmp_path)
    run_metrics(tmp_path, first, "--manifest", str(second))
    metrics_dir = tmp_path / "data" / "metrics"
    before = {
        path.name: path.read_text(encoding="utf-8")
        for path in (metrics_dir / "failures.jsonl", metrics_dir / "duration_screen.jsonl")
    }
    assert run_metrics(tmp_path, second) == 1
    for name, content in before.items():
        assert (metrics_dir / name).read_text(encoding="utf-8") == content


def test_dropping_a_model_is_possible_on_purpose(tmp_path: Path) -> None:
    first = write_metrics_repo(tmp_path, with_ground_truth=True)
    second = add_second_model(tmp_path)
    run_metrics(tmp_path, first, "--manifest", str(second))
    assert run_metrics(tmp_path, second, "--drop-missing-models") == 0
    failures = tmp_path / "data" / "metrics" / "failures.jsonl"
    models = {row.generation_id.split("__", 1)[0] for row in read_rows(failures, MetricRow)}
    assert models == {"other", "gt"}


def test_a_first_run_has_nothing_to_lose(tmp_path: Path) -> None:
    manifest = write_metrics_repo(tmp_path, with_ground_truth=True)
    assert run_metrics(tmp_path, manifest) == 0


def test_adding_a_model_is_never_blocked(tmp_path: Path) -> None:
    # El guard solo mira lo que DESAPARECE. Sumar un modelo tiene que pasar sin fricción.
    first = write_metrics_repo(tmp_path, with_ground_truth=True)
    assert run_metrics(tmp_path, first) == 0
    second = add_second_model(tmp_path)
    assert run_metrics(tmp_path, first, "--manifest", str(second)) == 0
