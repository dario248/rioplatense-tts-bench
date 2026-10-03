"""Speaker similarity: the cosine, the pairs that calibrate it, and the scale.

No torch here. Everything that decides *what* gets compared and *how the number is read* is
standard library, so it is pinned by tests that run in CI; only the embedding itself needs the
heavy environment, and that is the part with nothing to decide.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rtb.config import RefClip, Speaker, load_speakers
from rtb.manifest import read_rows
from rtb.metrics import speaker_sim as sim
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, MetricRow

from .test_schema import ok_row

SHARED = "la ciudad es bellisima cuales son las diferencias"


def clip(name: str, transcript: str | None = None) -> RefClip:
    return RefClip(path=Path(f"data/refs/{name}.wav"), transcript=transcript)


def speaker(index: int, *, shared: str = SHARED, controls: int = 2) -> Speaker:
    return Speaker(
        id=f"spk_{index:02d}",
        gender="F" if index % 2 else "M",
        ref_clips=[clip(f"s{index}_ref_00"), clip(f"s{index}_ref_01")],
        control_clips=[
            clip(f"s{index}_ctrl_00", shared),
            *(clip(f"s{index}_ctrl_{n:02d}", f"otro texto {n}") for n in range(1, controls)),
        ],
        calibration_clips=[clip(f"s{index}_calib_00", "no te entiendo nada")],
    )


# ── el coseno ────────────────────────────────────────────────────────────────────────────────────


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    assert sim.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_ignores_magnitude() -> None:
    # Es la razon por la que esta metrica NO necesita normalizacion de loudness: medido aparte,
    # ECAPA da coseno 1.0 sobre el mismo clip escalado -12, -6 y +6 dB.
    assert sim.cosine([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_and_opposite_vectors() -> None:
    assert sim.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert sim.cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_comparing_embeddings_of_different_size_is_an_error() -> None:
    # Silencioso de otro modo: zip() truncaria al mas corto y devolveria un numero plausible.
    with pytest.raises(sim.SpeakerSimError):
        sim.cosine([1.0, 2.0], [1.0, 2.0, 3.0])


def test_a_zero_embedding_is_an_error_not_a_zero() -> None:
    with pytest.raises(sim.SpeakerSimError):
        sim.cosine([0.0, 0.0], [1.0, 2.0])


# ── los pares de la escala ───────────────────────────────────────────────────────────────────────


def test_the_ceiling_pairs_are_controls_of_the_same_speaker() -> None:
    pairs = [pair for pair in sim.scale_pairs([speaker(1), speaker(2)]) if pair.kind == sim.CEILING]
    assert len(pairs) == 2  # un par por hablante, con dos controles cada uno
    for pair in pairs:
        assert "ctrl" in pair.left.stem and "ctrl" in pair.right.stem
        assert pair.left.stem.split("_")[0] == pair.right.stem.split("_")[0]


def test_a_third_control_clip_adds_ceiling_pairs_without_a_code_change() -> None:
    pairs = sim.scale_pairs([speaker(1, controls=3)])
    assert len([pair for pair in pairs if pair.kind == sim.CEILING]) == 3


def test_the_floor_pairs_are_the_shared_content_clip_across_speakers() -> None:
    speakers = [speaker(index) for index in (1, 2, 3, 4)]
    pairs = [pair for pair in sim.scale_pairs(speakers) if pair.kind == sim.FLOOR]
    assert len(pairs) == 6  # todos los pares de 4 hablantes
    for pair in pairs:
        assert pair.left.stem.endswith("ctrl_00") and pair.right.stem.endswith("ctrl_00")


def test_calibration_clips_never_enter_the_published_ceiling_or_floor() -> None:
    # SPEC 7.2: el embedding de un clip de 0.8 s es ruidoso, y promediarlo bajaria el techo, que
    # es la direccion que favorece a todos los modelos. Aparecen SOLO en su propio techo corto.
    for pair in sim.scale_pairs([speaker(1), speaker(2)]):
        touches_calibration = "calib" in pair.left.stem or "calib" in pair.right.stem
        assert touches_calibration == (pair.kind == sim.SHORT_CEILING)


def test_reference_clips_appear_only_in_the_diagnostics() -> None:
    for pair in sim.scale_pairs([speaker(1), speaker(2)]):
        touches_reference = "ref_" in pair.left.stem or "ref_" in pair.right.stem
        assert touches_reference == (pair.kind in {sim.REF_VS_CONTROL, sim.REF_VS_REF})


def test_the_diagnostics_cover_the_concatenation_decision() -> None:
    kinds = {pair.kind for pair in sim.scale_pairs([speaker(1), speaker(2)])}
    assert kinds == {
        sim.CEILING,
        sim.FLOOR,
        sim.SHORT_CEILING,
        sim.REF_VS_CONTROL,
        sim.REF_VS_REF,
    }


def test_the_short_ceiling_pairs_a_calibration_clip_with_a_control() -> None:
    # El clip corto queda FUERA del techo largo pero tiene su propio punto de comparacion: sin eso,
    # una sintesis de 0.8 s se lee contra un techo medido en clips de 7 s y pierde por duracion.
    pairs = [
        pair for pair in sim.scale_pairs([speaker(1), speaker(2)]) if pair.kind == sim.SHORT_CEILING
    ]
    assert len(pairs) == 2
    for pair in pairs:
        assert "calib" in pair.left.stem and pair.right.stem.endswith("ctrl_00")


def test_the_short_ceiling_never_enters_the_long_one() -> None:
    scale = sim.scale_from_rows(scale_rows([0.80, 0.84], [0.10], short_ceiling=[0.35, 0.31]))
    assert scale.ceiling == pytest.approx(0.82)
    assert scale.short_ceiling == pytest.approx(0.33)
    assert scale.end(sim.CEILING).n == 2


def test_a_floor_over_different_content_is_refused() -> None:
    # El fallo silencioso que este chequeo ataca: contenido distinto baja el coseno, un piso mas
    # bajo ensancha la escala, y todos los puntajes normalizados salen mas altos de lo que son.
    with pytest.raises(sim.SpeakerSimError, match="same text for every speaker"):
        sim.scale_pairs([speaker(1), speaker(2, shared="otro texto entero")])


def test_a_control_clip_without_transcript_cannot_anchor_the_floor() -> None:
    with pytest.raises(sim.SpeakerSimError, match="no transcript"):
        sim.scale_pairs([speaker(1), speaker(2, shared=None)])  # type: ignore[arg-type]


def test_pair_row_ids_are_stable_and_distinct() -> None:
    pairs = sim.scale_pairs([speaker(1), speaker(2)])
    ids = [pair.row_id for pair in pairs]
    assert len(ids) == len(set(ids))
    assert all(row_id.startswith("scale__") for row_id in ids)


# ── la escala ────────────────────────────────────────────────────────────────────────────────────


def scale_rows(
    ceiling: list[float], floor: list[float], **diagnostics: list[float]
) -> list[MetricRow]:
    rows = []
    for kind, values in [(sim.CEILING, ceiling), (sim.FLOOR, floor), *diagnostics.items()]:
        for index, value in enumerate(values):
            rows.append(
                MetricRow(
                    generation_id=f"scale__{kind}__{index}",
                    metric=sim.SCALE_METRIC_NAME,
                    metric_version="v",
                    values={"kind": kind, "label": str(index), "cosine": value},
                )
            )
    return rows


def test_the_scale_is_rebuilt_from_the_rows_without_audio() -> None:
    scale = sim.scale_from_rows(scale_rows([0.80, 0.84], [0.10, 0.12]))
    assert scale.ceiling == pytest.approx(0.82)
    assert scale.floor == pytest.approx(0.11)
    assert (scale.end(sim.CEILING).n, scale.end(sim.FLOOR).n) == (2, 2)
    assert scale.usable


def test_each_end_keeps_its_range_not_just_its_mean() -> None:
    # Cuatro pares es poca muestra: un techo de 0.83 que va de 0.61 a 0.95 no es un techo.
    end = sim.scale_from_rows(scale_rows([0.80, 0.84, 0.90], [0.10])).end(sim.CEILING)
    assert (end.low, end.high) == (0.80, 0.90)


def test_error_rows_do_not_drag_the_scale_down() -> None:
    rows = scale_rows([0.80, 0.84], [0.10])
    rows.append(
        MetricRow(
            generation_id="scale__ceiling__broken",
            metric=sim.SCALE_METRIC_NAME,
            metric_version="v",
            error="AudioError: no audio",
        )
    )
    assert sim.scale_from_rows(rows).end(sim.CEILING).n == 2


def test_normalizing_puts_the_floor_at_zero_and_the_ceiling_at_one() -> None:
    scale = sim.scale_from_rows(scale_rows([0.80], [0.10]))
    assert scale.normalized(0.10) == pytest.approx(0.0)
    assert scale.normalized(0.80) == pytest.approx(1.0)
    assert scale.normalized(0.45) == pytest.approx(0.5)


def test_a_score_above_the_ceiling_is_not_clipped() -> None:
    # Un modelo mas parecido al prompt que dos grabaciones reales del hablante entre si es un dato,
    # no un error de rango: suele significar que copio el canal y no solo la voz.
    scale = sim.scale_from_rows(scale_rows([0.80], [0.10]))
    assert scale.normalized(0.87) > 1.0


def test_an_unusable_scale_normalizes_to_nothing() -> None:
    # Sin techo no hay como leer un coseno, y devolver un numero igual seria inventarlo.
    scale = sim.scale_from_rows(scale_rows([], [0.10]))
    assert not scale.usable
    assert scale.normalized(0.5) is None


def test_a_short_clip_is_normalized_against_the_short_ceiling() -> None:
    # El mismo coseno vale distinto segun la duracion: 0.34 es un fracaso contra el techo largo y
    # es el techo mismo contra el corto.
    scale = sim.scale_from_rows(scale_rows([0.80], [0.10], short_ceiling=[0.34]))
    assert scale.normalized(0.34) == pytest.approx((0.34 - 0.10) / 0.70)
    assert scale.normalized(0.34, sim.SHORT_CEILING) == pytest.approx(1.0)


def test_normalizing_against_a_missing_ceiling_returns_nothing() -> None:
    scale = sim.scale_from_rows(scale_rows([0.80], [0.10]))
    assert scale.normalized(0.34, sim.SHORT_CEILING) is None


def test_the_diagnostic_passes_when_the_concatenated_reference_holds_up() -> None:
    scale = sim.scale_from_rows(
        scale_rows([0.80, 0.84], [0.10], ref_vs_control=[0.86, 0.89])
    )
    assert scale.diagnostic_verdict().startswith("ok")


def test_the_diagnostic_fails_loudly_when_the_clips_damage_identity() -> None:
    scale = sim.scale_from_rows(
        scale_rows([0.80, 0.84], [0.10], ref_vs_control=[0.62, 0.89])
    )
    assert "CHECK THE CLIPS" in scale.diagnostic_verdict()


# ── que filas se miden ───────────────────────────────────────────────────────────────────────────


def test_warmups_errors_and_ground_truth_are_not_measured() -> None:
    rows = [
        ok_row(generation_id="good"),
        ok_row(generation_id="warm", warmup=True),
        ok_row(generation_id="broken", error="CUDA out of memory", wav_path=None),
        GenerationRow(
            generation_id="gt__spk__ctrl_00",
            model=GROUND_TRUTH_MODEL,
            speaker_id="spk",
            phrase_id="ctrl_00",
            reference_text="una dos",
            wav_path="a.wav",
            sample_rate=16000,
            duration_s=1.0,
        ),
    ]
    assert [row.generation_id for row in sim.measurable(rows)] == ["good"]


def test_rows_already_measured_with_this_version_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "speaker_sim.jsonl"
    row = MetricRow(
        generation_id="g", metric=sim.METRIC_NAME, metric_version="v1", values={"cosine": 0.7}
    )
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    assert sim.already_done(path, "v1") == {"g"}
    # Otro checkpoint es otra metrica: se vuelve a medir.
    assert sim.already_done(path, "v2") == set()


def test_the_row_carries_the_keys_the_report_slices_by() -> None:
    row = sim.sim_row(
        ok_row(), Path("data/refs/f01_ref0.wav"), 0.7123456, "v", category="voseo"
    )
    assert row.values["cosine"] == 0.71235
    assert row.values["speaker_id"] == "slr61_f_01"
    assert row.values["ref_clip_idx"] == 0
    assert row.values["category"] == "voseo"
    assert row.values["ref_clip"].endswith("f01_ref0.wav")


def test_the_unseen_clip_is_the_other_reference_of_the_same_speaker() -> None:
    assert sim.unseen_index(speaker(1), 0) == 1
    assert sim.unseen_index(speaker(1), 1) == 0


def test_without_a_second_reference_there_is_no_unseen_clip() -> None:
    one_clip = speaker(1).model_copy(update={"ref_clips": [clip("s1_ref_00")]})
    with pytest.raises(sim.SpeakerSimError, match="only defined for two"):
        sim.unseen_index(one_clip, 0)


def test_the_unseen_row_records_both_clips_and_its_own_metric() -> None:
    # La fila tiene que decir que escucho el modelo Y contra que se midio: sin las dos, el coseno
    # del clip no visto es indistinguible del de la diagonal.
    row = sim.sim_row(
        ok_row(),
        Path("data/refs/f01_ref1.wav"),
        0.5,
        "v",
        metric=sim.UNSEEN_METRIC_NAME,
        target_clip_idx=1,
    )
    assert row.metric == sim.UNSEEN_METRIC_NAME
    assert row.values["ref_clip_idx"] == 0
    assert row.values["target_clip_idx"] == 1


def test_the_diagonal_row_carries_no_target_index() -> None:
    row = sim.sim_row(ok_row(), Path("data/refs/f01_ref0.wav"), 0.5, "v")
    assert row.metric == sim.METRIC_NAME
    assert "target_clip_idx" not in row.values


# ── resumenes ────────────────────────────────────────────────────────────────────────────────────


def sim_rows(*values: float, **fields: object) -> list[MetricRow]:
    return [
        MetricRow(
            generation_id=f"g{index}",
            metric=sim.METRIC_NAME,
            metric_version="v",
            values={"cosine": value, **fields},
        )
        for index, value in enumerate(values)
    ]


def test_the_summary_reports_the_low_tail_not_the_high_one() -> None:
    # En similitud la cola que importa es la baja: es la que se escucha como "esa no es la voz".
    rows = sim_rows(0.40, 0.70, 0.72, 0.74, 0.76)
    summary = sim.summarize(rows, sim.scale_from_rows(scale_rows([0.80], [0.10])))
    assert summary.n == 5
    assert summary.median == pytest.approx(0.72)
    assert summary.p05 == pytest.approx(0.40)
    assert summary.normalized_median == pytest.approx((0.72 - 0.10) / 0.70)


def test_summaries_group_by_a_stored_key() -> None:
    rows = [
        *sim_rows(0.70, 0.72, category="voseo"),
        *sim_rows(0.60, category="enunciados_cortos"),
    ]
    scale = sim.scale_from_rows(scale_rows([0.8], [0.1]))
    labels = [summary.label for summary in sim.summarize_by(rows, scale, "category")]
    assert labels == ["category=enunciados_cortos", "category=voseo"]


def test_an_empty_slice_summarizes_to_nothing_rather_than_zero() -> None:
    summary = sim.summarize([], sim.scale_from_rows(scale_rows([0.8], [0.1])), label="none")
    assert (summary.n, summary.median, summary.normalized_median) == (0, None, None)


# ── el factor de confusion: duracion ─────────────────────────────────────────────────────────────


def shape_rows(**by_id: float) -> dict[str, MetricRow]:
    return {
        generation_id: MetricRow(
            generation_id=generation_id,
            metric="audio_shape",
            metric_version="1",
            values={"speech_s": speech_s},
        )
        for generation_id, speech_s in by_id.items()
    }


def test_similarity_is_sliced_by_how_much_speech_the_clip_has() -> None:
    # El corte que hay que leer ANTES que las categorias: sobre la corrida de XTTS-v2 la
    # correlacion entre segundos de habla y coseno es 0.53, asi que una categoria de frases cortas
    # parece un fracaso de clonado sin serlo.
    rows = sim_rows(0.30, 0.70, 0.75)
    shapes = shape_rows(g0=0.8, g1=2.5, g2=2.7)
    scale = sim.scale_from_rows(scale_rows([0.80], [0.10], short_ceiling=[0.34]))
    summaries = sim.summarize_by_duration(rows, shapes, scale)
    assert [(summary.label, summary.n) for summary in summaries] == [
        ("speech 0-1 s", 1),
        ("speech 2-3 s", 2),
    ]


def test_the_sub_second_bin_is_read_against_the_short_ceiling() -> None:
    rows = sim_rows(0.34, 0.72)
    shapes = shape_rows(g0=0.8, g1=2.5)
    scale = sim.scale_from_rows(scale_rows([0.80], [0.10], short_ceiling=[0.34]))
    short, long = sim.summarize_by_duration(rows, shapes, scale)
    assert short.ceiling_kind == sim.SHORT_CEILING
    assert short.normalized_median == pytest.approx(1.0)
    assert long.ceiling_kind == sim.CEILING


def test_a_row_without_a_measured_shape_is_left_out_of_the_duration_slice() -> None:
    # Se omite, no se asume una duracion: inventarla movería el bin y con el la lectura.
    rows = sim_rows(0.30, 0.70)
    scale = sim.scale_from_rows(scale_rows([0.8], [0.1]))
    summaries = sim.summarize_by_duration(rows, shape_rows(g1=2.5), scale)
    assert [(summary.label, summary.n) for summary in summaries] == [("speech 2-3 s", 1)]


def test_long_clips_fall_in_the_open_top_bin() -> None:
    rows = sim_rows(0.75)
    scale = sim.scale_from_rows(scale_rows([0.8], [0.1]))
    summaries = sim.summarize_by_duration(rows, shape_rows(g0=10.7), scale)
    assert summaries[0].label == "speech 6+ s"


# ── los configs y los resultados reales ──────────────────────────────────────────────────────────


def test_the_committed_speakers_produce_the_pairs_the_spec_describes(repo_root: Path) -> None:
    speakers = load_speakers(repo_root / "configs" / "speakers.yaml")
    pairs = sim.scale_pairs(speakers)
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.kind] = counts.get(pair.kind, 0) + 1
    assert counts == {
        sim.CEILING: len(speakers),  # 2 controles por hablante -> 1 par
        sim.FLOOR: len(speakers) * (len(speakers) - 1) // 2,
        sim.SHORT_CEILING: len(speakers),
        sim.REF_VS_CONTROL: len(speakers),
        sim.REF_VS_REF: len(speakers),
    }


def test_the_calibration_clips_are_short_enough_for_the_short_ceiling(repo_root: Path) -> None:
    # `SHORT_CLIP_MAX_S` decide que bin de duracion usa el techo corto. Si alguien agregara un
    # clip de calibracion de 3 s, ese techo dejaria de ser corto sin que nada lo dijera.
    speakers = load_speakers(repo_root / "configs" / "speakers.yaml")
    for spk in speakers:
        for calibration in spk.calibration_clips:
            assert calibration.duration_s is not None
            assert calibration.duration_s <= sim.SHORT_CLIP_MAX_S, (
                f"{calibration.path} is {calibration.duration_s}s, above SHORT_CLIP_MAX_S"
            )


def test_the_committed_scale_is_readable(repo_root: Path) -> None:
    path = repo_root / "data" / "metrics" / "speaker_scale.jsonl"
    if not path.exists():
        pytest.skip("speaker similarity not computed in this checkout")
    scale = sim.scale_from_rows(read_rows(path, MetricRow))
    assert scale.usable, "a cosine with no ceiling above the floor cannot be read"
    assert scale.floor is not None and scale.floor < 0.5, (
        f"floor {scale.floor}: two different speakers reading the same text should not look alike"
    )


def test_the_committed_diagnostics_clear_the_concatenated_clips(repo_root: Path) -> None:
    # El punto de todo el pase: si los clips de referencia concatenados con 250 ms de silencio
    # bajaran el embedding, la columna de similitud mediria nuestra construccion y no al modelo.
    path = repo_root / "data" / "metrics" / "speaker_scale.jsonl"
    if not path.exists():
        pytest.skip("speaker similarity not computed in this checkout")
    scale = sim.scale_from_rows(read_rows(path, MetricRow))
    verdict = scale.diagnostic_verdict()
    assert verdict.startswith("ok"), verdict


def test_the_committed_rows_all_carry_a_finite_cosine(repo_root: Path) -> None:
    path = repo_root / "data" / "metrics" / "speaker_sim.jsonl"
    if not path.exists():
        pytest.skip("speaker similarity not computed in this checkout")
    rows = read_rows(path, MetricRow)
    assert rows
    for row in rows:
        if row.error:
            continue
        value = row.values["cosine"]
        assert isinstance(value, float) and math.isfinite(value)
        assert -1.0 <= value <= 1.0


# ── las cifras de la nota se miden, no se citan ──────────────────────────────────────────────────


def test_the_duration_correlation_is_computed_from_the_rows() -> None:
    # Estuvo hardcodeada con el valor de XTTS y se imprimia igual debajo de la tabla de Qwen: una
    # salvedad correcta convertida en afirmacion falsa por copiar un numero.
    rows = sim_rows(0.30, 0.50, 0.70, 0.90)
    shapes = shape_rows(g0=0.5, g1=1.5, g2=2.5, g3=3.5)
    assert sim.duration_correlation(rows, shapes) == pytest.approx(1.0)


def test_the_correlation_is_none_without_enough_rows() -> None:
    assert sim.duration_correlation(sim_rows(0.5), shape_rows(g0=1.0)) is None


def test_the_correlation_ignores_rows_with_no_measured_shape() -> None:
    rows = sim_rows(0.30, 0.50, 0.70, 0.99)
    correlation = sim.duration_correlation(rows, shape_rows(g0=0.5, g1=1.5, g2=2.5))
    assert correlation == pytest.approx(1.0)


def test_the_band_spread_shows_categories_collapsing_at_equal_duration() -> None:
    rows = [
        *sim_rows(*([0.70] * 8), category="voseo"),
        *sim_rows(*([0.75] * 8), category="rotica"),
    ]
    shapes = {row.generation_id: MetricRow(
        generation_id=row.generation_id, metric="audio_shape", metric_version="1",
        values={"speech_s": 2.5},
    ) for row in rows}
    band = sim.category_spread_in_band(rows, shapes)
    assert band is not None
    assert band.categories == 2
    assert band.spread == pytest.approx(0.05)


def test_a_category_with_too_few_clips_in_the_band_is_dropped() -> None:
    # Una mediana sobre tres clips no es una posicion, y ensancharia la dispersion con ruido.
    rows = [
        *sim_rows(*([0.70] * 8), category="voseo"),
        *sim_rows(*([0.20] * 3), category="rara"),
    ]
    shapes = {row.generation_id: MetricRow(
        generation_id=row.generation_id, metric="audio_shape", metric_version="1",
        values={"speech_s": 2.5},
    ) for row in rows}
    assert sim.category_spread_in_band(rows, shapes) is None
