"""WER, CER and the failure taxonomy built on top of them."""

from __future__ import annotations

from pathlib import Path

import pytest

from rtb.config import Phrase
from rtb.manifest import read_rows
from rtb.metrics import failures
from rtb.metrics.intelligibility import align, max_ngram_repeats, metric_version, score
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, MetricRow
from rtb.textnorm import expand_small_cardinals

from .conftest import make_thresholds
from .test_schema import ok_row

THRESHOLDS = make_thresholds()
ASR = "large-v3/int8_float16/faster-whisper==1.2.1"


def phrase(**overrides: object) -> Phrase:
    base: dict[str, object] = {
        "id": "vos_001",
        "text": "Vos tenés razón, pero no me lo dijiste antes.",
        "category": "voseo",
        "failure_hypothesis": "x",
    }
    base.update(overrides)
    return Phrase.model_validate(base)


def transcript(text: str, last_word_end_s: float | None = None) -> MetricRow:
    return MetricRow(
        generation_id="g",
        metric="transcripts",
        metric_version=ASR,
        values={"text": text, "last_word_end_s": last_word_end_s},
    )


def shape(**values: object) -> MetricRow:
    base: dict[str, object] = {"speech_end_s": 1.0, "silent": False}
    base.update(values)
    return MetricRow(generation_id="g", metric="audio_shape", metric_version="1", values=base)


# ── alineamiento ─────────────────────────────────────────────────────────────────────────────


def test_a_perfect_match_has_no_errors() -> None:
    result = align(["una", "dos", "tres"], ["una", "dos", "tres"])
    assert (result.errors, result.hits) == (0, 3)
    assert result.rate == 0.0


def test_extra_words_are_insertions_not_substitutions() -> None:
    # La distincion es la clasificacion: insercion es contenido que nadie pidio.
    result = align(["dale"], ["dale", "bermuda"])
    assert (result.insertions, result.substitutions, result.deletions) == (1, 0, 0)
    assert result.insertion_rate == 1.0


def test_missing_words_at_the_end_are_trailing_deletions() -> None:
    result = align(["una", "dos", "tres", "cuatro"], ["una", "dos"])
    assert result.trailing_deletions == 2


def test_a_deletion_in_the_middle_is_not_trailing() -> None:
    # Perder una palabra del medio es otro problema que cortarse al final.
    result = align(["una", "dos", "tres"], ["una", "tres"])
    assert result.deletions == 1
    assert result.trailing_deletions == 0


def test_the_rate_is_not_clipped_at_one() -> None:
    # Un modelo que emite el triple de palabras tiene que verse peor que uno que emite el doble.
    result = align(["dale"], ["dale", "uno", "dos", "tres"])
    assert result.rate == 3.0


def test_max_ngram_repeats_needs_consecutive_repetition() -> None:
    assert max_ngram_repeats(["adios", "adios", "y", "un", "unico"]) == 2
    assert max_ngram_repeats(["punto", "com", "punto", "com"]) == 2
    # Una palabra que aparece dos veces separada NO es un loop.
    assert max_ngram_repeats(["la", "casa", "de", "la", "esquina"]) == 1


# ── formas aceptadas ─────────────────────────────────────────────────────────────────────────


def test_it_scores_against_the_closest_accepted_form() -> None:
    # Un telefono admite varias lecturas validas: penalizar la que el modelo eligio seria
    # castigarlo por acertar.
    forms = ["cuatro mil quinientos sesenta y siete", "cuatro cinco seis siete"]
    result = score(forms, "cuatro cinco seis siete")
    assert result is not None
    assert result.matched_index == 1
    assert result.wer == 0.0


def test_the_matched_form_is_reported() -> None:
    result = score(["tres coma cinco", "tres punto cinco"], "tres punto cinco")
    assert result is not None and result.matched_index == 1


# ── conflicto de formato ─────────────────────────────────────────────────────────────────────


def test_a_small_number_written_as_a_digit_is_canonicalized() -> None:
    # "a las 3" contra "a las tres" es una diferencia de ESCRITURA, no de habla, y el numero es
    # incidental a lo que la frase mide.
    result = score(["tengo un meeting a las tres"], "Tengo un meeting a las 3")
    assert result is not None
    assert not result.formatting_conflict
    assert result.wer == 0.0


def test_a_formatted_number_stays_unscorable() -> None:
    # Acá el numero ES el test: adivinar como se pronuncio esconderia la falla que la categoria
    # `normalizacion` existe para encontrar.
    result = score(["el total es quince mil setecientos cincuenta"], "El total es 15.750")
    assert result is not None and result.formatting_conflict


def test_a_pronounced_symbol_makes_the_row_unscorable() -> None:
    # El "%" se pronuncia "por ciento" y la normalizacion lo borra como puntuacion: la hipotesis
    # pierde palabras que la referencia tiene, y el WER mediria eso.
    result = score(["tres coma cinco por ciento"], "3.5%")
    assert result is not None and result.formatting_conflict


def test_expand_small_cardinals_stops_at_a_hundred() -> None:
    assert expand_small_cardinals("a las 3 y 31 de 100") == "a las tres y treinta y uno de cien"
    assert expand_small_cardinals("son 750 pesos") == "son 750 pesos"


def test_metric_version_covers_the_normalizer_and_the_asr() -> None:
    # Un WER etiquetado solo con el normalizador pareceria vigente despues de re-transcribir con
    # otro modelo.
    version = metric_version(ASR)
    assert version.startswith("norm") and ASR in version


# ── taxonomia ────────────────────────────────────────────────────────────────────────────────


def classify_row(text: str, **kwargs: object) -> MetricRow:
    row = ok_row(generation_id="g", phrase_id="vos_001", duration_s=2.7)
    return failures.evaluate_row(
        row,
        phrase(),
        transcript(text, kwargs.get("last_word_end_s")),  # type: ignore[arg-type]
        shape(**{k: v for k, v in kwargs.items() if k != "last_word_end_s"}),
        THRESHOLDS,
        ASR,
    )


def test_a_clean_synthesis_is_not_a_failure() -> None:
    metric = classify_row("Vos tenés razón, pero no me lo dijiste antes.")
    assert metric.values["any_failure"] is False
    assert metric.values["wer"] == 0.0


def test_extra_words_are_a_hallucination() -> None:
    metric = classify_row("Vos tenés razón, pero no me lo dijiste antes. Adiós Bermuda.")
    assert metric.values["hallucinated"] is True
    assert metric.values["reason"] == "hallucination"


def test_stopping_early_is_a_truncation() -> None:
    metric = classify_row("Vos tenés razón, pero no")
    assert metric.values["truncated"] is True
    assert metric.values["trailing_deletions"] >= 2


def test_repetition_without_extra_material_is_not_a_loop() -> None:
    # El falso positivo que corrigio la regla: nor_005 dijo "punto com punto com", que repite y no
    # agrega. Es una sustitucion, no un decodificador en loop.
    result = score(
        ["info arroba ejemplo punto com punto ar"], "info arroba ejemplo punto com punto com"
    )
    assert result is not None
    verdict = failures.classify(result, None, THRESHOLDS.failures)
    # Hay repeticion, pero no material extra: la regla exige las dos cosas.
    assert max_ngram_repeats(result.hypothesis.split()) >= 2
    assert result.words.insertions == 0
    assert not verdict.looped


# ── la version de la taxonomia ───────────────────────────────────────────────────────────────────

# sha256 de la fuente de `classify` y `overruns`, que son las que deciden. Si este test falla, el
# codigo de clasificacion cambio: subi `TAXONOMY_VERSION` si cambio la CONDUCTA, o repiná el hash
# acá si fue un comentario. Sin el pin, el campo que declara la version se desactualiza en silencio
# y dos filas con el mismo `metric_version` salen de reglas distintas.
CLASSIFIER_SHA256 = "5097c0fd59f81a217388ee9a47bf228bb800741a86b7a6cf31c45f494c38b629"


def classifier_digest() -> str:
    import hashlib
    import inspect

    source = inspect.getsource(failures.classify) + inspect.getsource(failures.overruns)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def test_the_taxonomy_version_covers_the_code_that_decides() -> None:
    assert classifier_digest() == CLASSIFIER_SHA256, (
        "la fuente de classify/overruns cambio: subí TAXONOMY_VERSION si cambió la conducta, "
        f"o repiná CLASSIFIER_SHA256 = {classifier_digest()!r} si fue cosmético"
    )


def test_the_metric_version_names_both_the_rules_and_the_thresholds() -> None:
    metric = classify_row("Vos tenés razón, pero no me lo dijiste antes.")
    assert metric.metric_version.startswith(f"taxonomy{failures.TAXONOMY_VERSION}/thresholds")


def test_a_split_word_is_a_boundary_disagreement_and_not_a_hallucination() -> None:
    # El caso de ANALYSIS A1: el ASR partio "llamalo" en dos y el alineamiento de palabras vio
    # material extra donde no lo hay. Pegado, son dos caracteres.
    result = score(
        ["Llamalo ya, que la llave está en el llavero."],
        "Y amalo ya que la llave está en el llavero.",
    )
    assert result is not None
    verdict = failures.classify(result, None, THRESHOLDS.failures)
    assert result.words.insertions >= 1 and result.words.substitutions >= 1
    assert verdict.boundary_disagreement
    # Lo que hace el veredicto: la saca de `hallucinated` y NO la cuenta como falla.
    assert not verdict.hallucinated
    assert not verdict.any_failure
    assert verdict.reason == "boundary disagreement"


def test_inserting_a_word_without_consuming_one_stays_a_hallucination() -> None:
    # Contraste que define la regla: agregar "y" entre dos palabras intactas mueve poquisimo el
    # texto pegado, pero no sustituye nada. Sin el requisito de sustitucion, el corte de distancia
    # se traga esta fila por el solo hecho de que la frase es larga.
    result = score(
        ["El sistema toma el audio del usuario lo transcribe con un modelo."],
        "El sistema toma el audio del usuario y lo transcribe con un modelo.",
    )
    assert result is not None
    verdict = failures.classify(result, None, THRESHOLDS.failures)
    assert result.words.substitutions == 0
    assert result.glued_rate <= THRESHOLDS.failures.boundary_max_glued_rate
    assert not verdict.boundary_disagreement
    assert verdict.hallucinated


def test_added_material_is_not_forgiven_by_the_glued_text() -> None:
    # Una sustitucion mas material inventado supera el corte: la regla no absuelve por tener una
    # sustitucion a mano.
    result = score(
        ["Dejate de chamuyar y poné la guita sobre la mesa."],
        "Dejate de chamullar y poné la guita sobre la mesa oh no.",
    )
    assert result is not None
    verdict = failures.classify(result, None, THRESHOLDS.failures)
    assert result.glued_rate > THRESHOLDS.failures.boundary_max_glued_rate
    assert not verdict.boundary_disagreement
    assert verdict.hallucinated


def test_repetition_with_extra_material_is_a_loop() -> None:
    metric = classify_row(
        "Vos tenés razón, pero no me lo dijiste antes. Adiós, adiós, adiós y un unico."
    )
    assert metric.values["looped"] is True
    assert metric.values["reason"] == "loop"


def test_sound_after_the_last_word_is_trailing_audio() -> None:
    metric = classify_row(
        "Vos tenés razón, pero no me lo dijiste antes.",
        last_word_end_s=1.0,
        speech_end_s=2.4,
    )
    assert metric.values["trailing_audio"] is True
    assert metric.values["reason"] == "trailing audio"


def test_a_normal_tail_is_not_flagged() -> None:
    metric = classify_row(
        "Vos tenés razón, pero no me lo dijiste antes.",
        last_word_end_s=1.0,
        speech_end_s=1.2,
    )
    assert metric.values["trailing_audio"] is False


def test_the_tail_seconds_travel_with_the_verdict() -> None:
    # Un booleano pierde el tamaño de un defecto que se mide en segundos, y es lo que permite
    # barrer el umbral sin volver a clasificar.
    metric = classify_row(
        "Vos tenés razón, pero no me lo dijiste antes.",
        last_word_end_s=1.0,
        speech_end_s=1.2,
    )
    assert metric.values["trailing_audio_s"] == pytest.approx(0.2)


def test_a_word_timestamp_past_the_last_audible_frame_is_not_a_negative_tail() -> None:
    # Los timestamps de palabra tienen error de decenas de ms y pueden caer despues del ultimo
    # frame con sonido. Una cola negativa romperia el barrido y la anatomia.
    metric = classify_row(
        "Vos tenés razón, pero no me lo dijiste antes.",
        last_word_end_s=1.5,
        speech_end_s=1.2,
    )
    assert metric.values["trailing_audio_s"] == 0.0
    assert metric.values["trailing_audio"] is False


def test_a_missing_final_word_with_sound_there_is_a_trailing_disagreement() -> None:
    # El caso de ANALYSIS A2: falta "antes" y hay sonido justo donde deberia estar. La regla no
    # decide si el ASR no lo oyo o el modelo lo dijo mal — manda la fila a escucha.
    metric = classify_row(
        "Vos tenés razón, pero no me lo dijiste.",
        last_word_end_s=1.0,
        speech_end_s=2.4,
    )
    assert metric.values["trailing_disagreement"] is True
    # Las dos reglas que se contradicen quedan apagadas, y la fila sale del ranking.
    assert metric.values["trailing_audio"] is False
    assert metric.values["truncated"] is False
    assert metric.values["any_failure"] is False
    assert metric.values["reason"] == "trailing disagreement"


def test_a_trailing_disagreement_does_not_leave_a_hallucination_behind() -> None:
    # A un transcript al que le faltan palabras del final no se le puede contar material de mas: la
    # insercion que ve el alineador es el acomodo de lo que el ASR no escribio. El caso real son 3
    # filas de `nor_005` donde el modelo dijo la frase entera y Whisper se comio la mitad.
    #
    # La hipotesis tiene las dos cosas a la vez: una insercion —que sin esto marcaria alucinacion—
    # y palabras faltando al final, con sonido ahi.
    metric = classify_row(
        "Vos vos tenés razón, pero",
        last_word_end_s=1.0,
        speech_end_s=2.4,
    )
    assert metric.values["insertions"] >= 1
    assert metric.values["trailing_disagreement"] is True
    assert metric.values["hallucinated"] is False
    assert metric.values["any_failure"] is False
    assert metric.values["reason"] == "trailing disagreement"


def test_missing_words_with_silence_after_them_stay_truncation() -> None:
    # El contraste que define la regla: sin sonido al final no hay desacuerdo, hay un modelo que
    # se corto.
    metric = classify_row(
        "Vos tenés razón, pero no me lo.",
        last_word_end_s=1.0,
        speech_end_s=1.1,
    )
    assert metric.values["truncated"] is True
    assert metric.values["trailing_disagreement"] is False
    assert metric.values["reason"] == "truncation"


def test_a_tail_with_the_whole_phrase_transcribed_stays_trailing_audio() -> None:
    # El otro contraste: material de mas al final, sin nada faltando, sigue siendo cola.
    metric = classify_row(
        "Vos tenés razón, pero no me lo dijiste antes.",
        last_word_end_s=1.0,
        speech_end_s=2.4,
    )
    assert metric.values["trailing_disagreement"] is False
    assert metric.values["trailing_audio"] is True


def test_a_short_utterance_said_twice_is_a_loop() -> None:
    # El caso que la regla existe para agarrar: Chatterbox contestando "Dale, dale." a "Dale.".
    # Repite, agrega, y el resultado mide el doble de lo pedido. La deteccion de n-gramas llega
    # hasta 4, asi que los loops que agarra son de enunciados cortos — que es donde aparecen.
    result = score(["Dale."], "Dale, dale.")
    assert result is not None
    verdict = failures.classify(result, None, THRESHOLDS.failures)
    assert verdict.looped
    assert verdict.reason == "loop"


def test_repeating_a_word_without_running_long_is_not_a_loop() -> None:
    # El falso positivo que destapo la condicion: decir "uno uno" por "once" repite un unigrama y
    # agrega una palabra, pero la frase sale entera y del largo pedido. Un decodificador en loop se
    # pasa de largo; esto es otra cosa, y va a la cadena que sigue.
    metric = classify_row("Vos vos tenés razón, pero no me lo dijiste antes.")
    assert metric.values["ngram_repeats"] >= 2
    assert metric.values["insertions"] >= 1
    assert metric.values["length_ratio"] < THRESHOLDS.failures.loop_min_length_ratio
    assert metric.values["looped"] is False


def test_the_length_ratio_travels_with_the_verdict() -> None:
    # Igual que los segundos de cola: sin el numero en la fila, el barrido tendria que volver a
    # alinear las 2160 filas.
    metric = classify_row("Vos tenés razón, pero no me lo dijiste antes.")
    assert metric.values["length_ratio"] == pytest.approx(1.0)


def test_the_loop_sweep_reports_the_threshold_in_use() -> None:
    rows = [
        # Repite y NO se pasa de largo: 10 palabras sobre 9.
        classify_row("Vos vos tenés razón, pero no me lo dijiste antes."),
        # Repite y se pasa de largo: 13 sobre 9.
        classify_row("Vos tenés razón, pero no me lo dijiste antes antes antes antes."),
    ]
    points = failures.loop_sensitivity(rows, THRESHOLDS.failures)
    in_use = [point for point in points if point.in_use]
    assert len(in_use) == 1
    # Con el corte en 1.3 entra sólo el que dice la frase dos veces.
    assert in_use[0].flagged == 1
    assert in_use[0].total == 2


def test_the_tail_sweep_reports_the_threshold_in_use() -> None:
    rows = [
        classify_row(
            "Vos tenés razón, pero no me lo dijiste antes.",
            last_word_end_s=1.0,
            speech_end_s=end,
        )
        for end in (1.1, 1.4, 2.0, 4.0)
    ]
    points = failures.tail_sensitivity(rows, THRESHOLDS.failures)
    in_use = [point for point in points if point.in_use]
    assert len(in_use) == 1
    # Con el umbral en 0.5 s entran las colas de 1.0 y 3.0 s; con 2.0 s, solo la de 3.0.
    assert in_use[0].flagged == 2
    assert next(point for point in points if point.value == 2.0).flagged == 1
    assert {point.total for point in points} == {4}


def test_a_silent_clip_is_its_own_failure() -> None:
    metric = classify_row("Vos tenés razón, pero no me lo dijiste antes.", silent=True)
    assert metric.values["silent"] is True
    assert metric.values["reason"] == "silent"


def test_a_hard_error_needs_no_transcript() -> None:
    row = GenerationRow(
        generation_id="g",
        model="m",
        model_revision="a",
        lib_version="x==1",
        device="cuda",
        speaker_id="s",
        ref_clip_idx=0,
        phrase_id="vos_001",
        seed=1,
        error="CUDA out of memory",
    )
    metric = failures.evaluate_row(row, phrase(), None, None, THRESHOLDS, ASR)
    assert metric.values["hard_error"] is True and metric.values["any_failure"] is True


def test_an_unscorable_row_counts_as_neither(tmp_path: Path) -> None:
    row = ok_row(generation_id="g", phrase_id="nor_001", duration_s=2.0)
    numeric = phrase(
        id="nor_001",
        text="El total es $15.750 con IVA incluido.",
        category="normalizacion",
        expected_verbalization=(
            "el total es quince mil setecientos cincuenta pesos con iva incluido"
        ),
    )
    metric = failures.evaluate_row(
        row, numeric, transcript("El total es 15.750 con IVA incluido."), None, THRESHOLDS, ASR
    )
    assert metric.values["unscorable"] is True
    rates = failures.rates([metric])
    assert (rates.total, rates.scored, rates.any_failure) == (1, 0, 0)


def test_ground_truth_uses_its_own_reference_text() -> None:
    row = GenerationRow(
        generation_id="gt__s__ctrl_00",
        model=GROUND_TRUTH_MODEL,
        speaker_id="s",
        phrase_id="ctrl_00",
        reference_text="una dos tres",
        wav_path="a.wav",
        sample_rate=16000,
        duration_s=1.0,
    )
    metric = failures.evaluate_row(row, None, transcript("una dos tres"), None, THRESHOLDS, ASR)
    assert metric.values["wer"] == 0.0


# ── resultados commiteados ───────────────────────────────────────────────────────────────────


def test_the_committed_ground_truth_scores_zero(repo_root: Path) -> None:
    # Es el piso que le da sentido a todo lo demas: si el instrumento no acierta el audio real,
    # ningun numero sobre el sintetico es atribuible al modelo.
    path = repo_root / "data" / "metrics" / "failures.jsonl"
    if not path.exists():
        pytest.skip("failures not computed in this checkout")
    rows: list[MetricRow] = read_rows(path, MetricRow)
    gt = [row for row in rows if row.generation_id.startswith("gt__")]
    assert gt, "no ground-truth rows in failures.jsonl"
    for row in gt:
        assert row.values.get("wer") == 0.0, f"{row.generation_id}: WER {row.values.get('wer')}"
        assert row.values.get("any_failure") is False
