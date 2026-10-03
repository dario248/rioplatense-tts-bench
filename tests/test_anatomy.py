"""The four segments of a clip: the tail comes out of the speech, and the parts add up."""

from __future__ import annotations

import pytest

from rtb.metrics import anatomy
from rtb.schema import MetricRow

SHAPE = {
    "total_s": 4.0,
    "speech_start_s": 0.2,
    "speech_end_s": 3.4,
    "speech_s": 3.2,
    "padding_s": 0.8,
    "silent": False,
}


def shape_row(generation_id: str, **values: object) -> MetricRow:
    return MetricRow(
        generation_id=generation_id,
        metric="audio_shape",
        metric_version="1",
        values={**SHAPE, **values},
    )


def transcript_row(generation_id: str, last_word_end_s: float | None) -> MetricRow:
    return MetricRow(
        generation_id=generation_id,
        metric="transcripts",
        metric_version="large-v3",
        values={"text": "una dos tres", "last_word_end_s": last_word_end_s},
    )


# ── el corte ─────────────────────────────────────────────────────────────────────────────────


def test_the_four_segments_add_up_to_the_total() -> None:
    # Si no suman, alguna cantidad se esta contando dos veces o ninguna, que es exactamente el
    # defecto que este modulo corrige.
    result = anatomy.measure(SHAPE, last_word_end_s=2.0)
    assert result is not None
    parts = (
        result.lead_silence_s,
        result.speech_s,
        result.trailing_audio_s,
        result.tail_silence_s,
    )
    assert sum(parts) == pytest.approx(result.total_s)


def test_the_tail_comes_out_of_the_speech() -> None:
    result = anatomy.measure(SHAPE, last_word_end_s=2.0)
    assert result is not None
    # `audio_shape` cuenta 3.2 s de habla porque la cola esta arriba del umbral de silencio.
    assert SHAPE["speech_s"] == pytest.approx(3.2)
    assert result.speech_s == pytest.approx(1.8)
    assert result.trailing_audio_s == pytest.approx(1.4)


def test_a_word_timestamp_past_the_last_audible_frame_gives_no_negative_tail() -> None:
    # Los timestamps de palabra tienen error de decenas de ms y a veces caen despues del sonido.
    result = anatomy.measure(SHAPE, last_word_end_s=9.0)
    assert result is not None
    assert result.trailing_audio_s == 0.0
    assert result.speech_s == pytest.approx(3.2)


def test_a_word_timestamp_before_the_speech_gives_no_negative_speech() -> None:
    result = anatomy.measure(SHAPE, last_word_end_s=0.0)
    assert result is not None
    assert result.speech_s == 0.0
    assert result.trailing_audio_s == pytest.approx(3.2)


def test_a_clip_with_no_transcript_has_no_anatomy() -> None:
    # Sin transcript no se puede separar la cola del habla, y una anatomia inventada seria peor
    # que ninguna: hay que poder contar las filas que no se pudieron medir.
    assert anatomy.measure(SHAPE, last_word_end_s=None) is None


def test_a_shape_that_failed_to_measure_has_no_anatomy() -> None:
    assert anatomy.measure({"silent": True}, last_word_end_s=1.0) is None


def test_the_padding_figure_no_longer_hides_the_tail() -> None:
    result = anatomy.measure(SHAPE, last_word_end_s=2.0)
    assert result is not None
    # `padding_s` de audio_shape son 0.8 s de silencio; el no-habla real es mas del doble.
    assert result.silence_s == pytest.approx(0.8)
    assert result.non_speech_s == pytest.approx(2.2)


# ── el join ──────────────────────────────────────────────────────────────────────────────────


def test_rows_are_joined_by_generation_id() -> None:
    shapes = {"a": shape_row("a"), "b": shape_row("b")}
    transcripts = {"a": transcript_row("a", 2.0), "b": transcript_row("b", 3.0)}
    result = anatomy.from_rows(shapes, transcripts)
    assert set(result) == {"a", "b"}
    assert result["a"].trailing_audio_s == pytest.approx(1.4)


def test_a_row_missing_from_either_pass_is_left_out() -> None:
    shapes = {"a": shape_row("a"), "b": shape_row("b"), "c": shape_row("c")}
    shapes["c"] = MetricRow(
        generation_id="c", metric="audio_shape", metric_version="1", error="broken wav"
    )
    transcripts = {"a": transcript_row("a", 2.0)}
    assert set(anatomy.from_rows(shapes, transcripts)) == {"a"}


# ── el resumen ───────────────────────────────────────────────────────────────────────────────


def test_the_summary_reports_the_tail_as_seconds_and_as_a_share() -> None:
    items = [
        anatomy.Anatomy(
            total_s=4.0, lead_silence_s=0.2, speech_s=2.0, trailing_audio_s=tail, tail_silence_s=0.3
        )
        for tail in (0.1, 0.2, 0.9, 2.0)
    ]
    summary = anatomy.summarize(items, threshold_s=0.5, label="m")
    assert summary.n == 4
    assert summary.trailing_p50 == pytest.approx(0.55)
    assert summary.trailing_p95 == pytest.approx(2.0)
    # La mediana esconde este defecto: la mitad de las filas esta por encima del umbral.
    assert summary.over_threshold == 2
    assert summary.percent_over == pytest.approx(50.0)


def test_an_empty_slice_summarizes_to_nothing_rather_than_zero() -> None:
    # Cero segundos de cola y "no habia filas" son cosas distintas.
    summary = anatomy.summarize([], threshold_s=0.5)
    assert summary.n == 0
    assert summary.trailing_p50 is None
    assert summary.percent_over == 0.0
