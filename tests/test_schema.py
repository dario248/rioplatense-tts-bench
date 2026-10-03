"""Schema invariants. Every one of these encodes a way the dataset could go silently wrong."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rtb.schema import (
    GROUND_TRUTH_MODEL,
    GenerationRow,
    MetricRow,
    count_words,
    make_generation_id,
    make_warmup_id,
)


def ok_row(**overrides: object) -> GenerationRow:
    base: dict[str, object] = {
        "generation_id": "dummy__slr61_f_01__yei_003__s1234__ref0",
        "model": "dummy",
        "model_revision": "a1b2c3d",
        "lib_version": "rtb-dummy==0.0.1",
        "device": "cpu",
        "speaker_id": "slr61_f_01",
        "ref_clip_idx": 0,
        "ref_seconds_used": 3.0,
        "phrase_id": "yei_003",
        "seed": 1234,
        "wav_path": "audio/dummy/slr61_f_01/yei_003_s1234_ref0.wav",
        "sample_rate": 24000,
        "duration_s": 3.41,
    }
    base.update(overrides)
    return GenerationRow(**base)  # type: ignore[arg-type]


def test_generation_id_is_derived_from_the_grid() -> None:
    assert (
        make_generation_id("qwen3_tts", "slr61_f_01", "yei_003", 1234, 0)
        == "qwen3_tts__slr61_f_01__yei_003__s1234__ref0"
    )
    assert make_warmup_id("qwen3_tts", 2) == "qwen3_tts__warmup2"


@pytest.mark.parametrize("token", ["has__separator", "", "-leading", "with space"])
def test_generation_id_rejects_ambiguous_tokens(token: str) -> None:
    # Un token con "__" adentro hace que el id deje de ser parseable en el analisis.
    with pytest.raises(ValueError):
        make_generation_id(token, "spk", "phr", 1, 0)


def test_count_words_counts_written_tokens() -> None:
    # n_words es el largo del texto ESCRITO: un token con digitos cuenta como uno, y la
    # puntuacion suelta no cuenta como ninguno. Lo que se dice en voz alta puede ser muy
    # distinto, y para eso existe Phrase.n_spoken_words.
    assert count_words("El total es $15.750 con IVA incluido.") == 7
    assert count_words("Ibas a 120 km/h en una zona de 60.") == 9
    assert count_words("Bueno... a ver... ¿cómo te explico?") == 6
    assert count_words("¿Cómo?") == 1


def test_error_row_must_not_claim_a_wav() -> None:
    with pytest.raises(ValidationError):
        ok_row(error="boom")


def test_error_row_is_valid_without_audio() -> None:
    row = GenerationRow(
        generation_id="dummy__slr61_f_01__yei_003__s1234__ref0",
        model="dummy",
        model_revision="a1b2c3d",
        lib_version="rtb-dummy==0.0.1",
        device="cpu",
        speaker_id="slr61_f_01",
        ref_clip_idx=0,
        phrase_id="yei_003",
        seed=1234,
        error="RuntimeError: CUDA out of memory",
    )
    assert not row.ok
    assert row.wav_path is None


@pytest.mark.parametrize("field", ["wav_path", "sample_rate", "duration_s"])
def test_successful_row_requires_audio_metadata(field: str) -> None:
    with pytest.raises(ValidationError):
        ok_row(**{field: None})


@pytest.mark.parametrize("field", ["model_revision", "lib_version", "device"])
def test_synthesis_row_requires_provenance(field: str) -> None:
    # Sin el revision de HuggingFace el benchmark no es reproducible: los pesos cambian sin aviso.
    with pytest.raises(ValidationError):
        ok_row(**{field: None})


def test_ground_truth_row_may_omit_provenance() -> None:
    row = GenerationRow(
        generation_id="__ground_truth____slr61_f_01__ctrl0__s0__ref0",
        model=GROUND_TRUTH_MODEL,
        speaker_id="slr61_f_01",
        ref_clip_idx=0,
        phrase_id="ctrl0",
        reference_text="La llave quedó en la calle.",
        wav_path="data/refs/slr61/slr61_f_01/ctrl_00.wav",
        sample_rate=16000,
        duration_s=4.9,
    )
    assert row.is_ground_truth
    assert row.seed is None


def test_unknown_field_is_rejected() -> None:
    # extra="forbid" es lo que hace que CI detecte un JSONL viejo o un campo mal escrito.
    with pytest.raises(ValidationError):
        ok_row(latency_ms=12)


def test_created_at_serializes_as_utc_z() -> None:
    row = ok_row(created_at=datetime(2026, 8, 18, 14, 22, 1, tzinfo=timezone.utc))
    assert json.loads(row.model_dump_json())["created_at"] == "2026-08-18T14:22:01Z"


def test_metric_row_needs_values_or_error() -> None:
    with pytest.raises(ValidationError):
        MetricRow(generation_id="x", metric="wer", metric_version="1")
    assert MetricRow(
        generation_id="x", metric="wer", metric_version="1", values={"wer": 0.06}
    ).values["wer"] == 0.06
