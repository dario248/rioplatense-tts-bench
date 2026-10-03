"""XTTS adapter logic, exercised against a fake model.

There is no torch and no GPU here, and that is the point: everything in the adapter that could be
wrong about the *protocol* — how many seconds of reference get consumed, what a timing measures,
which kwargs reach which call — is decided in plain Python and can be checked on a laptop. What
cannot be checked here is whether the real checkpoint produces good audio; that needs the GPU box.
"""

from __future__ import annotations

import array
import math
import wave
from pathlib import Path
from typing import Any

import pytest

from rtb.config import ModelConfig
from rtb.generate.xtts import XttsAdapter, as_float_list, read_duration, write_wav

from .conftest import write_wav as write_test_wav

SAMPLE_RATE = 24000


class FakeXtts:
    """Stands in for `Xtts`, recording how it was called."""

    def __init__(self, seconds: float = 1.0, chunks: int = 4) -> None:
        self.seconds = seconds
        self.chunks = chunks
        self.conditioning_calls: list[dict[str, Any]] = []
        self.inference_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.raise_on_inference: BaseException | None = None

    def get_conditioning_latents(self, **kwargs: Any) -> tuple[str, str]:
        self.conditioning_calls.append(kwargs)
        return ("latent", "embedding")

    def _wave(self, n: int) -> list[float]:
        return [0.5 * math.sin(i / 20.0) for i in range(n)]

    def inference(self, **kwargs: Any) -> dict[str, Any]:
        self.inference_calls.append(kwargs)
        if self.raise_on_inference:
            raise self.raise_on_inference
        return {"wav": self._wave(int(self.seconds * SAMPLE_RATE))}

    def inference_stream(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        if self.raise_on_inference:
            raise self.raise_on_inference
        total = int(self.seconds * SAMPLE_RATE)
        per_chunk = total // self.chunks
        for _ in range(self.chunks):
            yield self._wave(per_chunk)


class FakeAdapter(XttsAdapter):
    """The real adapter with only the heavy load replaced."""

    def __init__(self, config: ModelConfig, model: FakeXtts) -> None:
        super().__init__(config)
        self._fake = model

    def _load_model(self) -> tuple[Any, Any]:
        self._lib_version = "coqui-tts==9.9.9"
        self._torch_version = "2.5.1"
        self._device = "cpu"
        return self._fake, _FakeCheckpointConfig()


class _FakeAudio:
    output_sample_rate = SAMPLE_RATE


class _FakeCheckpointConfig:
    temperature = 0.85
    length_penalty = 1.0
    repetition_penalty = 2.0
    top_k = 50
    top_p = 0.85
    audio = _FakeAudio()


def model_config_with(*, max_ref_seconds: float = 6.0, **params: Any) -> ModelConfig:
    base: dict[str, Any] = {
        "language": "es",
        "temperature": 0.85,
        "repetition_penalty": 2.0,
    }
    base.update(params)
    return ModelConfig(
        name="xtts_v2",
        adapter="rtb.generate.xtts:XttsAdapter",
        hf_repo="coqui/XTTS-v2",
        max_ref_seconds=max_ref_seconds,
        supports_streaming=True,
        default_sample_rate=SAMPLE_RATE,
        device="cpu",
        manifest_path=Path("data/manifests/xtts_v2.jsonl"),
        params=base,
    )


def model_config(**params: Any) -> ModelConfig:
    return model_config_with(**params)


@pytest.fixture
def ref_wav(tmp_path: Path) -> Path:
    return write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=22050)


def build(model: FakeXtts, **params: Any) -> FakeAdapter:
    adapter = FakeAdapter(model_config(**params), model)
    adapter.load()
    return adapter


# ── presupuesto de referencia ────────────────────────────────────────────────────────────────


def test_one_number_drives_every_conditioning_knob(ref_wav: Path, tmp_path: Path) -> None:
    # Si gpt_cond_len y max_ref_length pudieran divergir de max_ref_seconds, el modelo se
    # condicionaria con mas audio del que el manifest declara y la similitud dejaria de ser
    # comparable entre modelos.
    model = FakeXtts()
    adapter = build(model)
    adapter.synthesize("Llovió toda la noche.", ref_wav, 1234, tmp_path / "out.wav")

    call = model.conditioning_calls[0]
    assert call["gpt_cond_len"] == 6
    assert call["gpt_cond_chunk_len"] == 6
    assert call["max_ref_length"] == 6
    assert call["audio_path"] == [str(ref_wav)]


def test_conditioning_lengths_are_whole_seconds(ref_wav: Path, tmp_path: Path) -> None:
    # XTTS los usa como multiplicadores del sample rate DENTRO de un indice de slice
    # (`audio[:, : 22050 * length]`), asi que un float revienta con TypeError adentro de coqui.
    # Comparar por igualdad no alcanza para detectarlo: 6 == 6.0 es True.
    model = FakeXtts()
    adapter = build(model)
    adapter.synthesize("Dale.", ref_wav, 1234, tmp_path / "out.wav")

    call = model.conditioning_calls[0]
    for field in ("gpt_cond_len", "gpt_cond_chunk_len", "max_ref_length"):
        assert isinstance(call[field], int), f"{field} llego como {type(call[field]).__name__}"


def test_a_fractional_budget_is_floored_and_reported_as_floored(
    ref_wav: Path, tmp_path: Path
) -> None:
    # Truncar en silencio haria mentir al manifest: ref_seconds_used diria 6.5 mientras el modelo
    # consumio 6. El piso se aplica una sola vez y de ahi sale tambien lo que se reporta.
    model = FakeXtts()
    adapter = FakeAdapter(model_config_with(max_ref_seconds=6.5), model)
    adapter.load()
    adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")

    assert model.conditioning_calls[0]["max_ref_length"] == 6
    assert adapter.reference_seconds(ref_wav) == 6.0


def test_a_budget_under_one_second_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1 second"):
        FakeAdapter(model_config_with(max_ref_seconds=0.5), FakeXtts())


def test_reference_seconds_are_capped_by_the_budget(ref_wav: Path) -> None:
    adapter = build(FakeXtts())
    assert adapter.reference_seconds(ref_wav) == 6.0


def test_reference_seconds_report_a_short_clip_honestly(tmp_path: Path) -> None:
    # Un clip mas corto que el presupuesto entrega menos: la fila tiene que decir 4.0, no 6.0.
    short = write_test_wav(tmp_path / "short.wav", seconds=4.0, sample_rate=22050)
    adapter = build(FakeXtts())
    assert adapter.reference_seconds(short) == pytest.approx(4.0)


def test_reference_clip_is_not_normalized_or_trimmed(ref_wav: Path, tmp_path: Path) -> None:
    # Normalizar loudness es cosa de la etapa de metricas, sobre una copia (SPEC 7).
    model = FakeXtts()
    adapter = build(model)
    adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")
    call = model.conditioning_calls[0]
    assert call["sound_norm_refs"] is False
    assert call["librosa_trim_db"] is None


def test_conditioning_is_computed_once_per_reference(ref_wav: Path, tmp_path: Path) -> None:
    # Cachear es lo que hace un deployment, y ademas mantiene los timings comparables: cobrarle
    # el conditioning a la primera de las 135 sintesis de un hablante la volveria un outlier.
    model = FakeXtts()
    adapter = build(model)
    for index in range(3):
        adapter.synthesize("Dale.", ref_wav, index, tmp_path / f"out{index}.wav")
    assert len(model.conditioning_calls) == 1
    assert len(model.inference_calls) == 3


def test_a_second_reference_is_conditioned_separately(ref_wav: Path, tmp_path: Path) -> None:
    other = write_test_wav(tmp_path / "ref1.wav", seconds=7.0, sample_rate=22050)
    model = FakeXtts()
    adapter = build(model)
    adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "a.wav")
    adapter.synthesize("Dale.", other, 1, tmp_path / "b.wav")
    assert len(model.conditioning_calls) == 2


# ── audio y timings ──────────────────────────────────────────────────────────────────────────


def test_synthesis_writes_a_readable_wav(ref_wav: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.wav"
    adapter = build(FakeXtts(seconds=2.0))
    result = adapter.synthesize("Llovió toda la noche.", ref_wav, 1234, out)

    assert result.ok
    assert result.sample_rate == SAMPLE_RATE
    assert result.duration_s == pytest.approx(2.0, abs=0.01)
    with wave.open(str(out), "rb") as fh:
        assert fh.getframerate() == SAMPLE_RATE
        assert fh.getnchannels() == 1
        assert fh.getnframes() / fh.getframerate() == pytest.approx(result.duration_s, abs=0.01)


def test_whole_mode_reports_ttfb_equal_to_total(ref_wav: Path, tmp_path: Path) -> None:
    # Sin streaming el primer byte existe cuando termina todo: ttfb == total es el dato correcto,
    # no un relleno.
    adapter = build(FakeXtts(), streaming=False)
    result = adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")
    assert result.streaming is False
    assert result.ttfb_ms == result.total_ms


def test_streaming_mode_reports_a_first_chunk_ahead_of_the_total(
    ref_wav: Path, tmp_path: Path
) -> None:
    adapter = build(FakeXtts(seconds=2.0, chunks=5), streaming=True)
    result = adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")
    assert result.streaming is True
    assert result.ttfb_ms is not None and result.total_ms is not None
    assert result.ttfb_ms <= result.total_ms


def test_streaming_never_passes_num_beams(ref_wav: Path, tmp_path: Path) -> None:
    # inference_stream no acepta num_beams: pasarlo revienta la corrida entera del modelo.
    model = FakeXtts()
    adapter = build(model, streaming=True)
    adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")
    assert "num_beams" not in model.stream_calls[0]
    assert model.stream_calls[0]["stream_chunk_size"] == 20


def test_sampling_params_reach_the_model(ref_wav: Path, tmp_path: Path) -> None:
    model = FakeXtts()
    adapter = build(model, temperature=0.7, top_k=30)
    adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")
    call = model.inference_calls[0]
    assert (call["temperature"], call["top_k"]) == (0.7, 30)
    assert call["language"] == "es"
    assert call["enable_text_splitting"] is False
    assert call["num_beams"] == 1


def test_a_misspelled_param_is_rejected() -> None:
    # Un parametro con el nombre mal escrito se ignoraria en silencio y la corrida mediria otra
    # configuracion que la declarada.
    with pytest.raises(ValueError, match="unknown params"):
        FakeAdapter(model_config(temperatura=0.9), FakeXtts())


# ── fallas ───────────────────────────────────────────────────────────────────────────────────


def test_out_of_memory_becomes_a_row_not_a_crash(ref_wav: Path, tmp_path: Path) -> None:
    # Un OOM a mitad de una corrida de 540 sintesis no puede abortar el resto.
    model = FakeXtts()
    model.raise_on_inference = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    adapter = build(model)
    result = adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")
    assert not result.ok
    assert "out of memory" in (result.error or "")
    assert result.ref_seconds_used == 6.0
    assert not (tmp_path / "out.wav").exists()


def test_other_errors_propagate_to_the_runner(ref_wav: Path, tmp_path: Path) -> None:
    model = FakeXtts()
    model.raise_on_inference = ValueError("tokenizer exploded")
    adapter = build(model)
    with pytest.raises(ValueError, match="tokenizer exploded"):
        adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")


def test_empty_audio_is_a_failure(ref_wav: Path, tmp_path: Path) -> None:
    adapter = build(FakeXtts(seconds=0.0))
    result = adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")
    assert not result.ok
    assert "no audio" in (result.error or "")


def test_synthesize_before_load_is_an_error(ref_wav: Path, tmp_path: Path) -> None:
    adapter = FakeAdapter(model_config(), FakeXtts())
    with pytest.raises(RuntimeError, match="load\\(\\) was not called"):
        adapter.synthesize("Dale.", ref_wav, 1, tmp_path / "out.wav")


def test_a_reference_without_audio_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    with wave.open(str(empty), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(22050)
    adapter = build(FakeXtts())
    with pytest.raises(ValueError, match="no audio"):
        adapter.synthesize("Dale.", empty, 1, tmp_path / "out.wav")


# ── procedencia ──────────────────────────────────────────────────────────────────────────────


def test_describe_carries_what_actually_ran(ref_wav: Path) -> None:
    adapter = build(FakeXtts())
    described = adapter.describe()
    assert described["lib_version"] == "coqui-tts==9.9.9"
    assert described["torch_version"] == "2.5.1"
    assert described["device"] == "cpu"


def test_missing_checkpoint_source_is_a_clear_error() -> None:
    config = ModelConfig(
        name="xtts_v2",
        adapter="rtb.generate.xtts:XttsAdapter",
        max_ref_seconds=6.0,
        default_sample_rate=SAMPLE_RATE,
        manifest_path=Path("data/manifests/xtts_v2.jsonl"),
    )
    with pytest.raises(ValueError, match="checkpoint_dir or hf_repo"):
        XttsAdapter(config)._resolve_checkpoint_dir()


# ── helpers ──────────────────────────────────────────────────────────────────────────────────


def test_as_float_list_accepts_arrays_and_sequences() -> None:
    class FakeTensor:
        def tolist(self) -> list[float]:
            return [0.1, 0.2]

    assert as_float_list(FakeTensor()) == [0.1, 0.2]
    assert as_float_list([0.1, 0.2]) == [0.1, 0.2]
    assert as_float_list([[0.1], [0.2]]) == [0.1, 0.2]


def test_write_wav_clamps_out_of_range_samples(tmp_path: Path) -> None:
    # Un modelo puede devolver muestras fuera de [-1, 1]; sin recortar, el int16 da la vuelta y
    # el wav suena a chasquidos en vez de a audio saturado.
    out = tmp_path / "clip.wav"
    write_wav(out, [2.0, -2.0, 0.0], SAMPLE_RATE)
    assert read_duration(out) == pytest.approx(3 / SAMPLE_RATE)

    with wave.open(str(out), "rb") as fh:
        samples = array.array("h")
        samples.frombytes(fh.readframes(fh.getnframes()))
    assert list(samples) == [32767, -32767, 0]
