"""Pocket TTS adapter logic, exercised against a fake model.

No torch and no GPU. Everything that could be wrong about the *protocol* — how many seconds of
reference the model sees, which kwargs reach the generator, what a timing means, where the sample
rate comes from, and whether the weights that loaded can clone at all — is plain Python and
belongs here. Whether the real checkpoint produces good Rioplatense audio needs the weights.

The refusal to run on the non-cloning fallback gets the most attention: it is the one failure that
would otherwise produce a full grid measuring a model that cannot do the thing being measured.
"""

from __future__ import annotations

import math
import types
from pathlib import Path
from typing import Any

import pytest

from rtb.config import ModelConfig
from rtb.generate.base import AdapterError
from rtb.generate.pocket import PocketAdapter

from .conftest import write_wav as write_test_wav

SAMPLE_RATE = 24000
REF_RATE = 22050
REVISION = "3e82814a68665eec246ff649b14c71331f955c06"


class FakePocket:
    """Stands in for `TTSModel`, recording how it was called."""

    def __init__(
        self,
        seconds: float = 1.0,
        sr: int = SAMPLE_RATE,
        *,
        has_voice_cloning: bool = True,
        revision: str | None = REVISION,
        chunks: int = 3,
    ) -> None:
        self.seconds = seconds
        self.sample_rate = sr
        self.has_voice_cloning = has_voice_cloning
        self.temp = 0.3
        weights = "hf://kyutai/pocket-tts/languages/spanish/model.safetensors"
        self.config = types.SimpleNamespace(
            weights_path=f"{weights}@{revision}" if revision else weights
        )
        self.chunks = chunks
        self.state_calls: list[str] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.raise_on_generate: BaseException | None = None
        self.return_empty = False

    def get_state_for_audio_prompt(self, audio_conditioning: Any, truncate: bool = False) -> Any:
        self.state_calls.append(str(audio_conditioning))
        return {"kv": str(audio_conditioning)}

    def _wave(self, n: int) -> list[float]:
        return [0.5 * math.sin(i / 20.0) for i in range(n)]

    def generate_audio(self, model_state: Any, text_to_generate: str, **kwargs: Any) -> Any:
        self.generate_calls.append(
            {"state": model_state, "text": text_to_generate, **kwargs}
        )
        if self.raise_on_generate:
            raise self.raise_on_generate
        if self.return_empty:
            return []
        return self._wave(int(self.seconds * self.sample_rate))

    def generate_audio_stream(self, model_state: Any, text_to_generate: str, **kwargs: Any) -> Any:
        self.stream_calls.append({"state": model_state, "text": text_to_generate, **kwargs})
        if self.raise_on_generate:
            raise self.raise_on_generate
        total = int(self.seconds * self.sample_rate)
        size = max(1, total // self.chunks)
        for start in range(0, total, size):
            yield self._wave(min(size, total - start))


class FakeAdapter(PocketAdapter):
    """The real adapter with only the heavy load replaced."""

    def __init__(self, config: ModelConfig, model: FakePocket) -> None:
        super().__init__(config)
        self._fake = model

    def _load_model(self) -> Any:
        self._lib_version = "pocket-tts==9.9.9"
        self._torch_version = "2.6.0"
        self._device = "cpu"
        return self._fake


def model_config(
    *,
    max_ref_seconds: float = 6.0,
    audio_root: Path | None = None,
    device: str = "cpu",
    name: str = "pocket_tts_es",
    model_revision: str | None = REVISION,
    **params: Any,
) -> ModelConfig:
    base: dict[str, Any] = {"language": "spanish", "max_tokens": 50}
    base.update(params)
    return ModelConfig(
        name=name,
        adapter="rtb.generate.pocket:PocketAdapter",
        hf_repo="kyutai/pocket-tts",
        model_revision=model_revision,
        max_ref_seconds=max_ref_seconds,
        supports_streaming=True,
        default_sample_rate=SAMPLE_RATE,
        device=device,
        audio_root=audio_root or Path("audio"),
        manifest_path=Path("data/manifests/pocket_tts_es.jsonl"),
        params=base,
    )


@pytest.fixture
def loaded(tmp_path: Path):
    model = FakePocket()
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=11.62, sample_rate=REF_RATE)
    return adapter, model, ref


# ── los pesos que cargaron ───────────────────────────────────────────────────────────────────────


def test_loading_the_non_cloning_fallback_is_refused(tmp_path: Path) -> None:
    # El modo de falla que este chequeo existe para matar: la libreria tiene un segundo
    # weights_path al repo sin gate y cae ahi ante CUALQUIER fallo de descarga. Esos pesos no
    # clonan, asi que la corrida entera mediria otra cosa.
    model = FakePocket(has_voice_cloning=False)
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    with pytest.raises(AdapterError, match="WITHOUT voice cloning"):
        adapter.load()
    assert not adapter.loaded


def test_the_revision_comes_from_the_weights_path_the_library_pinned(loaded) -> None:
    adapter, _, _ = loaded
    assert adapter.describe()["model_revision"] == REVISION


def test_a_library_that_pins_other_weights_is_reported_not_hidden(tmp_path: Path, caplog) -> None:
    # El pin viaja adentro del paquete: subir la libreria puede cambiar los pesos sin que el YAML
    # se entere, y lo que vale es lo que corrio.
    other = "f" * 40
    model = FakePocket(revision=other)
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    with caplog.at_level("WARNING"):
        adapter.load()
    assert adapter.describe()["model_revision"] == other
    assert "pins weights" in caplog.text


def test_weights_without_a_revision_fall_back_to_the_config(tmp_path: Path) -> None:
    model = FakePocket(revision=None)
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    adapter.load()
    assert adapter.describe()["model_revision"] == REVISION


# ── el recorte de la referencia ──────────────────────────────────────────────────────────────────


def test_the_model_receives_the_trimmed_copy_not_the_original(loaded, tmp_path: Path) -> None:
    # `get_state_for_audio_prompt` lee el clip entero: su unico recorte propio es a 30 s.
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    trimmed, seconds = adapter._trimmer.trimmed(ref)
    assert model.state_calls == [str(trimmed)]
    assert seconds == pytest.approx(6.0, abs=0.01)


def test_ref_seconds_used_reports_what_was_handed_over(loaded, tmp_path: Path) -> None:
    # 6.0 y no 11.62: es la cifra con la que se compara la similitud contra XTTS-v2 y Chatterbox,
    # que reciben 6.0 exactos.
    adapter, _, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert result.ref_seconds_used == pytest.approx(6.0, abs=0.01)


def test_a_short_reference_is_passed_through_untouched(tmp_path: Path) -> None:
    model = FakePocket()
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "short.wav", seconds=5.0, sample_rate=REF_RATE)
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert model.state_calls == [str(ref)]


# ── el estado de voz se cachea ───────────────────────────────────────────────────────────────────


def test_the_voice_state_is_encoded_once_per_reference(loaded, tmp_path: Path) -> None:
    # Encodear el prompt es el paso caro y va fuera del reloj (SPEC 6.3): cobrarselo a la primera
    # de las 135 sintesis de un hablante la volveria un outlier.
    adapter, model, ref = loaded
    for index in range(3):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / f"out{index}.wav")
    assert len(model.state_calls) == 1
    assert len(model.generate_calls) == 3


def test_a_second_reference_gets_its_own_state(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    other = write_test_wav(tmp_path / "other.wav", seconds=7.0, sample_rate=REF_RATE)
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "a.wav")
    adapter.synthesize("Dale.", other, 1234, tmp_path / "b.wav")
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "c.wav")
    # Vuelve al primero sin reencodear: `generate_audio` copia el estado antes de usarlo, asi que
    # uno cacheado sirve a todas las filas del hablante.
    assert len(model.state_calls) == 2


def test_unload_drops_the_cached_states(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "a.wav")
    adapter.unload()
    assert not adapter.loaded
    adapter.load()
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "b.wav")
    assert len(model.state_calls) == 2


# ── que recibe el generador ──────────────────────────────────────────────────────────────────────


def test_the_declared_params_reach_the_generator(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    call = model.generate_calls[0]
    assert call["text"] == "Dale."
    assert call["max_tokens"] == 50


def test_max_tokens_is_configurable_because_it_chunks_the_text(tmp_path: Path) -> None:
    # No limita la salida: parte el texto. Se mide en el default del vendor, pero una frase larga
    # cortada de otro modo es otro experimento, asi que el valor viaja en el config.
    model = FakePocket()
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio", max_tokens=120), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=REF_RATE)
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert model.generate_calls[0]["max_tokens"] == 120


def test_the_reference_transcript_is_never_passed(loaded, tmp_path: Path) -> None:
    # Clona solo desde audio. La asimetria contra Qwen3-TTS se registra por fila (SPEC 6.4).
    adapter, model, ref = loaded
    result = adapter.synthesize(
        "Dale.", ref, 1234, tmp_path / "out.wav", ref_text="hola que tal"
    )
    assert result.ref_text_provided is False
    assert "hola que tal" not in str(model.generate_calls[0])


def test_the_speaker_id_is_ignored(loaded, tmp_path: Path) -> None:
    # El runner lo pasa siempre; lo usan solo los modelos que eligen la voz por id (SPEC 6.5).
    adapter, _, ref = loaded
    result = adapter.synthesize(
        "Dale.", ref, 1234, tmp_path / "out.wav", speaker_id="slr61_f_01"
    )
    assert result.ok


# ── los timings y el audio ───────────────────────────────────────────────────────────────────────


def test_whole_mode_reports_ttfb_equal_to_total(loaded, tmp_path: Path) -> None:
    # Sin streaming el primer audio existe cuando termina todo: es el dato correcto, no relleno.
    adapter, _, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert result.streaming is False
    assert result.ttfb_ms == result.total_ms


def test_streaming_mode_reports_a_first_chunk_before_the_end(tmp_path: Path) -> None:
    model = FakePocket(seconds=2.0, chunks=4)
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio", streaming=True), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=REF_RATE)
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert result.streaming is True
    assert len(model.stream_calls) == 1 and not model.generate_calls
    assert result.ttfb_ms is not None and result.total_ms is not None
    assert result.ttfb_ms <= result.total_ms
    # Los chunks se concatenan enteros: un stream que pierde audio daria una duracion mas corta.
    assert result.duration_s == pytest.approx(2.0, abs=0.01)


def test_the_sample_rate_comes_from_the_model(tmp_path: Path) -> None:
    # Lo que se escribe en el wav y en el manifest es lo que devolvio el modelo, no lo declarado.
    model = FakePocket(sr=16000)
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=REF_RATE)
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert result.sample_rate == 16000


def test_the_wav_lands_on_disk_with_the_reported_duration(loaded, tmp_path: Path) -> None:
    import wave

    adapter, _, ref = loaded
    out = tmp_path / "out.wav"
    result = adapter.synthesize("Dale.", ref, 1234, out)
    with wave.open(str(out), "rb") as fh:
        assert fh.getnframes() / fh.getframerate() == pytest.approx(result.duration_s, abs=0.01)


# ── las fallas son datos ─────────────────────────────────────────────────────────────────────────


def test_a_model_that_raises_becomes_an_error_row(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    model.raise_on_generate = RuntimeError("boom")
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert not result.ok
    assert "RuntimeError: boom" in (result.error or "")


def test_an_empty_waveform_is_an_error(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    model.return_empty = True
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert not result.ok
    assert not (tmp_path / "out.wav").exists()


def test_synthesizing_before_load_is_a_programming_error(tmp_path: Path) -> None:
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), FakePocket())
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=REF_RATE)
    with pytest.raises(RuntimeError, match="load"):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")


# ── el config se valida al construir ─────────────────────────────────────────────────────────────


def test_an_unknown_param_is_rejected(tmp_path: Path) -> None:
    # Ignorarlo en silencio dejaria la corrida midiendo otra configuracion que la declarada.
    with pytest.raises(ValueError, match="unknown params"):
        FakeAdapter(model_config(audio_root=tmp_path / "audio", temperture=0.3), FakePocket())


def test_an_unknown_language_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="language must be one of"):
        FakeAdapter(model_config(audio_root=tmp_path / "audio", language="espanol"), FakePocket())


def test_quantizing_on_cuda_is_rejected_before_anything_downloads(tmp_path: Path) -> None:
    # La libreria lo deja construir y revienta recien al sintetizar, con un NotImplementedError
    # del backend que no nombra ni a quantize ni al device.
    with pytest.raises(ValueError, match="CPU-only"):
        FakeAdapter(
            model_config(audio_root=tmp_path / "audio", device="cuda", quantize=True), FakePocket()
        )
