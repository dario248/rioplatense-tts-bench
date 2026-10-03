"""Chatterbox adapter logic, exercised against a fake model.

No torch and no GPU. Everything that could be wrong about the *protocol* — how many seconds of
reference the model actually sees, which kwargs reach `generate`, what a timing measures, where the
sample rate comes from — is plain Python and belongs here. Whether the real checkpoint produces
good Rioplatense audio needs the GPU box.

The reference trimming gets the most attention on purpose: it is the one thing this adapter does
that the library does not, and the reason it exists is that Chatterbox reads the reference at three
different lengths at once.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import pytest

from rtb.config import ModelConfig
from rtb.generate.chatterbox import ChatterboxAdapter

from .conftest import write_wav as write_test_wav

SAMPLE_RATE = 24000
REF_RATE = 22050


class FakeChatterbox:
    """Stands in for `ChatterboxMultilingualTTS`, recording how it was called."""

    def __init__(self, seconds: float = 1.0, sr: int = SAMPLE_RATE) -> None:
        self.seconds = seconds
        self.sr = sr
        self.conditional_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.raise_on_generate: BaseException | None = None
        self.return_empty = False

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5) -> None:
        self.conditional_calls.append({"wav_fpath": wav_fpath, "exaggeration": exaggeration})

    def generate(self, text: str, **kwargs: Any) -> list[list[float]]:
        self.generate_calls.append({"text": text, **kwargs})
        if self.raise_on_generate:
            raise self.raise_on_generate
        if self.return_empty:
            return [[]]
        n = int(self.seconds * self.sr)
        # (1, N) como devuelve la libreria: `as_float_list` tiene que aplanarlo.
        return [[0.5 * math.sin(i / 20.0) for i in range(n)]]


class FakeAdapter(ChatterboxAdapter):
    """The real adapter with only the heavy load replaced."""

    def __init__(self, config: ModelConfig, model: FakeChatterbox) -> None:
        super().__init__(config)
        self._fake = model

    def _load_model(self) -> Any:
        self._lib_version = "chatterbox-tts==9.9.9"
        self._torch_version = "2.6.0"
        self._device = "cpu"
        self._resolved_revision = "cafebabe"
        return self._fake


def model_config(*, max_ref_seconds: float = 6.0, audio_root: Path | None = None, **params: Any):
    base: dict[str, Any] = {"language_id": "es", "exaggeration": 0.5, "cfg_weight": 0.5}
    base.update(params)
    return ModelConfig(
        name="chatterbox_v3",
        adapter="rtb.generate.chatterbox:ChatterboxAdapter",
        hf_repo="ResembleAI/chatterbox",
        max_ref_seconds=max_ref_seconds,
        supports_streaming=False,
        default_sample_rate=SAMPLE_RATE,
        device="cpu",
        audio_root=audio_root or Path("audio"),
        manifest_path=Path("data/manifests/chatterbox_v3.jsonl"),
        params=base,
    )


@pytest.fixture
def loaded(tmp_path: Path):
    model = FakeChatterbox()
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=11.62, sample_rate=REF_RATE)
    return adapter, model, ref


# ── el recorte de la referencia ──────────────────────────────────────────────────────────────────


def test_a_long_reference_is_trimmed_to_the_budget(loaded, tmp_path: Path) -> None:
    # El motivo de existir de este recorte: sin el, el voice encoder de Chatterbox ve el clip
    # ENTERO (11.62 s acá) mientras el prompt T3 ve 6 s y el decoder 10 s. Tres largos distintos
    # para la misma fila del manifest.
    adapter, _, ref = loaded
    trimmed, seconds = adapter.trimmed_reference(ref)
    assert trimmed != ref
    assert seconds == pytest.approx(6.0, abs=0.01)
    with wave.open(str(trimmed), "rb") as fh:
        assert fh.getnframes() / fh.getframerate() == pytest.approx(6.0, abs=0.01)


def test_the_trimmed_copy_keeps_format_and_rate(loaded) -> None:
    # Sin resamplear ni cambiar niveles: la misma regla con la que se construyeron los clips.
    adapter, _, ref = loaded
    trimmed, _ = adapter.trimmed_reference(ref)
    with wave.open(str(trimmed), "rb") as fh:
        assert (fh.getnchannels(), fh.getsampwidth(), fh.getframerate()) == (1, 2, REF_RATE)


def test_the_trim_takes_the_start_of_the_clip(loaded) -> None:
    adapter, _, ref = loaded
    trimmed, _ = adapter.trimmed_reference(ref)
    with wave.open(str(ref), "rb") as source:
        head = source.readframes(int(6.0 * REF_RATE))
    with wave.open(str(trimmed), "rb") as fh:
        assert fh.readframes(fh.getnframes()) == head


def test_a_short_reference_is_passed_through_untouched(tmp_path: Path) -> None:
    # No se escribe una copia identica: seria un archivo mas para desincronizarse.
    model = FakeChatterbox()
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "short.wav", seconds=5.0, sample_rate=REF_RATE)
    trimmed, seconds = adapter.trimmed_reference(ref)
    assert trimmed == ref
    assert seconds == pytest.approx(5.0, abs=0.01)


def test_the_trim_is_written_once_and_cached(loaded) -> None:
    adapter, _, ref = loaded
    first, _ = adapter.trimmed_reference(ref)
    mtime = first.stat().st_mtime_ns
    second, _ = adapter.trimmed_reference(ref)
    assert second == first
    assert first.stat().st_mtime_ns == mtime


def test_two_budgets_never_collide_on_disk(tmp_path: Path) -> None:
    # El nombre del recorte lleva el presupuesto: si no, cambiar max_ref_seconds reusaria en
    # silencio el recorte viejo y la corrida mediria otra referencia que la declarada.
    ref = write_test_wav(tmp_path / "ref.wav", seconds=11.62, sample_rate=REF_RATE)
    paths = []
    for budget in (6.0, 8.0):
        adapter = FakeAdapter(
            model_config(max_ref_seconds=budget, audio_root=tmp_path / "audio"), FakeChatterbox()
        )
        adapter.load()
        trimmed, seconds = adapter.trimmed_reference(ref)
        paths.append(trimmed)
        assert seconds == pytest.approx(budget, abs=0.01)
    assert paths[0] != paths[1]


def test_the_model_receives_the_trimmed_copy_not_the_original(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    trimmed, _ = adapter.trimmed_reference(ref)
    assert model.conditional_calls[0]["wav_fpath"] == str(trimmed)


def test_ref_seconds_used_reports_what_the_model_saw(loaded, tmp_path: Path) -> None:
    # 6.0 y no 11.62: es la cifra con la que se compara la similitud contra XTTS-v2, que tambien
    # recibe 6.0 exactos.
    adapter, _, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert result.ref_seconds_used == pytest.approx(6.0, abs=0.01)


# ── el conditioning se cachea ────────────────────────────────────────────────────────────────────


def test_the_conditionals_are_prepared_once_per_reference(loaded, tmp_path: Path) -> None:
    # La primera llamada real cuesta ~11 s (inicializacion perezosa) y las siguientes ~90 ms.
    # Cobrarsela a la primera de las 135 sintesis de un hablante la volveria un outlier (SPEC 6.3).
    adapter, model, ref = loaded
    for index in range(3):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / f"out{index}.wav")
    assert len(model.conditional_calls) == 1
    assert len(model.generate_calls) == 3


def test_switching_reference_reconditions(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    other = write_test_wav(tmp_path / "other.wav", seconds=7.0, sample_rate=REF_RATE)
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "a.wav")
    adapter.synthesize("Dale.", other, 1234, tmp_path / "b.wav")
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "c.wav")
    assert len(model.conditional_calls) == 3  # vuelve a condicionar al volver al primero


def test_unload_clears_the_caches(loaded, tmp_path: Path) -> None:
    adapter, _, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "a.wav")
    adapter.unload()
    assert not adapter.loaded
    with pytest.raises(RuntimeError, match="load"):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / "b.wav")


# ── que llega a generate ─────────────────────────────────────────────────────────────────────────


def test_the_language_is_explicit(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert model.generate_calls[0]["language_id"] == "es"


def test_the_vendor_sampling_params_are_forwarded(tmp_path: Path) -> None:
    model = FakeChatterbox()
    adapter = FakeAdapter(
        model_config(
            audio_root=tmp_path / "audio",
            cfg_weight=0.5,
            temperature=0.8,
            repetition_penalty=2.0,
            min_p=0.05,
            top_p=1.0,
        ),
        model,
    )
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=REF_RATE)
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    call = model.generate_calls[0]
    assert call["cfg_weight"] == 0.5
    assert call["temperature"] == 0.8
    assert call["min_p"] == 0.05


def test_exaggeration_reaches_both_conditioning_and_generation(tmp_path: Path) -> None:
    # Entra en las dos, y por eso es parte de la clave del cache: cambiarlo sin recondicionar
    # mediria una emocion en el prompt y otra en la generacion.
    model = FakeChatterbox()
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio", exaggeration=0.7), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=REF_RATE)
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert model.conditional_calls[0]["exaggeration"] == 0.7
    assert model.generate_calls[0]["exaggeration"] == 0.7


def test_an_unknown_param_is_refused(tmp_path: Path) -> None:
    # Un nombre mal escrito se ignoraria en silencio y la corrida mediria otra configuracion.
    with pytest.raises(ValueError, match="unknown params"):
        ChatterboxAdapter(model_config(exagerration=0.7))


def test_the_english_only_class_is_not_what_we_drive() -> None:
    # `ChatterboxTTS` existe y es la variante solo-ingles: usarla mediria otro modelo.
    import rtb.generate.chatterbox as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "ChatterboxMultilingualTTS" in source
    assert "from chatterbox import ChatterboxTTS" not in source


# ── el audio y los timings ───────────────────────────────────────────────────────────────────────


def test_it_writes_a_readable_wav(loaded, tmp_path: Path) -> None:
    adapter, _, ref = loaded
    out = tmp_path / "out.wav"
    result = adapter.synthesize("Dale.", ref, 1234, out)
    assert result.ok
    with wave.open(str(out), "rb") as fh:
        assert (fh.getnchannels(), fh.getsampwidth()) == (1, 2)
        assert fh.getframerate() == SAMPLE_RATE


def test_the_sample_rate_comes_from_the_model(tmp_path: Path) -> None:
    model = FakeChatterbox(sr=16000)
    adapter = FakeAdapter(model_config(audio_root=tmp_path / "audio"), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=REF_RATE)
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert result.sample_rate == 16000


def test_ttfb_equals_total_and_the_row_says_audio_only(loaded, tmp_path: Path) -> None:
    adapter, _, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert result.ttfb_ms == result.total_ms
    assert result.streaming is False
    # Clona solo desde audio: la asimetria contra Qwen3-TTS queda declarada por fila.
    assert result.ref_text_provided is False


def test_a_reference_transcript_is_accepted_and_ignored(loaded, tmp_path: Path) -> None:
    # La ABC lo pasa siempre; este modelo no lo usa y la fila tiene que decirlo.
    adapter, _, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text="algo")
    assert result.ok and result.ref_text_provided is False


def test_a_model_exception_becomes_an_error_row(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    model.raise_on_generate = RuntimeError("CUDA out of memory")
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert not result.ok and "CUDA out of memory" in (result.error or "")
    assert not (tmp_path / "out.wav").exists()


def test_an_empty_waveform_is_an_error_not_a_silent_wav(loaded, tmp_path: Path) -> None:
    adapter, model, ref = loaded
    model.return_empty = True
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")
    assert not result.ok and "empty waveform" in (result.error or "")


def test_describe_carries_the_resolved_revision(loaded) -> None:
    adapter, _, _ = loaded
    described = adapter.describe()
    assert described["model_revision"] == "cafebabe"
    assert described["lib_version"] == "chatterbox-tts==9.9.9"
