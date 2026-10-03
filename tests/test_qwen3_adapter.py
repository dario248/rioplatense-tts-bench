"""Qwen3-TTS adapter logic, exercised against a fake model.

No torch and no GPU, same as the XTTS adapter tests: everything that could be wrong about the
*protocol* — whether the reference is passed whole, whether the transcript actually reaches the
model, which flag decides the input mode, what a timing measures — is plain Python and belongs
here. Whether the real checkpoint produces good Rioplatense audio needs the GPU box.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import pytest

from rtb.config import ConfigError, ModelConfig, load_bench_config
from rtb.generate.qwen3 import Qwen3Adapter
from rtb.schema import GenerationRow

from .conftest import write_wav as write_test_wav

SAMPLE_RATE = 24000
REF_TEXT = "El otro día ví un video sobre un aparatito que traducía."


class FakeQwen:
    """Stands in for `Qwen3TTSModel`, recording how it was called."""

    def __init__(self, seconds: float = 1.0, sample_rate: int = SAMPLE_RATE) -> None:
        self.seconds = seconds
        self.sample_rate = sample_rate
        self.prompt_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.raise_on_generate: BaseException | None = None
        self.return_no_audio = False

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[str]:
        self.prompt_calls.append(kwargs)
        return ["prompt-item"]

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[list[float]], int]:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate:
            raise self.raise_on_generate
        if self.return_no_audio:
            return [], self.sample_rate
        n = int(self.seconds * self.sample_rate)
        return [[0.5 * math.sin(i / 20.0) for i in range(n)]], self.sample_rate


class FakeAdapter(Qwen3Adapter):
    """The real adapter with only the heavy load replaced."""

    def __init__(self, config: ModelConfig, model: FakeQwen) -> None:
        super().__init__(config)
        self._fake = model

    def _load_model(self) -> Any:
        self._lib_version = "qwen-tts==9.9.9"
        self._torch_version = "2.13.0"
        self._device = "cpu"
        self._resolved_revision = "deadbeef"
        return self._fake


def model_config(**params: Any) -> ModelConfig:
    base: dict[str, Any] = {"language": "Spanish", "non_streaming_mode": True}
    base.update(params)
    return ModelConfig(
        name="qwen3_tts",
        adapter="rtb.generate.qwen3:Qwen3Adapter",
        hf_repo="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        max_ref_seconds=12.0,
        supports_streaming=False,
        default_sample_rate=SAMPLE_RATE,
        device="cpu",
        manifest_path=Path("data/manifests/qwen3_tts.jsonl"),
        params=base,
    )


@pytest.fixture
def loaded(tmp_path: Path) -> tuple[FakeAdapter, FakeQwen, Path]:
    model = FakeQwen()
    adapter = FakeAdapter(model_config(), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=8.2, sample_rate=22050)
    return adapter, model, ref


# ── el par (audio, texto) ────────────────────────────────────────────────────────────────────────


def test_the_reference_transcript_reaches_the_model(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert model.prompt_calls[0]["ref_text"] == REF_TEXT
    assert model.prompt_calls[0]["x_vector_only_mode"] is False


def test_the_reference_audio_is_passed_whole_not_trimmed(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    # El punto del modulo: recortar el audio a max_ref_seconds dejaria el texto describiendo mas
    # de lo que se oye, y un par desalineado clona peor SIN error.
    adapter, model, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert model.prompt_calls[0]["ref_audio"] == str(ref)
    assert result.ref_seconds_used == pytest.approx(8.2, abs=0.01)


def test_ref_seconds_used_reports_the_clip_and_not_the_budget(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    # Es la cifra con la que despues se compara la similitud contra XTTS, que recibe 6.0 s exactos.
    adapter, _, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert result.ref_seconds_used != adapter.max_ref_seconds


def test_an_empty_transcript_is_refused_in_icl_mode(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    adapter, _, ref = loaded
    with pytest.raises(ValueError, match="reference transcript"):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text="  ")


def test_the_row_declares_that_the_transcript_was_provided(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    # No es un detalle de implementacion: es una condicion experimental que XTTS no comparte.
    adapter, _, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert result.ref_text_provided is True


def test_x_vector_only_mode_drops_the_text_and_says_so(tmp_path: Path) -> None:
    model = FakeQwen()
    adapter = FakeAdapter(model_config(x_vector_only_mode=True), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=8.2, sample_rate=22050)
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    # El texto NO se pasa aunque lo hayan dado: el manifest tiene que decir lo que el modelo vio.
    assert model.prompt_calls[0]["ref_text"] is None
    assert result.ref_text_provided is False


# ── el protocolo de entrada ──────────────────────────────────────────────────────────────────────


def test_the_whole_sentence_is_fed_at_once(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    # El default de la libreria es non_streaming_mode=False, que segun su docstring "simula
    # entrada de texto en streaming". Todos los otros modelos reciben la frase entera.
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert model.generate_calls[0]["non_streaming_mode"] is True


def test_the_adapter_default_is_the_comparable_one_not_the_library_one() -> None:
    adapter = Qwen3Adapter(model_config(non_streaming_mode=None))
    assert adapter.non_streaming_mode is False  # explicito gana
    adapter = Qwen3Adapter(
        ModelConfig(
            name="qwen3_tts",
            adapter="rtb.generate.qwen3:Qwen3Adapter",
            max_ref_seconds=12.0,
            default_sample_rate=SAMPLE_RATE,
            manifest_path=Path("m.jsonl"),
            params={},
        )
    )
    assert adapter.non_streaming_mode is True


def test_the_language_is_explicit(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert model.generate_calls[0]["language"] == "Spanish"


def test_extra_params_are_forwarded_to_generate(tmp_path: Path) -> None:
    model = FakeQwen()
    adapter = FakeAdapter(model_config(top_p=0.9, temperature=0.7), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=22050)
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    call = model.generate_calls[0]
    assert call["top_p"] == 0.9 and call["temperature"] == 0.7
    assert call["max_new_tokens"] == 2048


# ── el prompt se cachea ──────────────────────────────────────────────────────────────────────────


def test_the_clone_prompt_is_built_once_per_reference(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    # Igual que los latents de XTTS: es lo que hace un deployment, y cobrarle el encode a la
    # primera de las 135 sintesis de un hablante la volveria un outlier (SPEC 6.3).
    adapter, model, ref = loaded
    for index in range(3):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / f"out{index}.wav", ref_text=REF_TEXT)
    assert len(model.prompt_calls) == 1
    assert len(model.generate_calls) == 3


def test_a_different_reference_gets_its_own_prompt(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    adapter, model, ref = loaded
    other = write_test_wav(tmp_path / "other.wav", seconds=6.6, sample_rate=22050)
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "a.wav", ref_text=REF_TEXT)
    adapter.synthesize("Dale.", other, 1234, tmp_path / "b.wav", ref_text="otro texto")
    assert len(model.prompt_calls) == 2


def test_unload_clears_the_cache(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    adapter, model, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "a.wav", ref_text=REF_TEXT)
    adapter.unload()
    assert not adapter.loaded
    with pytest.raises(RuntimeError, match="load"):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / "b.wav", ref_text=REF_TEXT)


# ── el audio y los timings ───────────────────────────────────────────────────────────────────────


def test_it_writes_a_readable_wav(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    adapter, _, ref = loaded
    out = tmp_path / "out.wav"
    result = adapter.synthesize("Dale.", ref, 1234, out, ref_text=REF_TEXT)
    assert result.ok
    with wave.open(str(out), "rb") as fh:
        assert (fh.getnchannels(), fh.getsampwidth()) == (1, 2)
        assert fh.getframerate() == SAMPLE_RATE


def test_the_sample_rate_comes_from_the_model_not_the_config(tmp_path: Path) -> None:
    # Asumirlo es como se escriben wavs con el header equivocado, que la etapa de metricas
    # despues lee como otra duracion.
    model = FakeQwen(sample_rate=16000)
    adapter = FakeAdapter(model_config(), model)
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=7.0, sample_rate=22050)
    out = tmp_path / "out.wav"
    result = adapter.synthesize("Dale.", ref, 1234, out, ref_text=REF_TEXT)
    assert result.sample_rate == 16000
    with wave.open(str(out), "rb") as fh:
        assert fh.getframerate() == 16000


def test_ttfb_equals_total_without_generation_streaming(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    # No es un placeholder: sin streaming de generacion el primer audio existe cuando termina todo.
    adapter, _, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert result.ttfb_ms == result.total_ms
    assert result.streaming is False


def test_a_model_exception_becomes_an_error_row(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    adapter, model, ref = loaded
    model.raise_on_generate = RuntimeError("CUDA out of memory")
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert not result.ok
    assert "CUDA out of memory" in (result.error or "")
    assert not (tmp_path / "out.wav").exists()


def test_no_audio_returned_is_an_error_not_an_empty_wav(
    loaded: tuple[FakeAdapter, FakeQwen, Path], tmp_path: Path
) -> None:
    adapter, model, ref = loaded
    model.return_no_audio = True
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", ref_text=REF_TEXT)
    assert not result.ok and "no audio" in (result.error or "")


def test_describe_carries_the_resolved_revision(
    loaded: tuple[FakeAdapter, FakeQwen, Path],
) -> None:
    adapter, _, _ = loaded
    described = adapter.describe()
    assert described["model_revision"] == "deadbeef"
    assert described["lib_version"] == "qwen-tts==9.9.9"
    # Cada fila del manifest lo lleva, y el schema exige que no sea nulo salvo en ground truth.
    GenerationRow(
        generation_id="qwen3_tts__s__p__s1__ref0",
        model="qwen3_tts",
        model_revision=described["model_revision"],
        lib_version=described["lib_version"],
        device=described["device"],
        speaker_id="s",
        ref_clip_idx=0,
        phrase_id="p",
        seed=1,
        wav_path="audio/qwen3_tts/s/p_s1_ref0.wav",
        sample_rate=SAMPLE_RATE,
        duration_s=1.0,
        ref_text_provided=True,
    )


# ── los nombres que invalidan el benchmark ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "repo",
    ["Qwen/Qwen3-TTS-Flash", "Qwen/Qwen-Audio-3.0-TTS"],
)
def test_a_hosted_only_checkpoint_is_refused(repo: str) -> None:
    # Los tres productos tienen nombres casi iguales y solo la linea 12Hz tiene pesos. Un config
    # que nombre a los otros no pudo haber corrido, asi que falla al construirse y no al cargar.
    config = model_config()
    with pytest.raises(ValueError, match="hosted-only"):
        Qwen3Adapter(config.model_copy(update={"hf_repo": repo}))


# ── la validacion que corre antes de sintetizar ──────────────────────────────────────────────────


QWEN_MODEL_YAML = """
name: qwen3_tts
adapter: rtb.generate.qwen3:Qwen3Adapter
hf_repo: Qwen/Qwen3-TTS-12Hz-1.7B-Base
requires_ref_text: true
max_ref_seconds: 12.0
supports_streaming: false
default_sample_rate: 24000
device: cpu
audio_root: audio
manifest_path: data/manifests/qwen3_tts.jsonl
params:
  language: Spanish
"""


def test_a_run_without_reference_transcripts_is_refused_at_config_time(mini_repo: Path) -> None:
    # `mini_repo` declara ref_clips SIN transcript. Fallar acá cuesta un segundo; fallar al
    # sintetizar cuesta la descarga de un checkpoint de 1.7 B primero.
    (mini_repo / "configs" / "models" / "qwen3.yaml").write_text(QWEN_MODEL_YAML, encoding="utf-8")
    with pytest.raises(ConfigError, match="requires_ref_text"):
        load_bench_config("configs/models/qwen3.yaml", root=mini_repo, check_paths=True)


def test_the_same_run_loads_once_the_transcripts_are_there(mini_repo: Path) -> None:
    (mini_repo / "configs" / "models" / "qwen3.yaml").write_text(QWEN_MODEL_YAML, encoding="utf-8")
    speakers = (mini_repo / "configs" / "speakers.yaml").read_text(encoding="utf-8")
    speakers = speakers.replace(
        "    - path: data/refs/f01_ref0.wav",
        '    - path: data/refs/f01_ref0.wav\n      transcript: "una dos tres"',
    )
    (mini_repo / "configs" / "speakers.yaml").write_text(speakers, encoding="utf-8")
    config = load_bench_config("configs/models/qwen3.yaml", root=mini_repo, check_paths=True)
    assert config.model.requires_ref_text


def test_a_model_that_does_not_need_the_text_loads_either_way(mini_repo: Path) -> None:
    # XTTS-v2 no lo necesita: la validacion no debe frenar a los modelos que clonan solo de audio.
    config = load_bench_config("configs/models/dummy.yaml", root=mini_repo, check_paths=True)
    assert config.model.requires_ref_text is False
