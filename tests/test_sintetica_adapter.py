"""Sintética adapter logic, exercised against a fake model.

No torch and no coqui. Lo que puede estar mal del *protocolo* es Python plano y va acá: que la voz
salga del id y no del clip, que cada hablante caiga en su checkpoint, que sólo uno quede residente,
y que la fila declare que no consumió referencia. Si el checkpoint real suena bien es otra pregunta
y necesita los pesos.

Este adapter es el primero que elige la voz por id en vez de clonarla, así que la mitad de estos
tests son sobre el límite entre las dos clases de modelo.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import pytest

from rtb.config import ModelConfig
from rtb.generate.sintetica import SinteticaAdapter

from .conftest import write_wav as write_test_wav

SAMPLE_RATE = 22050


class FakeAudioProcessor:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate


class FakeModel:
    length_scale: float | None = None


class FakeSynthesizer:
    def __init__(self, sample_rate: int) -> None:
        self.vocoder_ap = FakeAudioProcessor(sample_rate)
        self.tts_model = FakeModel()


class FakeTTS:
    """Stands in for the vendored `TTS`, recording how it was called."""

    def __init__(self, checkpoint: str, seconds: float = 1.0, sr: int = SAMPLE_RATE) -> None:
        self.checkpoint = checkpoint
        self.seconds = seconds
        self.sr = sr
        self.calls: list[dict[str, Any]] = []
        self.synthesizer = FakeSynthesizer(sr)

    def tts(self, text: str, speaker: str) -> list[float]:
        self.calls.append({"text": text, "speaker": speaker})
        n = int(self.seconds * self.sr)
        return [0.5 * math.sin(i / 20.0) for i in range(n)]


class FakeAdapter(SinteticaAdapter):
    """The real adapter with only the heavy build replaced."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.built: list[str] = []
        self.models: dict[str, FakeTTS] = {}

    def load(self) -> None:
        self._lib_version = "coqui-tts==0.1.9"
        self._torch_version = "1.13.1"
        self._revision = "fem=aaaa,masc=bbbb,vocoder=cccc"
        self._device = "cpu"
        self._loaded = True

    def _build(self, checkpoint: Any) -> Any:
        self.built.append(checkpoint.name)
        model = FakeTTS(checkpoint.name)
        self.models[checkpoint.name] = model
        return model


def write_weights(root: Path) -> Path:
    """The on-disk layout the config points at, with the configs the adapter has to patch."""
    for name, prefix in (("adultos_femenino", "fem"), ("adultos_masculino", "mal")):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{prefix}.pth").write_bytes(b"weights")
        (folder / f"{prefix}_speakers.pth").write_bytes(b"speakers")
        (folder / "stats.npy").write_bytes(b"stats")
        (folder / f"config_{prefix}.json").write_text(
            json.dumps(
                {
                    "speakers_file": "/entrenamiento/viejo/speakers.pth",
                    "model_args": {"speakers_file": "/entrenamiento/viejo/speakers.pth"},
                    "audio": {"sample_rate": SAMPLE_RATE, "stats_path": "/viejo/stats.npy"},
                }
            )
        )
    vocoder = root / "vocoder"
    vocoder.mkdir(parents=True, exist_ok=True)
    (vocoder / "hifi.pth").write_bytes(b"vocoder")
    (vocoder / "config_hifi.json").write_text(json.dumps({"audio": {"stats_path": "/viejo.npy"}}))
    return root


def model_config(root: Path, **overrides: Any) -> ModelConfig:
    params: dict[str, Any] = {
        "weights_root": str(root),
        "voices": {
            "slr61_f_01": "arf_00295",
            "slr61_f_02": "arf_04766",
            "slr61_m_01": "arm_02484",
            "slr61_m_02": "arm_01523",
        },
        "checkpoints": [
            {
                "name": "adultos_femenino",
                "synth": "adultos_femenino/fem.pth",
                "config": "adultos_femenino/config_fem.json",
                "speakers": "adultos_femenino/fem_speakers.pth",
                "stats": "adultos_femenino/stats.npy",
                "voices": ["arf_00295", "arf_04766"],
            },
            {
                "name": "adultos_masculino",
                "synth": "adultos_masculino/mal.pth",
                "config": "adultos_masculino/config_mal.json",
                "speakers": "adultos_masculino/mal_speakers.pth",
                "stats": "adultos_masculino/stats.npy",
                "voices": ["arm_02484", "arm_01523"],
            },
        ],
        "vocoder": {"weights": "vocoder/hifi.pth", "config": "vocoder/config_hifi.json"},
        "speed": 1.0,
    }
    params.update(overrides)
    return ModelConfig(
        name="sintetica",
        adapter="rtb.generate.sintetica:SinteticaAdapter",
        max_ref_seconds=6.0,
        supports_streaming=False,
        default_sample_rate=SAMPLE_RATE,
        device="cpu",
        reference_only=True,
        audio_root=root / "audio",
        manifest_path=Path("data/manifests/sintetica.jsonl"),
        params=params,
    )


@pytest.fixture
def loaded(tmp_path: Path):
    adapter = FakeAdapter(model_config(write_weights(tmp_path / "weights")))
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=11.62, sample_rate=SAMPLE_RATE)
    return adapter, ref


# ── la voz sale del id, no del clip ──────────────────────────────────────────────────────────────


def test_the_voice_comes_from_the_speaker_id(loaded, tmp_path: Path) -> None:
    adapter, ref = loaded
    adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", speaker_id="slr61_f_02")
    call = adapter.models["adultos_femenino"].calls[-1]
    assert call["speaker"] == "arf_04766"


def test_a_row_without_a_speaker_is_an_error_and_not_a_default_voice(loaded, tmp_path) -> None:
    # Elegir una voz cualquiera produciria 540 filas medidas contra el hablante equivocado, y el
    # unico sintoma seria una similitud baja que se lee como un modelo malo.
    adapter, ref = loaded
    with pytest.raises(ValueError, match="selects its voice by id"):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav")


def test_an_unmapped_speaker_is_an_error(loaded, tmp_path: Path) -> None:
    adapter, ref = loaded
    with pytest.raises(ValueError, match="has no voice"):
        adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", speaker_id="slr61_f_09")


def test_a_voice_no_checkpoint_declares_is_caught_when_building_the_adapter(tmp_path) -> None:
    # Antes de sintetizar y no en la fila 300: el mapeo esta en el config y se valida al cargarlo.
    root = write_weights(tmp_path / "weights")
    config = model_config(root, voices={"slr61_f_01": "arf_inexistente"})
    with pytest.raises(ValueError, match="no checkpoint declares"):
        FakeAdapter(config)


def test_the_reference_clip_is_not_consumed(loaded, tmp_path: Path) -> None:
    # La fila lleva un clip porque es el OBJETIVO del coseno de similitud (ANALYSIS A15), no un
    # prompt. `ref_seconds_used: None` es lo que dice que no entro a la sintesis.
    adapter, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", speaker_id="slr61_f_01")
    assert result.ref_seconds_used is None
    assert result.ref_text_provided is False
    assert adapter.selects_voice_by_id


# ── los dos checkpoints ──────────────────────────────────────────────────────────────────────────


def test_each_speaker_lands_on_its_own_checkpoint(loaded, tmp_path: Path) -> None:
    adapter, ref = loaded
    adapter.synthesize("Dale.", ref, 1, tmp_path / "a.wav", speaker_id="slr61_f_01")
    adapter.synthesize("Dale.", ref, 1, tmp_path / "b.wav", speaker_id="slr61_m_01")
    assert adapter.built == ["adultos_femenino", "adultos_masculino"]


def test_only_one_checkpoint_stays_resident(loaded, tmp_path: Path) -> None:
    # Tener los dos en memoria duplicaria la RAM pico que mide SPEC 7.3, y el numero publicado
    # dejaria de parecerse al de un deployment.
    adapter, ref = loaded
    adapter.synthesize("Dale.", ref, 1, tmp_path / "a.wav", speaker_id="slr61_f_01")
    adapter.synthesize("Dale.", ref, 1, tmp_path / "b.wav", speaker_id="slr61_m_01")
    assert adapter._resident == "adultos_masculino"


def test_speakers_of_the_same_checkpoint_do_not_reload_it(loaded, tmp_path: Path) -> None:
    # La grilla itera por hablante, asi que la corrida entera cuesta UN recambio.
    adapter, ref = loaded
    adapter.synthesize("Dale.", ref, 1, tmp_path / "a.wav", speaker_id="slr61_f_01")
    adapter.synthesize("Dale.", ref, 1, tmp_path / "b.wav", speaker_id="slr61_f_02")
    assert adapter.built == ["adultos_femenino"]


# ── el config del checkpoint ─────────────────────────────────────────────────────────────────────


def test_the_trained_config_gets_its_paths_repointed(loaded) -> None:
    # Los configs entrenados traen paths del layout original. Sin reapuntarlos el modelo carga los
    # speakers equivocados o falla, segun que haya en esa ruta en esta maquina.
    adapter, _ = loaded
    checkpoint = adapter.checkpoints["adultos_femenino"]
    patched = Path(
        adapter._patched_config(checkpoint.config, checkpoint.speakers, checkpoint.stats)
    )
    try:
        data = json.loads(patched.read_text(encoding="utf-8"))
    finally:
        patched.unlink(missing_ok=True)
    assert data["speakers_file"] == str(checkpoint.speakers.resolve())
    assert data["model_args"]["speakers_file"] == str(checkpoint.speakers.resolve())
    assert data["audio"]["stats_path"] == str(checkpoint.stats.resolve())


def test_the_patched_config_is_a_temporary_file(loaded) -> None:
    adapter, _ = loaded
    checkpoint = adapter.checkpoints["adultos_femenino"]
    patched = Path(adapter._patched_config(checkpoint.config, checkpoint.speakers, None))
    try:
        assert patched != checkpoint.config
        assert json.loads(checkpoint.config.read_text())["speakers_file"].startswith("/entrena")
    finally:
        patched.unlink(missing_ok=True)


# ── lo que devuelve la sintesis ──────────────────────────────────────────────────────────────────


def test_the_wav_lands_at_the_sample_rate_the_vocoder_reports(loaded, tmp_path: Path) -> None:
    adapter, ref = loaded
    out = tmp_path / "out.wav"
    result = adapter.synthesize("Dale.", ref, 1234, out, speaker_id="slr61_f_01")
    assert result.sample_rate == SAMPLE_RATE
    with wave.open(str(out), "rb") as fh:
        assert fh.getframerate() == SAMPLE_RATE
        assert fh.getnframes() / fh.getframerate() == pytest.approx(1.0, abs=0.01)


def test_ttfb_equals_total_because_nothing_streams(loaded, tmp_path: Path) -> None:
    # Sin streaming el primer byte existe cuando termina todo: no es un relleno (SPEC 6.3).
    adapter, ref = loaded
    result = adapter.synthesize("Dale.", ref, 1234, tmp_path / "out.wav", speaker_id="slr61_f_01")
    assert result.ttfb_ms == result.total_ms
    assert result.streaming is False


def test_speed_drives_the_only_generation_knob_fastpitch_exposes(tmp_path: Path) -> None:
    root = write_weights(tmp_path / "weights")
    adapter = FakeAdapter(model_config(root, speed=2.0))
    adapter.load()
    ref = write_test_wav(tmp_path / "ref.wav", seconds=6.0, sample_rate=SAMPLE_RATE)
    adapter.synthesize("Dale.", ref, 1, tmp_path / "out.wav", speaker_id="slr61_f_01")
    assert adapter.models["adultos_femenino"].synthesizer.tts_model.length_scale == 0.5


def test_synthesizing_before_load_is_an_error(tmp_path: Path) -> None:
    adapter = FakeAdapter(model_config(write_weights(tmp_path / "weights")))
    ref = write_test_wav(tmp_path / "ref.wav", seconds=6.0, sample_rate=SAMPLE_RATE)
    with pytest.raises(RuntimeError, match="load"):
        adapter.synthesize("Dale.", ref, 1, tmp_path / "out.wav", speaker_id="slr61_f_01")


def test_an_unknown_param_is_rejected_instead_of_ignored(tmp_path: Path) -> None:
    # Un `temperture: 0.8` mal escrito no puede pasar en silencio: el config es la fuente de verdad
    # del experimento.
    root = write_weights(tmp_path / "weights")
    with pytest.raises(ValueError, match="unknown params"):
        FakeAdapter(model_config(root, temperture=0.8))


def test_the_weights_root_expands_the_environment_variable(tmp_path, monkeypatch) -> None:
    root = write_weights(tmp_path / "weights")
    monkeypatch.setenv("SINTETICA_TTS_ROOT", str(tmp_path))
    adapter = FakeAdapter(model_config(root, weights_root="${SINTETICA_TTS_ROOT}/weights"))
    assert adapter.weights_root == root


def test_an_undefined_weights_variable_is_an_error(tmp_path, monkeypatch) -> None:
    # Sin esto el literal `${SINTETICA_TTS_ROOT}` viaja hasta el sha256 de los checkpoints y sale
    # como un FileNotFoundError que no se parece a la causa.
    monkeypatch.delenv("SINTETICA_TTS_ROOT", raising=False)
    root = write_weights(tmp_path / "weights")
    with pytest.raises(ValueError, match="unexpanded variables"):
        FakeAdapter(model_config(root, weights_root="${SINTETICA_TTS_ROOT}/weights"))


def test_describe_says_the_voice_came_from_an_id(loaded) -> None:
    # Cambia como se lee la fila entera: sin esto, una similitud alta parece clonado.
    adapter, _ = loaded
    described = adapter.describe()
    assert described["voice_selection"] == "speaker_id"
    assert described["model_revision"] == "fem=aaaa,masc=bbbb,vocoder=cccc"
