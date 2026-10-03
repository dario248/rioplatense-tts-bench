"""The energy envelope: where sound starts and stops, so speech can be told from padding."""

from __future__ import annotations

import array
import math
import wave
from pathlib import Path

import pytest

from rtb.metrics import audio_shape
from rtb.metrics.asr import AsrConfig, Transcriber, installed_version
from rtb.metrics.audio_utils import AudioError

from .conftest import make_thresholds

THRESHOLDS = make_thresholds().audio_shape
RATE = 24000


def write_clip(
    path: Path,
    *,
    lead_s: float = 0.0,
    tone_s: float = 1.0,
    tail_s: float = 0.0,
    amplitude: float = 0.5,
) -> Path:
    """A clip made of silence, a tone, and silence — so the expected answer is known exactly."""
    samples = array.array("h", [0] * int(lead_s * RATE))
    peak = int(amplitude * 32767)
    samples.extend(
        int(peak * math.sin(2 * math.pi * 220 * i / RATE)) for i in range(int(tone_s * RATE))
    )
    samples.extend([0] * int(tail_s * RATE))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes(samples.tobytes())
    return path


def test_it_finds_the_tone_between_the_silences(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "a.wav", lead_s=0.2, tone_s=1.0, tail_s=0.5)
    shape = audio_shape.measure(clip, THRESHOLDS)

    assert shape.total_s == pytest.approx(1.7, abs=0.02)
    assert shape.speech_start_s == pytest.approx(0.2, abs=0.02)
    assert shape.speech_end_s == pytest.approx(1.2, abs=0.02)
    assert shape.speech_s == pytest.approx(1.0, abs=0.04)
    assert shape.lead_silence_s == pytest.approx(0.2, abs=0.02)
    assert shape.tail_silence_s == pytest.approx(0.5, abs=0.02)
    assert shape.padding_s == pytest.approx(0.7, abs=0.04)


def test_padding_is_what_the_duration_screen_could_not_see(tmp_path: Path) -> None:
    # El mismo habla con distinto relleno da duraciones totales distintas, y `duration_s` no puede
    # decir cual de las dos cosas cambio: un modelo lento y uno que emite cola se ven igual. Estas
    # dos cifras si los distinguen.
    trimmed = audio_shape.measure(write_clip(tmp_path / "t.wav", tone_s=1.0), THRESHOLDS)
    padded = audio_shape.measure(
        write_clip(tmp_path / "p.wav", lead_s=0.15, tone_s=1.0, tail_s=0.45), THRESHOLDS
    )
    assert padded.total_s > trimmed.total_s
    assert padded.speech_s == pytest.approx(trimmed.speech_s, abs=0.05)
    assert padded.padding_s > 0.5
    assert trimmed.padding_s == pytest.approx(0.0, abs=0.05)


def test_a_silent_clip_is_reported_as_silent_not_as_speech(tmp_path: Path) -> None:
    # Un umbral relativo sobre un archivo mudo encontraria "habla" en el ruido de fondo, y un
    # modelo que devuelve silencio —modo de falla real con inputs cortos— pasaria por bueno.
    quiet = write_clip(tmp_path / "q.wav", tone_s=1.0, amplitude=0.002)
    shape = audio_shape.measure(quiet, THRESHOLDS)
    assert shape.silent
    assert shape.speech_s == 0.0


def test_stereo_or_wrong_width_is_a_loud_error(tmp_path: Path) -> None:
    # Todo el pipeline escribe mono de 16 bits: otro formato es un bug de otra etapa. El lector
    # vive en audio_utils, compartido con el pase de similitud, asi que el error tambien.
    stereo = tmp_path / "s.wav"
    with wave.open(str(stereo), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes(array.array("h", [0] * 100).tobytes())
    with pytest.raises(AudioError, match="16-bit mono"):
        audio_shape.measure(stereo, THRESHOLDS)


def test_shape_row_reports_a_broken_file_instead_of_raising(tmp_path: Path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav")
    row = audio_shape.shape_row("gen__1", broken, THRESHOLDS, "1")
    assert row.error
    assert row.metric == "audio_shape"


def test_shape_row_carries_the_threshold_version(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "a.wav", tone_s=0.5)
    row = audio_shape.shape_row("gen__1", clip, THRESHOLDS, "7")
    assert row.metric_version == "7"
    assert row.values["speech_s"] > 0


def test_real_reference_clips_have_almost_no_padding(repo_root: Path) -> None:
    # Es el hecho que sesga el screening de duracion: la vara viene recortada y el sintetizado no.
    clip = repo_root / "data" / "refs" / "slr61" / "slr61_f_01" / "calib_00.wav"
    if not clip.exists():
        pytest.skip("reference clips not built in this checkout")
    shape = audio_shape.measure(clip, THRESHOLDS)
    assert shape.padding_s < 0.1, f"padding {shape.padding_s:.3f}s"


# ── identidad del ASR ────────────────────────────────────────────────────────────────────────


def test_the_asr_identity_is_the_metric_version() -> None:
    # La cuantizacion cambia las transcripciones, asi que un WER sin el modelo y el compute type
    # al lado no es reproducible. Mismo argumento que model_revision en generacion.
    version = Transcriber(AsrConfig(model="large-v3", compute_type="int8_float16")).metric_version
    assert version.startswith("large-v3/int8_float16/faster-whisper==")


def test_a_different_compute_type_is_a_different_metric_version() -> None:
    a = Transcriber(AsrConfig(compute_type="int8_float16")).metric_version
    b = Transcriber(AsrConfig(compute_type="float16")).metric_version
    assert a != b


def test_the_version_is_readable_without_importing_faster_whisper() -> None:
    # El chequeo de idempotencia necesita la version ANTES de cargar el modelo, y cargar un modelo
    # para averiguarla anularia el ahorro.
    assert installed_version().startswith("faster-whisper==")


def test_transcribing_before_load_is_an_error(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "a.wav", tone_s=0.2)
    with pytest.raises(RuntimeError, match="load\\(\\) was not called"):
        Transcriber(AsrConfig()).transcribe(clip)
