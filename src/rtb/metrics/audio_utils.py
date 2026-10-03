"""Reading and resampling audio for the metrics that need samples, not just a header.

`read_mono_pcm` is stdlib only, so the envelope and the duration screen run in CI; `load_resampled`
imports torchaudio lazily and only the speaker-similarity pass calls it. The 16 kHz resample is the
metric and not hygiene (SPEC 7.2).
"""

from __future__ import annotations

import array
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — solo para el type checker, no en runtime
    import torch

# 16 kHz porque es lo que espera ECAPA-TDNN. Cambiarlo invalida todos los cosenos, asi que va en
# el `metric_version` de cada fila.
TARGET_SAMPLE_RATE = 16000
FULL_SCALE = 32768.0


class AudioError(Exception):
    """The file is not the 16-bit mono PCM every stage of this project writes."""


def read_mono_pcm(path: Path) -> tuple[array.array, int]:
    """Raw 16-bit samples and their sample rate. Standard library only."""
    with wave.open(str(path), "rb") as fh:
        channels, width, rate = fh.getnchannels(), fh.getsampwidth(), fh.getframerate()
        frames = fh.readframes(fh.getnframes())
    if channels != 1 or width != 2:
        # Todo el pipeline escribe mono de 16 bits; un formato distinto es un error de otra etapa
        # y no algo para adivinar acá.
        raise AudioError(f"{path}: expected 16-bit mono, got {channels}ch/{width * 8}bit")
    samples = array.array("h")
    samples.frombytes(frames)
    return samples, rate


def load_resampled(path: Path, target_rate: int = TARGET_SAMPLE_RATE) -> Any:
    """One clip as a 1-D float32 tensor at `target_rate`, scaled to ±1.

    Returns a `torch.Tensor`; annotated `Any` so this module keeps importing where torch is not
    installed, which is most of them.
    """
    import torch  # noqa: PLC0415 — dependencia pesada: va acá
    import torchaudio  # noqa: PLC0415

    samples, rate = read_mono_pcm(path)
    if not len(samples):
        raise AudioError(f"{path}: no audio")
    audio: torch.Tensor = torch.tensor(samples, dtype=torch.float32) / FULL_SCALE
    if rate != target_rate:
        # Sinc interpolation de torchaudio, no un decimado ingenuo: el aliasing de un decimado se
        # ve como timbre y esta metrica mide justamente timbre.
        audio = torchaudio.functional.resample(audio, rate, target_rate)
    return audio
