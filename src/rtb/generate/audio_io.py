"""Reading and writing wavs in the generation stage. Standard library only.

Shared by every adapter, so no model environment needs an audio dependency on top of its TTS stack
and an adapter can be tested where neither numpy nor torch is installed. An adapter must never
import another adapter, or the import graph starts deciding which environment needs which library.
"""

from __future__ import annotations

import array
import wave
from pathlib import Path
from typing import Any

FULL_SCALE = 32767


def read_duration(path: Path) -> float:
    """Duration of a wav from its header alone, so no audio library is needed."""
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        frames = fh.getnframes()
    if rate <= 0 or frames <= 0:
        raise ValueError(f"reference clip has no audio: {path}")
    return frames / float(rate)


def as_float_list(samples: Any) -> list[float]:
    """Flatten whatever the model returned into plain floats.

    Accepts numpy arrays and torch tensors (both expose `tolist()`) as well as plain sequences,
    which is what lets the adapters be unit-tested in an environment with neither installed.
    """
    if hasattr(samples, "tolist"):
        samples = samples.tolist()
    flat: list[float] = []
    for value in samples:
        if isinstance(value, (list, tuple)):
            flat.extend(float(inner) for inner in value)
        else:
            flat.append(float(value))
    return flat


def write_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    """Write float samples in [-1, 1] as 16-bit mono PCM.

    Pure Python on purpose: it keeps the adapters free of an audio-writing dependency, and it runs
    outside the timed section, so the tens of milliseconds it costs never land in `total_ms`.
    """
    pcm = array.array(
        "h", (int(max(-1.0, min(1.0, value)) * FULL_SCALE) for value in samples)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
