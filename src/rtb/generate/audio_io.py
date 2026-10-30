"""Reading and writing wavs in the generation stage. Standard library only.

Shared by every adapter, so no model environment needs an audio dependency on top of its TTS stack
and an adapter can be tested where neither numpy nor torch is installed. An adapter must never
import another adapter, or the import graph starts deciding which environment needs which library.
"""

from __future__ import annotations

import array
import logging
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


class ReferenceTrimmer:
    """Cuts each reference clip to a seconds budget, once, and remembers the copy.

    Shared because two models need the same cut for different reasons: Chatterbox reads the
    reference at three lengths at once and Pocket TTS reads it whole (SPEC 6.4). Two copies of
    this would drift into two different references and the manifest would not show it.
    """

    def __init__(self, out_dir: Path, max_seconds: float) -> None:
        self.out_dir = out_dir
        self.max_seconds = float(max_seconds)
        self._cache: dict[tuple[str, float], tuple[Path, float]] = {}

    def path_for(self, ref_wav: Path) -> Path:
        """Where the trimmed copy goes. Named after the budget, so two budgets never collide."""
        budget = f"{self.max_seconds:g}".replace(".", "_")
        return self.out_dir / f"{ref_wav.parent.name}__{ref_wav.stem}__{budget}s.wav"

    def trimmed(self, ref_wav: Path) -> tuple[Path, float]:
        """The clip cut to the budget, and the seconds it really holds.

        Cut from the START, whole samples, no resampling and no level change: the same rule the
        reference clips were built with.
        """
        key = (str(ref_wav), self.max_seconds)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        full_seconds = read_duration(ref_wav)
        if full_seconds <= self.max_seconds:
            # Ya entra en el presupuesto: se pasa tal cual, sin escribir una copia identica.
            result = (ref_wav, full_seconds)
            self._cache[key] = result
            return result

        with wave.open(str(ref_wav), "rb") as source:
            channels, width, rate = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            )
            keep = min(source.getnframes(), int(self.max_seconds * rate))
            frames = source.readframes(keep)

        target = self.path_for(ref_wav)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as sink:
            sink.setnchannels(channels)
            sink.setsampwidth(width)
            sink.setframerate(rate)
            sink.writeframes(frames)

        seconds = keep / float(rate)
        logger.info(
            "reference trimmed",
            extra={
                "fields": {
                    "source": str(ref_wav),
                    "trimmed": str(target),
                    "from_seconds": round(full_seconds, 3),
                    "to_seconds": round(seconds, 3),
                }
            },
        )
        result = (target, seconds)
        self._cache[key] = result
        return result

    def clear(self) -> None:
        self._cache.clear()


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
