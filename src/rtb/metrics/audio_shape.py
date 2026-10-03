"""Energy envelope: where the sound in a clip starts and stops (SPEC 7.5).

Separates speech from padding. Stdlib only, so it runs in any environment, CI included.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

from rtb.config import AudioShapeThresholds
from rtb.metrics.audio_utils import FULL_SCALE, AudioError, read_mono_pcm
from rtb.schema import MetricRow

METRIC_NAME = "audio_shape"


@dataclass
class AudioShape:
    """The measured anatomy of one clip, in seconds."""

    total_s: float
    speech_start_s: float
    speech_end_s: float
    peak: float
    silent: bool

    @property
    def speech_s(self) -> float:
        # Incluye la cola posterior a la ultima palabra: el habla pedida la mide `anatomy`.
        return max(0.0, self.speech_end_s - self.speech_start_s)

    @property
    def lead_silence_s(self) -> float:
        return self.speech_start_s

    @property
    def tail_silence_s(self) -> float:
        return max(0.0, self.total_s - self.speech_end_s)

    @property
    def padding_s(self) -> float:
        return self.lead_silence_s + self.tail_silence_s


def rms_envelope(path: Path, frame_ms: int) -> tuple[list[float], float]:
    """RMS per fixed frame, normalized to full scale, plus the frame length in seconds.

    Shared with the listening worklist so a tail is not described by one criterion and judged by
    another.
    """
    samples, rate = read_mono_pcm(path)
    frame_len = max(1, int(rate * frame_ms / 1000))
    rms = [
        math.sqrt(sum(value * value for value in samples[start : start + frame_len]) / frame_len)
        / FULL_SCALE
        for start in range(0, len(samples) - frame_len + 1, frame_len)
    ]
    return rms, frame_len / float(rate)


def measure(path: Path, thresholds: AudioShapeThresholds) -> AudioShape:
    """RMS envelope in fixed frames, then the first and last frame carrying sound."""
    samples, rate = read_mono_pcm(path)
    total_s = len(samples) / float(rate)
    rms, frame_s = rms_envelope(path, thresholds.frame_ms)

    peak = max(rms, default=0.0)
    if peak < thresholds.silent_peak_ratio:
        # Un umbral relativo sobre un archivo mudo encontraria "habla" en el ruido de fondo, y un
        # modelo que devuelve silencio pasaria por bueno.
        return AudioShape(total_s, 0.0, 0.0, peak, silent=True)

    floor = peak * thresholds.silence_threshold_ratio
    loud = [index for index, value in enumerate(rms) if value > floor]
    return AudioShape(
        total_s=total_s,
        speech_start_s=loud[0] * frame_s,
        speech_end_s=(loud[-1] + 1) * frame_s,
        peak=peak,
        silent=False,
    )


def shape_row(
    generation_id: str,
    wav_path: Path,
    thresholds: AudioShapeThresholds,
    version: str,
) -> MetricRow:
    """One row per clip. Measures, never judges — the thresholds for failure live elsewhere."""
    try:
        shape = measure(wav_path, thresholds)
    except (AudioError, OSError, wave.Error) as err:
        return MetricRow(
            generation_id=generation_id,
            metric=METRIC_NAME,
            metric_version=version,
            error=f"{type(err).__name__}: {err}",
        )
    return MetricRow(
        generation_id=generation_id,
        metric=METRIC_NAME,
        metric_version=version,
        values={
            "total_s": round(shape.total_s, 3),
            "speech_start_s": round(shape.speech_start_s, 3),
            "speech_end_s": round(shape.speech_end_s, 3),
            "speech_s": round(shape.speech_s, 3),
            "lead_silence_s": round(shape.lead_silence_s, 3),
            "tail_silence_s": round(shape.tail_silence_s, 3),
            "padding_s": round(shape.padding_s, 3),
            "peak": round(shape.peak, 4),
            "silent": shape.silent,
        },
    )
