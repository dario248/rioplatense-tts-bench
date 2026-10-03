"""Audio for the videos: one loudness for every clip, and nothing else done to it.

Normalization is the only processing. No denoising, no EQ, no compression, and above all no
trimming of the tail — in post 2 the tail *is* the finding, and the loudest moment of that XTTS
clip lives inside it.

Loudness is measured with `ebur128` and applied as a plain gain, not with `loudnorm`. Two
reasons, both load-bearing. `loudnorm` returns a broken `target_offset` when it measures an LRA
of 0, which it does on clips this short — it sent a 2.2 s clip to −62 LUFS against a −16 target.
And its dynamic mode is a compressor, which would reshape exactly the loud tail these videos
exist to show.

The measurement pads the clip with silence first: under about 3 s the R128 integration has too
few blocks to be stable, and the gate drops the padding, so the number is the one the speech
deserves.

The target is per video, not fixed: a clip already peaking near full scale cannot be lifted to
−16 LUFS without clipping, so the renderer picks the loudest level ALL clips can reach under the
true-peak ceiling and puts every one of them there. Equal loudness is the requirement — no model
may sound better for being hot — and the absolute number is reported with the video.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48_000
CHANNELS = 2
TARGET_I = -16.0
TARGET_TP = -1.5
MEASURE_PAD_S = 5.0


class AudioError(Exception):
    """ffmpeg could not do what the renderer asked."""


def require_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise AudioError(f"{binary} is not on PATH; it is a system dependency, not a wheel")


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"{' '.join(args[:3])}… failed:\n{result.stderr[-2000:]}")
    return result


def probe_duration(path: Path) -> float:
    result = run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nk=1:nw=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


@dataclass(frozen=True)
class Measurement:
    """What one clip measures before anything is done to it."""

    path: Path
    integrated: float  # LUFS
    true_peak: float  # dBTP

    @property
    def headroom(self) -> float:
        """How much gain this clip can take before it breaks the true-peak ceiling."""
        return TARGET_TP - self.true_peak

    @property
    def reachable(self) -> float:
        """The loudest integrated level this clip can reach without clipping."""
        return self.integrated + self.headroom


_I_RE = re.compile(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS")
_PEAK_RE = re.compile(r"Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS")


def measure(path: Path) -> Measurement:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
            "-af", f"apad=pad_dur={MEASURE_PAD_S},ebur128=peak=true",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioError(f"{path}: ebur128 failed\n{result.stderr[-2000:]}")
    summary = result.stderr[result.stderr.rfind("Summary:") :]
    integrated = _I_RE.search(summary)
    peak = _PEAK_RE.search(summary)
    if not integrated or not peak:
        raise AudioError(f"{path}: ebur128 printed no summary\n{result.stderr[-2000:]}")
    return Measurement(path, float(integrated.group(1)), float(peak.group(1)))


def common_target(measurements: list[Measurement]) -> float:
    """The loudest level every clip in one video can reach under the true-peak ceiling."""
    if not measurements:
        return TARGET_I
    return min([TARGET_I, *(m.reachable for m in measurements)])


@dataclass(frozen=True)
class Loudness:
    """What normalizing one clip actually did, for the sources file next to the video."""

    source: Path
    input_i: float
    input_tp: float
    gain_db: float
    output_i: float
    output_tp: float


def apply_gain(measurement: Measurement, dest: Path, target: float) -> Loudness:
    """One linear gain, then 48 kHz stereo. Nothing is compressed, limited or trimmed."""
    gain = target - measurement.integrated
    if measurement.true_peak + gain > TARGET_TP + 1e-9:
        raise AudioError(
            f"{measurement.path}: a gain of {gain:+.2f} dB would peak at "
            f"{measurement.true_peak + gain:.2f} dBTP, over the {TARGET_TP} dBTP ceiling"
        )
    run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(measurement.path),
            "-af", f"volume={gain:.4f}dB",
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", "pcm_s16le",
            str(dest),
        ]
    )
    verified = measure(dest)
    return Loudness(
        source=measurement.path,
        input_i=measurement.integrated,
        input_tp=measurement.true_peak,
        gain_db=gain,
        output_i=verified.integrated,
        output_tp=verified.true_peak,
    )


def read(path: Path) -> np.ndarray:
    """A normalized clip as float32 (frames, 2) at 48 kHz."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise AudioError(f"{path}: {rate} Hz, expected {SAMPLE_RATE}")
    if data.shape[1] == 1:
        data = np.repeat(data, CHANNELS, axis=1)
    return data[:, :CHANNELS]


def silence(seconds: float) -> np.ndarray:
    return np.zeros((int(round(seconds * SAMPLE_RATE)), CHANNELS), dtype=np.float32)


def write(path: Path, blocks: list[np.ndarray]) -> float:
    data = np.concatenate(blocks) if blocks else silence(0.0)
    sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")
    return len(data) / SAMPLE_RATE
