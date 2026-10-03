# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Write noise wavs wherever `configs/speakers.yaml` expects a clip that is not there yet.

Placeholders, not data: they exercise the pipeline with the dummy model before SLR61 is
downloaded, and are useless for any metric. Existing files are never overwritten.

Depends on pyyaml alone and does not import `rtb`: it has to run in a bare environment.
"""

from __future__ import annotations

import argparse
import array
import random
import sys
import wave
from pathlib import Path

import yaml

SAMPLE_RATE = 16000


def write_noise(path: Path, seconds: float, *, amplitude: float = 0.02) -> None:
    rng = random.Random(path.name)
    peak = int(amplitude * 32767)
    frames = int(seconds * SAMPLE_RATE)
    samples = array.array("h", (rng.randint(-peak, peak) for _ in range(frames)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(samples.tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speakers", type=Path, default=Path("configs/speakers.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args(argv)

    speakers = yaml.safe_load(args.speakers.read_text(encoding="utf-8"))
    created = 0
    for speaker in speakers:
        clips = [
            *speaker.get("ref_clips", []),
            *speaker.get("control_clips", []),
            *speaker.get("calibration_clips", []),
        ]
        for clip in clips:
            path = args.root / clip["path"]
            if path.exists():
                continue
            write_noise(path, clip.get("duration_s") or args.seconds)
            created += 1
            print(f"created {path}")

    print(f"{created} placeholder clip(s) created; these are noise, not speech")
    return 0


if __name__ == "__main__":
    sys.exit(main())
