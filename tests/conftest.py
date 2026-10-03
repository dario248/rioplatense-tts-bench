"""Shared fixtures. No GPU, no network, no TTS library: these tests run in CI."""

from __future__ import annotations

import array
import random
import wave
from pathlib import Path

import pytest

from rtb.config import (
    AudioShapeThresholds,
    BenchConfig,
    DurationThresholds,
    FailureThresholds,
    Thresholds,
    load_bench_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_thresholds(**overrides: object) -> Thresholds:
    """A complete Thresholds for tests, so adding a section does not break every fixture."""
    values: dict[str, object] = {
        "version": "test",
        "duration": DurationThresholds(
            too_short_ratio=0.5,
            too_long_ratio=3.0,
            short_utterance_max_words=3,
            short_utterance_allowance_s=0.6,
        ),
        "audio_shape": AudioShapeThresholds(
            frame_ms=20,
            silence_threshold_ratio=0.05,
            silent_peak_ratio=0.01,
        ),
        "failures": FailureThresholds(
            insertions_min=1,
            trailing_deletions_min=2,
            ngram_repeats_min=2,
            tail_after_last_word_s=0.5,
            boundary_max_glued_rate=0.10,
            loop_min_length_ratio=1.3,
        ),
    }
    values.update(overrides)
    return Thresholds.model_validate(values)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The real repository, so tests can validate the committed configs like CI does."""
    return REPO_ROOT


def write_wav(path: Path, seconds: float = 5.0, sample_rate: int = 16000) -> Path:
    """Write a short mono noise wav using the standard library only."""
    rng = random.Random(path.name)
    frames = max(1, int(seconds * sample_rate))
    samples = array.array("h", (rng.randint(-1600, 1600) for _ in range(frames)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(samples.tobytes())
    return path


PHRASES_YAML = """
- id: yei_001
  text: "Llovió toda la noche y la calle amaneció inundada."
  category: yeismo_rehilado
  failure_hypothesis: "Realiza [ʝ] en vez de [ʃ]."
  targets:
    - {word: "Llovió", phenomenon: yeismo}
- id: vos_001
  text: "Vos tenés razón, pero no me lo dijiste antes."
  category: voseo
  failure_hypothesis: "Lee las formas voseantes como tuteo."
  targets: []
"""

SPEAKERS_YAML = """
- id: slr61_f_01
  source: slr61
  gender: F
  ref_clips:
    - path: data/refs/f01_ref0.wav
    - path: data/refs/f01_ref1.wav
  control_clips:
    - path: data/refs/f01_ctrl0.wav
    - path: data/refs/f01_ctrl1.wav
"""

GRID_YAML = """
seeds: [11, 22]
ref_clip_indices: [0]
warmup_syntheses: 1
warmup_text: "Probando."
"""

DUMMY_MODEL_YAML = """
name: dummy
adapter: rtb.generate.dummy:DummyAdapter
model_revision: dummy-noise-v1
max_ref_seconds: 4.0
supports_streaming: false
default_sample_rate: 24000
device: cpu
audio_root: audio
manifest_path: data/manifests/dummy.jsonl
params:
  words_per_second: 4.0
"""


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """A throwaway repository with two phrases, one speaker, two seeds: a four-cell grid."""
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "phrases.yaml").write_text(PHRASES_YAML, encoding="utf-8")
    (tmp_path / "configs" / "speakers.yaml").write_text(SPEAKERS_YAML, encoding="utf-8")
    (tmp_path / "configs" / "grid.yaml").write_text(GRID_YAML, encoding="utf-8")
    (tmp_path / "configs" / "models" / "dummy.yaml").write_text(
        DUMMY_MODEL_YAML, encoding="utf-8"
    )
    for name in ("f01_ref0", "f01_ref1", "f01_ctrl0", "f01_ctrl1"):
        write_wav(tmp_path / "data" / "refs" / f"{name}.wav", seconds=5.0)
    return tmp_path


@pytest.fixture
def mini_config(mini_repo: Path) -> BenchConfig:
    return load_bench_config(
        "configs/models/dummy.yaml",
        root=mini_repo,
        check_paths=True,
    )
