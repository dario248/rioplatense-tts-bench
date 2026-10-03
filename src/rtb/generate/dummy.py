"""A white-noise adapter, so the whole pipeline can be exercised without a GPU.

Writes real wavs and can be told to fail, truncate or loop on demand — the failure modes the
metrics stage has to detect (SPEC 7.5).
"""

from __future__ import annotations

import array
import logging
import random
import time
import wave
from pathlib import Path

from rtb import __version__
from rtb.config import ModelConfig
from rtb.generate.base import SynthesisResult, TTSAdapter
from rtb.schema import count_words

logger = logging.getLogger(__name__)


class DummyAdapter(TTSAdapter):
    """Deterministic noise generator with the timing surface of a real model."""

    name = "dummy"
    max_ref_seconds = 6.0
    supports_streaming = False
    default_sample_rate = 24000

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        params = config.params
        self.words_per_second: float = float(params.get("words_per_second", 2.6))
        self.tail_silence_s: float = float(params.get("tail_silence_s", 0.25))
        self.amplitude: float = float(params.get("amplitude", 0.05))
        self.ttfb_ms: float = float(params.get("ttfb_ms", 0.0))
        self.ms_per_word: float = float(params.get("ms_per_word", 0.0))
        # Disparadores de fallo para los tests del runner y de las metricas de tasa de fallo.
        self.fail_if_text_contains: list[str] = list(params.get("fail_if_text_contains", []))
        self.truncate_if_text_contains: list[str] = list(
            params.get("truncate_if_text_contains", [])
        )
        self.loop_if_text_contains: list[str] = list(params.get("loop_if_text_contains", []))

    def load(self) -> None:
        # Un modelo real importaría torch acá. Este no importa nada: es la prueba de que el
        # runner no depende de ninguna libreria de TTS.
        self._loaded = True

    def describe(self) -> dict[str, str | None]:
        return {
            "model_revision": self.config.model_revision or "dummy-noise-v1",
            "lib_version": f"rtb-dummy=={__version__}",
            "torch_version": None,
            "device": "cpu",
            "gpu": None,
        }

    def synthesize(
        self,
        text: str,
        ref_wav: Path,
        seed: int,
        out_path: Path,
        ref_text: str | None = None,
        speaker_id: str | None = None,
    ) -> SynthesisResult:
        # `ref_text` se ignora a proposito: el doble de prueba genera ruido y no clona nada. Lo
        # acepta para satisfacer la ABC y para que el runner ejercite el paso del parametro.
        del ref_text, speaker_id
        started = time.perf_counter()

        for marker in self.fail_if_text_contains:
            if marker in text:
                # Fallo "blando": el adapter lo reporta en vez de tirar excepcion, para que el
                # runner ejercite la rama de fila con error sin pasar por el except.
                return SynthesisResult(error=f"simulated failure triggered by {marker!r}")

        ref_seconds_used = min(self._read_duration(ref_wav), self.max_ref_seconds)

        duration_s = count_words(text) / self.words_per_second + self.tail_silence_s
        if any(marker in text for marker in self.truncate_if_text_contains):
            duration_s = min(duration_s, 0.35)
        if any(marker in text for marker in self.loop_if_text_contains):
            duration_s *= 4.0

        ttfb_ms = self.ttfb_ms
        if ttfb_ms:
            time.sleep(ttfb_ms / 1000.0)
        first_byte = time.perf_counter()
        if self.ms_per_word:
            time.sleep(count_words(text) * self.ms_per_word / 1000.0)

        self._write_noise(out_path, duration_s, seed=seed, text=text)
        total_ms = (time.perf_counter() - started) * 1000.0

        return SynthesisResult(
            sample_rate=self.default_sample_rate,
            duration_s=duration_s,
            ttfb_ms=(first_byte - started) * 1000.0,
            total_ms=total_ms,
            ref_seconds_used=ref_seconds_used,
        )

    def _read_duration(self, path: Path) -> float:
        """Duration of the reference clip, read from the wav header only."""
        with wave.open(str(path), "rb") as fh:
            frames = fh.getnframes()
            rate = fh.getframerate()
        if rate <= 0 or frames <= 0:
            raise ValueError(f"reference clip has no audio: {path}")
        return frames / float(rate)

    def _write_noise(self, out_path: Path, duration_s: float, *, seed: int, text: str) -> None:
        rate = self.default_sample_rate
        total_frames = max(1, int(round(duration_s * rate)))
        peak = int(max(0.0, min(1.0, self.amplitude)) * 32767)

        # Un bloque de un segundo, repetido: generar 40 M de muestras en Python puro por corrida
        # completa tarda mas que el resto del pipeline junto, y para un doble de prueba la
        # periodicidad no cambia nada.
        rng = random.Random(f"{seed}:{text}")
        block_len = min(total_frames, rate)
        block = array.array("h", (rng.randint(-peak, peak) for _ in range(block_len)))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            remaining = total_frames
            while remaining > 0:
                chunk = block if remaining >= block_len else block[:remaining]
                fh.writeframes(chunk.tobytes())
                remaining -= len(chunk)
