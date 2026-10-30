"""Pocket TTS (Kyutai), zero-shot, in-process, on CPU.

Clones from audio alone, with no transcript, and reads the reference whole, so the adapter cuts it
to `max_ref_seconds` first and every zero-shot model gets the same budget (SPEC 6.4). The voice
state is the model's KV cache after listening to that clip: computed once per clip, cached, and
kept outside the clock (SPEC 6.3).

The cloning weights are gated. When the library cannot download them it loads a non-cloning
checkpoint instead and only fails later, at the first clip, so `load()` refuses that fallback
rather than letting a run die one row at a time.

Signatures verified with `inspect.signature` on the installed package, not read off the README.
Heavy imports live inside `load()`; `tests/test_core_isolation.py` fails if that stops being true.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from rtb.config import ModelConfig
from rtb.generate.audio_io import ReferenceTrimmer, as_float_list, write_wav
from rtb.generate.base import AdapterError, SynthesisResult, TTSAdapter

logger = logging.getLogger(__name__)

# Las dos variantes de español publicadas: 6 capas y 24 capas sobre el mismo d_model. Se nombran
# acá para que un typo en el YAML falle al construir y no después de bajar los pesos.
SPANISH_LANGUAGES = ("spanish", "spanish_24l")


def _samples_of(audio: Any) -> list[float]:
    """Flatten what the model returned, taking it off the GPU first when it is there.

    `generate_audio` returns a tensor on the model's device, and the benchmark writes wavs with
    the standard library.
    """
    detach = getattr(audio, "detach", None)
    if detach is not None:
        audio = detach().cpu()
    return as_float_list(audio)


class PocketAdapter(TTSAdapter):
    """Drives one Pocket TTS language checkpoint over the benchmark grid."""

    name = "pocket_tts_es"
    max_ref_seconds = 6.0
    supports_streaming = True
    default_sample_rate = 24000
    requires_ref_text = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        params = dict(config.params)

        self.language: str = str(params.pop("language", "spanish"))
        if self.language not in SPANISH_LANGUAGES:
            raise ValueError(
                f"{self.name}: language must be one of {', '.join(SPANISH_LANGUAGES)}, "
                f"got {self.language!r}"
            )

        # None deja el default del checkpoint. Se registra el valor efectivo al cargar, porque es
        # lo que gobierna el muestreo y por lo tanto el audio que se mide.
        temp = params.pop("temp", None)
        self.temp: float | None = None if temp is None else float(temp)
        self.sampler_decode_steps: int = int(params.pop("sampler_decode_steps", 1))
        # El texto se parte en chunks de a lo sumo estos tokens. Es segmentación del modelo, igual
        # que la de Sintética (SPEC 6.5), y se mide en el default del vendor.
        self.max_tokens: int = int(params.pop("max_tokens", 50))
        self.quantize: bool = bool(params.pop("quantize", False))

        self.streaming: bool = bool(params.pop("streaming", False))
        self.trimmed_ref_dir: str = str(params.pop("trimmed_ref_dir", "_trimmed_refs"))

        if params:
            # Un parametro mal escrito se ignoraria en silencio y la corrida mediria otra cosa.
            raise ValueError(f"unknown params for {self.name}: {', '.join(sorted(params))}")

        if self.quantize and config.device.startswith("cuda"):
            # La cuantizacion int8 es solo-CPU y revienta recien al sintetizar, con un
            # NotImplementedError del backend que no nombra ni a `quantize` ni al device.
            raise ValueError(f"{self.name}: quantize is CPU-only, but device is {config.device}")

        self._model: Any = None
        self._states: dict[str, Any] = {}
        self._trimmer = ReferenceTrimmer(
            config.audio_root / self.name / self.trimmed_ref_dir, self.max_ref_seconds
        )
        self._resolved_revision: str | None = config.model_revision
        self._lib_version: str | None = None
        self._torch_version: str | None = None
        self._device: str = config.device
        self._gpu: str | None = None
        self._sample_rate: int = config.default_sample_rate

    # ── carga ────────────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        model = self._load_model()
        # Fuera de `_load_model`, que es lo unico que los tests reemplazan: todo lo que puede
        # invalidar una corrida entera se chequea sin la libreria instalada.
        self._check_cloning_weights(model)
        self._resolved_revision = self._revision_of(model)
        self._resolve_sample_rate(model)
        self._log_effective_config(model)
        self._model = model
        self._loaded = True

    def _load_model(self) -> Any:
        """Build the model. Overridden in tests, which is why nothing else here touches torch."""
        import torch  # noqa: PLC0415 — import pesado: va acá, no a nivel de modulo
        from pocket_tts import TTSModel  # noqa: PLC0415

        self._torch_version = torch.__version__
        self._lib_version = self._detect_lib_version()

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            # Irse a CPU en silencio cambia la latencia por decenas y el audio por completo: el
            # muestreo toma numeros de un generador distinto en cada device.
            logger.warning("cuda requested but not available; falling back to cpu")
            self._device = "cpu"
        if self._device.startswith("cuda"):
            self._gpu = torch.cuda.get_device_name(0)

        model = TTSModel.load_model(
            language=self.language,
            temp=self.temp,
            sampler_decode_steps=self.sampler_decode_steps,
            quantize=self.quantize,
        )
        # `load_model` devuelve siempre en CPU; el device lo pone el llamador, igual que el CLI.
        model.to(self._device)
        return model

    def _detect_lib_version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            return f"pocket-tts=={version('pocket-tts')}"
        except PackageNotFoundError:
            return "pocket-tts==unknown"

    def _check_cloning_weights(self, model: Any) -> None:
        """Refuse the non-cloning fallback.

        The language config carries a second weights path for the ungated repository and the
        library swaps to it on any download failure, including a missing gate acceptance. Those
        weights cannot clone, so the whole run measures nothing.
        """
        if getattr(model, "has_voice_cloning", True):
            return
        raise AdapterError(
            f"{self.name}: loaded the weights WITHOUT voice cloning, because the gated "
            "repository could not be downloaded. Accept the terms at "
            "https://huggingface.co/kyutai/pocket-tts and log in locally. Refusing to run: "
            "these weights cannot clone from a reference clip"
        )

    def _revision_of(self, model: Any) -> str | None:
        """The weights commit, which the language config pins as an `hf://…@<revision>` suffix."""
        declared = str(getattr(model.config, "weights_path", "") or "")
        _, _, suffix = declared.partition("@")
        revision: str | None = suffix or None
        if revision and self.config.model_revision and revision != self.config.model_revision:
            # El pin viaja adentro del paquete, asi que subir la libreria puede cambiar los pesos
            # sin que el YAML del modelo se entere.
            logger.warning(
                "the installed library pins weights at %s but the model config declares %s; the "
                "manifest records what ran",
                revision,
                self.config.model_revision,
            )
        return revision or self.config.model_revision

    def _resolve_sample_rate(self, model: Any) -> None:
        self._sample_rate = int(getattr(model, "sample_rate", self.default_sample_rate))
        if self._sample_rate != self.default_sample_rate:
            logger.warning(
                "model returns %d Hz but the config declares %d; the manifest records the real "
                "one, and the metrics stage resamples",
                self._sample_rate,
                self.default_sample_rate,
            )

    def _log_effective_config(self, model: Any) -> None:
        """What actually governs the sampling, resolved after the checkpoint had its say."""
        logger.info(
            "pocket model loaded",
            extra={
                "fields": {
                    "language": self.language,
                    "temp": getattr(model, "temp", None),
                    "sampler_decode_steps": self.sampler_decode_steps,
                    "max_tokens": self.max_tokens,
                    "quantize": self.quantize,
                    "sample_rate": self._sample_rate,
                    "revision": self._resolved_revision,
                }
            },
        )

    def unload(self) -> None:
        self._model = None
        self._states.clear()
        self._trimmer.clear()
        self._loaded = False

    # ── el estado de voz ─────────────────────────────────────────────────────────────────────

    def voice_state(self, ref_wav: Path) -> Any:
        """The model's KV cache after listening to one clip: computed once, cached, off the clock.

        `generate_audio` copies the state before using it, so one cached state serves every row of
        a speaker without carrying anything over between syntheses.
        """
        key = str(ref_wav)
        cached = self._states.get(key)
        if cached is not None:
            return cached

        started = time.perf_counter()
        state = self._model.get_state_for_audio_prompt(ref_wav)
        self._states[key] = state
        logger.info(
            "voice state encoded",
            extra={
                "fields": {
                    "ref_wav": key,
                    "ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "encoded_seconds": self._encoded_seconds(ref_wav),
                }
            },
        )
        return state

    def _encoded_seconds(self, ref_wav: Path) -> float | None:
        """How much of the clip the model really encodes, which is not the whole trimmed copy.

        It cuts the trailing silence, fades 20 ms and appends an 80 ms pause, so the prompt ends
        between words. Diagnostic only: `ref_seconds_used` reports the seconds handed over, which
        is what the same column means for the other models.
        """
        try:
            from pocket_tts.data.audio import audio_read  # noqa: PLC0415
            from pocket_tts.data.audio_utils import (  # noqa: PLC0415
                convert_audio,
                end_on_pause,
            )
        except ImportError:
            return None
        try:
            audio, rate = audio_read(ref_wav)
            audio = convert_audio(audio, rate, self._sample_rate, 1)
            samples = int(end_on_pause(audio, self._sample_rate).shape[-1])
            return round(samples / self._sample_rate, 3)
        except Exception:  # noqa: BLE001 — es una linea de log, no puede tumbar una corrida
            return None

    # ── sintesis ─────────────────────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        ref_wav: Path,
        seed: int,
        out_path: Path,
        ref_text: str | None = None,
        speaker_id: str | None = None,
    ) -> SynthesisResult:
        # Clona SOLO desde audio: no hay forma de pasarle la transcripcion. Esa asimetria contra
        # Qwen3-TTS queda registrada por fila en `ref_text_provided` (SPEC 6.4).
        del ref_text, speaker_id, seed
        if self._model is None:
            raise RuntimeError(f"{self.name}: load() was not called")

        trimmed, ref_seconds = self._trimmer.trimmed(ref_wav)
        state = self.voice_state(trimmed)

        try:
            if self.streaming:
                samples, ttfb_ms, total_ms = self._synthesize_streaming(state, text)
            else:
                samples, ttfb_ms, total_ms = self._synthesize_whole(state, text)
        except Exception as err:  # noqa: BLE001 — un modelo que explota es un dato, no un abort
            logger.exception("pocket synthesis failed", extra={"fields": {"text": text[:60]}})
            return SynthesisResult(error=f"{type(err).__name__}: {err}")

        if not samples:
            return SynthesisResult(error="generate returned an empty waveform")

        write_wav(out_path, samples, self._sample_rate)
        return SynthesisResult(
            sample_rate=self._sample_rate,
            duration_s=len(samples) / float(self._sample_rate),
            ttfb_ms=ttfb_ms,
            total_ms=total_ms,
            ref_seconds_used=ref_seconds,
            ref_text_provided=False,
            streaming=self.streaming,
        )

    def _synthesize_whole(self, state: Any, text: str) -> tuple[list[float], float, float]:
        started = time.perf_counter()
        audio = self._model.generate_audio(state, text, max_tokens=self.max_tokens)
        total_ms = (time.perf_counter() - started) * 1000.0
        # Sin streaming el primer audio recien existe cuando termina todo, asi que ttfb == total.
        return _samples_of(audio), total_ms, total_ms

    def _synthesize_streaming(self, state: Any, text: str) -> tuple[list[float], float, float]:
        started = time.perf_counter()
        ttfb_ms: float | None = None
        samples: list[float] = []
        for chunk in self._model.generate_audio_stream(state, text, max_tokens=self.max_tokens):
            if ttfb_ms is None:
                ttfb_ms = (time.perf_counter() - started) * 1000.0
            samples.extend(_samples_of(chunk))
        total_ms = (time.perf_counter() - started) * 1000.0
        return samples, total_ms if ttfb_ms is None else ttfb_ms, total_ms

    def describe(self) -> dict[str, str | None]:
        return {
            "model_revision": self._resolved_revision,
            "lib_version": self._lib_version,
            "torch_version": self._torch_version,
            "device": self._device,
            "gpu": self._gpu,
        }
