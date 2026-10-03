"""Chatterbox Multilingual v3 (Resemble AI), zero-shot, in-process.

Clones from audio alone but consumes the reference at three different lengths in one call, so the
adapter trims the wav before handing it over and all three paths see the same seconds (SPEC 6.4).
The library watermarks its own output and that is left on — see `envs/chatterbox_v3.txt`.

Signatures verified with `inspect.signature` on the installed package, not read off the README.
Heavy imports live inside `load()`; `tests/test_core_isolation.py` fails if that stops being true.
"""

from __future__ import annotations

import logging
import time
import wave
from pathlib import Path
from typing import Any

from rtb.config import ModelConfig
from rtb.generate.audio_io import as_float_list, read_duration, write_wav
from rtb.generate.base import SynthesisResult, TTSAdapter
from rtb.logconf import event

logger = logging.getLogger(__name__)

DEFAULT_REPO = "ResembleAI/chatterbox"
# Los pesos que `from_pretrained` de la libreria pide. Se replica la lista para poder resolver el
# commit por nuestra cuenta sin bajar el repo entero.
CHECKPOINT_PATTERNS = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)
# `ChatterboxTTS` es la variante solo-ingles: nombrarla acá seria medir otro modelo.
ENGLISH_ONLY_CLASS = "ChatterboxTTS"
# Parametros de `generate`, con los defaults del vendor. `exaggeration` va aparte porque tambien
# entra en el conditioning.
_GENERATE_PARAMS = ("cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p")


class ChatterboxAdapter(TTSAdapter):
    """Drives one Chatterbox Multilingual checkpoint over the benchmark grid."""

    name = "chatterbox_v3"
    max_ref_seconds = 6.0
    supports_streaming = False
    default_sample_rate = 24000
    requires_ref_text = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        params = dict(config.params)

        self.language_id: str = str(params.pop("language_id", "es"))
        # Entra en el conditioning ademas de en generate, asi que es parte de la clave del cache.
        self.exaggeration: float = float(params.pop("exaggeration", 0.5))
        self.sampling: dict[str, Any] = {
            name: params.pop(name) for name in _GENERATE_PARAMS if name in params
        }
        self.checkpoint_dir: str | None = params.pop("checkpoint_dir", None)
        # Donde van los clips de referencia recortados. Bajo audio_root porque es gitignoreado y
        # queda a mano para escucharlos: un recorte mal hecho se oye.
        self.trimmed_ref_dir: str = str(params.pop("trimmed_ref_dir", "_trimmed_refs"))
        # Mitigador candidato para la deriva de RTF de ANALYSIS A14. Default False: la corrida
        # publicada mide el modelo como es, no con una intervencion nuestra adentro.
        self.empty_cache_between: bool = bool(params.pop("empty_cache_between", False))

        if params:
            # Un parametro mal escrito se ignoraria en silencio y la corrida mediria otra cosa.
            raise ValueError(f"unknown params for {self.name}: {', '.join(sorted(params))}")

        self._model: Any = None
        self._conditioned: tuple[str, float] | None = None
        self._trimmed: dict[tuple[str, float], tuple[Path, float]] = {}
        self._resolved_revision: str | None = config.model_revision
        self._lib_version: str | None = None
        self._torch_version: str | None = None
        self._device: str = config.device
        self._gpu: str | None = None

    @property
    def repo(self) -> str:
        return self.config.hf_repo or DEFAULT_REPO

    # ── carga ────────────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        self._model = self._load_model()
        self._loaded = True

    def _resolve_checkpoint_dir(self) -> str:
        """Download the snapshot and keep the commit hash.

        Not `from_pretrained`, which hardcodes `revision="main"`. The snapshot directory is named
        after the commit, so provenance comes off the disk (SPEC 5.3).
        """
        if self.checkpoint_dir:
            return self.checkpoint_dir
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        local = Path(
            snapshot_download(
                self.repo,
                repo_type="model",
                revision=self.config.model_revision,
                allow_patterns=list(CHECKPOINT_PATTERNS),
            )
        )
        self._resolved_revision = local.name
        return str(local)

    def _load_model(self) -> Any:
        """Build the model. Overridden in tests, which is why nothing else here touches torch."""
        import torch  # noqa: PLC0415 — import pesado: va acá, no a nivel de modulo
        from chatterbox import ChatterboxMultilingualTTS  # noqa: PLC0415

        self._torch_version = torch.__version__
        self._lib_version = self._detect_lib_version()

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            # Correr en CPU no es un fallback silencioso: cambia la latencia por decenas y esos
            # numeros no son comparables con los de GPU.
            logger.warning("cuda requested but not available; falling back to cpu")
            self._device = "cpu"
        if self._device.startswith("cuda"):
            self._gpu = torch.cuda.get_device_name(0)

        return ChatterboxMultilingualTTS.from_local(
            self._resolve_checkpoint_dir(), self._device
        )

    def _detect_lib_version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            return f"chatterbox-tts=={version('chatterbox-tts')}"
        except PackageNotFoundError:
            return "chatterbox-tts==unknown"

    def unload(self) -> None:
        self._model = None
        self._conditioned = None
        self._trimmed.clear()
        self._loaded = False

    # ── la referencia recortada ──────────────────────────────────────────────────────────────

    def trimmed_reference(self, ref_wav: Path) -> tuple[Path, float]:
        """A copy of the reference cut to `max_ref_seconds`, written once and cached.

        Trimmed here because the model trims inconsistently across its three paths (SPEC 6.4). Cut
        from the START, whole samples, no resampling and no level change.
        """
        key = (str(ref_wav), float(self.max_ref_seconds))
        cached = self._trimmed.get(key)
        if cached is not None:
            return cached

        full_seconds = read_duration(ref_wav)
        if full_seconds <= self.max_ref_seconds:
            # Ya entra en el presupuesto: se pasa tal cual, sin escribir una copia identica.
            result = (ref_wav, full_seconds)
            self._trimmed[key] = result
            return result

        with wave.open(str(ref_wav), "rb") as source:
            channels, width, rate = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            )
            keep = min(source.getnframes(), int(self.max_ref_seconds * rate))
            frames = source.readframes(keep)

        target = self._trimmed_path(ref_wav)
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
        self._trimmed[key] = result
        return result

    def _trimmed_path(self, ref_wav: Path) -> Path:
        """Where the trimmed copy goes. Named after the budget, so two budgets never collide."""
        root = self.config.audio_root / self.name / self.trimmed_ref_dir
        budget = f"{self.max_ref_seconds:g}".replace(".", "_")
        return root / f"{ref_wav.parent.name}__{ref_wav.stem}__{budget}s.wav"

    def _condition(self, ref_wav: Path) -> None:
        """Cloning conditionals for one reference: computed once, cached, and outside the clock.

        The first call costs ~11 s of lazy initialization and is paid by a warm-up (SPEC 6.3).
        """
        key = (str(ref_wav), self.exaggeration)
        if self._conditioned == key:
            return
        started = time.perf_counter()
        self._model.prepare_conditionals(str(ref_wav), exaggeration=self.exaggeration)
        self._conditioned = key
        logger.info(
            "conditionals prepared",
            extra={
                "fields": {
                    "ref_wav": str(ref_wav),
                    "exaggeration": self.exaggeration,
                    "ms": round((time.perf_counter() - started) * 1000.0, 1),
                }
            },
        )

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
        # Chatterbox clona SOLO desde audio: `prepare_conditionals` no acepta texto. Esa asimetria
        # contra Qwen3-TTS queda registrada por fila en `ref_text_provided` (SPEC 6.4).
        del ref_text, speaker_id
        if self._model is None:
            raise RuntimeError(f"{self.name}: load() was not called")

        trimmed, ref_seconds = self.trimmed_reference(ref_wav)
        self._condition(trimmed)

        try:
            started = time.perf_counter()
            wav = self._model.generate(
                text,
                language_id=self.language_id,
                exaggeration=self.exaggeration,
                **self.sampling,
            )
            total_ms = (time.perf_counter() - started) * 1000.0
        except Exception as err:  # noqa: BLE001 — un modelo que explota es un dato, no un abort
            logger.exception("chatterbox synthesis failed", extra={"fields": {"text": text[:60]}})
            return SynthesisResult(error=f"{type(err).__name__}: {err}")

        self._after_synthesis()

        samples = as_float_list(wav)
        if not samples:
            return SynthesisResult(error="generate returned an empty waveform")

        sample_rate = int(getattr(self._model, "sr", self.default_sample_rate))
        write_wav(out_path, samples, sample_rate)
        if sample_rate != self.default_sample_rate:
            logger.warning(
                "model returned %d Hz but the config declares %d; the manifest records the real "
                "one, and the metrics stage resamples",
                sample_rate,
                self.default_sample_rate,
            )

        return SynthesisResult(
            sample_rate=sample_rate,
            duration_s=len(samples) / float(sample_rate),
            # Sin streaming de generacion el primer audio existe cuando termina todo, asi que
            # ttfb == total. Es el dato correcto, no un placeholder.
            ttfb_ms=total_ms,
            total_ms=total_ms,
            ref_seconds_used=ref_seconds,
            ref_text_provided=False,
            streaming=False,
        )

    def _after_synthesis(self) -> None:
        """VRAM del proceso, y el vaciado opcional del cache. Siempre despues del reloj (SPEC 6.3).

        La deriva de A14 se ve como crecimiento del allocator entre sintesis, asi que sin
        registrarlo por fila no hay forma de saber si un mitigador funciono.
        """
        if not self._device.startswith("cuda"):
            return
        import torch  # noqa: PLC0415 — import pesado: va acá, no a nivel de modulo

        event(
            logger,
            "vram_after_synthesis",
            allocated_mib=round(torch.cuda.memory_allocated() / 2**20, 1),
            reserved_mib=round(torch.cuda.memory_reserved() / 2**20, 1),
        )
        if self.empty_cache_between:
            torch.cuda.empty_cache()

    def describe(self) -> dict[str, str | None]:
        return {
            "model_revision": self._resolved_revision,
            "lib_version": self._lib_version,
            "torch_version": self._torch_version,
            "device": self._device,
            "gpu": self._gpu,
        }
