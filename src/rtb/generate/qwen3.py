"""Qwen3-TTS 12Hz Base (QwenLM, Apache-2.0), zero-shot, in-process.

Clones from an (audio, text) pair, so the reference is passed WHOLE and never trimmed: a misaligned
pair raises nothing and clones worse (SPEC 6.4). The config sets `non_streaming_mode: true` against
the library default, which simulates streaming text input — see `envs/qwen3_tts.txt`.

Signatures verified with `inspect.signature` on the installed package, not read off the README.
Heavy imports live inside `load()`; `tests/test_core_isolation.py` fails if that stops being true.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from rtb.config import ModelConfig
from rtb.generate.audio_io import as_float_list, read_duration, write_wav
from rtb.generate.base import SynthesisResult, TTSAdapter

logger = logging.getLogger(__name__)

DEFAULT_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# Nombres de repo que NO son la linea abierta. Confundirlos invalida el benchmark: el flagship
# comercial no tiene pesos, asi que un config que los nombre no puede haber corrido de verdad.
_HOSTED_ONLY = ("qwen3-tts-flash", "qwen-audio-3.0-tts")


class Qwen3Adapter(TTSAdapter):
    """Drives one Qwen3-TTS Base checkpoint over the benchmark grid."""

    name = "qwen3_tts"
    max_ref_seconds = 12.0
    supports_streaming = False
    default_sample_rate = 24000
    requires_ref_text = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        params = dict(config.params)

        # Explicito, nunca "Auto": el idioma es la variable central de este benchmark y dejarselo
        # adivinar al modelo la vuelve una variable no controlada.
        self.language: str = str(params.pop("language", "Spanish"))
        self.x_vector_only_mode: bool = bool(params.pop("x_vector_only_mode", False))
        self.non_streaming_mode: bool = bool(params.pop("non_streaming_mode", True))
        self.dtype: str = str(params.pop("dtype", "bfloat16"))
        self.attn_implementation: str | None = params.pop("attn_implementation", None)
        self.max_new_tokens: int = int(params.pop("max_new_tokens", 2048))
        self.checkpoint_dir: str | None = params.pop("checkpoint_dir", None)
        # Todo lo que sobra se reenvia a `generate()`, que acepta los kwargs de Transformers. Se
        # guarda aparte para que quede claro que no lo interpreta el adapter.
        self.generate_kwargs: dict[str, Any] = params

        if self.x_vector_only_mode:
            # No es un error, pero cambia lo que se esta midiendo: sin ref_text el modelo clona
            # solo del speaker embedding y su propia doc avisa que la calidad baja.
            logger.warning(
                "x_vector_only_mode is on: cloning from the speaker embedding alone, which the "
                "vendor documents as lower quality. ref_text will be ignored and "
                "ref_text_provided will be False"
            )

        repo = (config.hf_repo or DEFAULT_REPO).lower()
        if any(name in repo for name in _HOSTED_ONLY):
            raise ValueError(
                f"{config.hf_repo} is a hosted-only Qwen TTS product with no published weights. "
                f"The open 12Hz line is {DEFAULT_REPO} — see envs/qwen3_tts.txt"
            )

        self._model: Any = None
        self._prompts: dict[tuple[str, str, bool], Any] = {}
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

        The snapshot directory is named after the commit, so provenance comes off the disk and the
        run stays pinned to one revision instead of to `main` (SPEC 5.3).
        """
        if self.checkpoint_dir:
            return self.checkpoint_dir
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        local = Path(snapshot_download(self.repo, revision=self.config.model_revision))
        self._resolved_revision = local.name
        return str(local)

    def _load_model(self) -> Any:
        """Build the model. Overridden in tests, which is why nothing else here touches torch."""
        import torch  # noqa: PLC0415 — import pesado: va acá, no a nivel de modulo
        from qwen_tts import Qwen3TTSModel  # noqa: PLC0415

        self._torch_version = torch.__version__
        self._lib_version = self._detect_lib_version()

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            # No es un fallback silencioso: cambia la latencia por un factor de decenas y esos
            # numeros no son comparables con los de GPU.
            logger.warning("cuda requested but not available; falling back to cpu")
            self._device = "cpu"
        if self._device.startswith("cuda"):
            self._gpu = torch.cuda.get_device_name(0)

        kwargs: dict[str, Any] = {
            "device_map": self._device,
            "dtype": getattr(torch, self.dtype),
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        return Qwen3TTSModel.from_pretrained(self._resolve_checkpoint_dir(), **kwargs)

    def _detect_lib_version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            return f"qwen-tts=={version('qwen-tts')}"
        except PackageNotFoundError:
            return "qwen-tts==unknown"

    def unload(self) -> None:
        self._model = None
        self._prompts.clear()
        self._loaded = False

    # ── sintesis ─────────────────────────────────────────────────────────────────────────────

    def _clone_prompt(self, ref_wav: Path, ref_text: str) -> Any:
        """The cloning prompt for one reference: computed once, cached, and outside the clock.

        The cost is logged separately rather than charged to whichever synthesis ran first
        (SPEC 6.3).
        """
        key = (str(ref_wav), ref_text, self.x_vector_only_mode)
        if key not in self._prompts:
            started = time.perf_counter()
            self._prompts[key] = self._model.create_voice_clone_prompt(
                ref_audio=str(ref_wav),
                # En modo x_vector_only la libreria ignora el texto; se pasa None para que la fila
                # del manifest y lo que recibio el modelo digan lo mismo.
                ref_text=None if self.x_vector_only_mode else ref_text,
                x_vector_only_mode=self.x_vector_only_mode,
            )
            logger.info(
                "voice clone prompt built",
                extra={
                    "fields": {
                        "ref_wav": str(ref_wav),
                        "ref_seconds": round(read_duration(ref_wav), 3),
                        "ms": round((time.perf_counter() - started) * 1000.0, 1),
                    }
                },
            )
        return self._prompts[key]

    def synthesize(
        self,
        text: str,
        ref_wav: Path,
        seed: int,
        out_path: Path,
        ref_text: str | None = None,
        speaker_id: str | None = None,
    ) -> SynthesisResult:
        # Clona del clip: la voz no se elige por id.
        del speaker_id
        if self._model is None:
            raise RuntimeError(f"{self.name}: load() was not called")
        if not self.x_vector_only_mode and not (ref_text or "").strip():
            # El runner ya lo valida al cargar el config; esto cubre a quien llame al adapter
            # directo. Mejor un error que un clonado peor sin aviso.
            raise ValueError(
                f"{self.name}: ICL mode needs the reference transcript, but ref_text is empty. "
                "Populate `transcript` on the ref_clips in configs/speakers.yaml"
            )

        # La referencia entra COMPLETA y a proposito: ver el docstring del modulo.
        # `ref_seconds_used` reporta el clip entero, no `max_ref_seconds`, porque es lo que el
        # modelo realmente uso.
        ref_seconds = read_duration(ref_wav)
        if ref_seconds > self.max_ref_seconds:
            # No recorta —romperia el par (audio, texto)— pero avisa, porque significa que el
            # config declara un presupuesto que la corrida no respeta.
            logger.warning(
                "reference clip %s is %.2fs, above max_ref_seconds=%.2f; passing it whole because "
                "trimming would misalign it from ref_text",
                ref_wav.name,
                ref_seconds,
                self.max_ref_seconds,
            )
        prompt = self._clone_prompt(ref_wav, ref_text or "")

        try:
            started = time.perf_counter()
            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                language=self.language,
                voice_clone_prompt=prompt,
                non_streaming_mode=self.non_streaming_mode,
                max_new_tokens=self.max_new_tokens,
                **self.generate_kwargs,
            )
            total_ms = (time.perf_counter() - started) * 1000.0
        except Exception as err:  # noqa: BLE001 — un modelo que explota es un dato, no un abort
            logger.exception("qwen3 synthesis failed", extra={"fields": {"text": text[:60]}})
            return SynthesisResult(error=f"{type(err).__name__}: {err}")

        if not wavs:
            return SynthesisResult(error="generate_voice_clone returned no audio")

        # El sample rate sale del retorno, no del config: asumirlo es como se generan wavs con el
        # header equivocado, que despues la etapa de metricas lee como otra duracion.
        samples = as_float_list(wavs[0])
        if not samples:
            return SynthesisResult(error="generate_voice_clone returned an empty waveform")
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
            ref_text_provided=not self.x_vector_only_mode,
            streaming=False,
        )

    def describe(self) -> dict[str, str | None]:
        return {
            "model_revision": self._resolved_revision,
            "lib_version": self._lib_version,
            "torch_version": self._torch_version,
            "device": self._device,
            "gpu": self._gpu,
        }
