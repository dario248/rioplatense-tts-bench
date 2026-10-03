"""XTTS-v2 (coqui), zero-shot, in-process.

Drives the low-level API instead of the `TTS` facade: the facade hides the conditioning step, and
that step is what the protocol has to control (SPEC 6.4). One number, `max_ref_seconds`, drives
the three conditioning knobs together so the manifest cannot claim 6 s while the model reads 12.

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

# Parametros de muestreo que el checkpoint trae en su config.json. Se declaran igual en el YAML
# del modelo, para que la corrida sea reproducible aunque el checkpoint cambie, y el adapter avisa
# cuando los dos no coinciden: ahi hay que decidir a mano, no dejar que gane uno en silencio.
_SAMPLING_PARAMS = (
    "temperature",
    "length_penalty",
    "repetition_penalty",
    "top_k",
    "top_p",
)


class XttsAdapter(TTSAdapter):
    """Drives one XTTS-v2 checkpoint over the benchmark grid."""

    name = "xtts_v2"
    max_ref_seconds = 6.0
    supports_streaming = True
    default_sample_rate = 24000

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        params = dict(config.params)

        self.language: str = str(params.pop("language", "es"))

        # Presupuesto de referencia: un solo numero manda sobre los tres knobs de conditioning.
        # Van en SEGUNDOS ENTEROS porque XTTS los usa como multiplicadores del sample rate dentro
        # de indices de slice (`audio[:, : 22050 * length]`): un float revienta con TypeError.
        budget = float(self.max_ref_seconds)
        self.gpt_cond_len: int = self._whole_seconds(
            params.pop("gpt_cond_len", budget), "gpt_cond_len"
        )
        self.gpt_cond_chunk_len: int = self._whole_seconds(
            params.pop("gpt_cond_chunk_len", budget), "gpt_cond_chunk_len"
        )
        self.max_ref_length: int = self._whole_seconds(
            params.pop("max_ref_length", budget), "max_ref_length"
        )
        self.sound_norm_refs: bool = bool(params.pop("sound_norm_refs", False))
        self.librosa_trim_db: float | None = params.pop("librosa_trim_db", None)

        self.streaming: bool = bool(params.pop("streaming", False))
        self.stream_chunk_size: int = int(params.pop("stream_chunk_size", 20))
        self.overlap_wav_len: int = int(params.pop("overlap_wav_len", 1024))

        self.checkpoint_dir: str | None = params.pop("checkpoint_dir", None)
        self.use_deepspeed: bool = bool(params.pop("use_deepspeed", False))

        self.do_sample: bool = bool(params.pop("do_sample", True))
        self.num_beams: int = int(params.pop("num_beams", 1))
        self.speed: float = float(params.pop("speed", 1.0))
        self.enable_text_splitting: bool = bool(params.pop("enable_text_splitting", False))
        self.sampling: dict[str, Any] = {
            name: params.pop(name) for name in _SAMPLING_PARAMS if name in params
        }

        if params:
            # Un parametro con el nombre mal escrito se ignoraria en silencio y la corrida
            # mediria otra configuracion que la declarada.
            raise ValueError(f"unknown params for {self.name}: {', '.join(sorted(params))}")

        self._model: Any = None
        self._latents: dict[tuple[str, float, float], tuple[Any, Any]] = {}
        self._resolved_revision: str | None = config.model_revision
        self._lib_version: str | None = None
        self._torch_version: str | None = None
        self._device: str = config.device
        self._gpu: str | None = None

    def _whole_seconds(self, value: float, field: str) -> int:
        """Floor a duration to whole seconds, the only thing XTTS can honor.

        Silently truncating would leave `ref_seconds_used` reporting the budget and not what the
        model consumed.
        """
        seconds = int(value)
        if seconds != value:
            logger.warning(
                "%s=%s is not a whole number of seconds; XTTS slices the reference by whole "
                "seconds, so %s will be used",
                field,
                value,
                seconds,
            )
        if seconds < 1:
            raise ValueError(f"{self.name}: {field} must be at least 1 second, got {value}")
        return seconds

    # ── carga ────────────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        model, model_config = self._load_model()
        self._model = model
        self._check_checkpoint_agrees(model_config)
        self._loaded = True

    def _load_model(self) -> tuple[Any, Any]:
        """Build the model. Overridden in tests, which is why nothing else here touches torch."""
        import torch  # noqa: PLC0415 — import pesado: va acá, no a nivel de modulo
        from TTS.tts.configs.xtts_config import XttsConfig  # noqa: PLC0415
        from TTS.tts.models.xtts import Xtts  # noqa: PLC0415

        self._torch_version = torch.__version__
        self._lib_version = self._detect_lib_version()

        checkpoint_dir = Path(self._resolve_checkpoint_dir())
        model_config = XttsConfig()
        model_config.load_json(str(checkpoint_dir / "config.json"))
        model = Xtts.init_from_config(model_config)
        model.load_checkpoint(
            config=model_config,
            checkpoint_dir=str(checkpoint_dir),
            use_deepspeed=self.use_deepspeed,
            eval=True,
        )

        if self._device.startswith("cuda") and torch.cuda.is_available():
            model.cuda()
            self._gpu = torch.cuda.get_device_name(0)
        else:
            if self._device.startswith("cuda"):
                # Correr en CPU no es un fallback silencioso: cambia la latencia por un factor de
                # decenas, y esos numeros no son comparables con los de GPU.
                logger.warning("cuda requested but not available; falling back to cpu")
            self._device = "cpu"
        return model, model_config

    def _detect_lib_version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        for package in ("coqui-tts", "TTS"):
            try:
                return f"{package}=={version(package)}"
            except PackageNotFoundError:
                continue
        return "coqui-tts==unknown"

    def _resolve_checkpoint_dir(self) -> str:
        """Local directory when given, otherwise the pinned HuggingFace snapshot."""
        if self.checkpoint_dir:
            return self.checkpoint_dir
        if not self.config.hf_repo:
            raise ValueError(f"{self.name}: set either params.checkpoint_dir or hf_repo")

        from huggingface_hub import snapshot_download  # noqa: PLC0415

        local_dir = str(
            snapshot_download(
                repo_id=self.config.hf_repo,
                revision=self.config.model_revision,
            )
        )
        # El layout de HF es .../snapshots/<commit_sha>/, asi que el nombre del directorio ES el
        # hash del commit — incluso cuando se pidio una rama. Es la unica forma de dejar asentado
        # en el manifest QUE pesos corrieron: sin el hash el benchmark no es reproducible.
        resolved = Path(local_dir).name
        if self.config.model_revision and resolved != self.config.model_revision:
            logger.info(
                "revision %s resolved to commit %s", self.config.model_revision, resolved
            )
        self._resolved_revision = resolved
        return local_dir

    def _check_checkpoint_agrees(self, model_config: Any) -> None:
        """Warn when the checkpoint ships different sampling defaults than the config declares."""
        for name, declared in self.sampling.items():
            shipped = getattr(model_config, name, None)
            if shipped is not None and shipped != declared:
                logger.warning(
                    "checkpoint ships %s=%s but the model config declares %s; using the config",
                    name,
                    shipped,
                    declared,
                )
        shipped_rate = getattr(getattr(model_config, "audio", None), "output_sample_rate", None)
        if shipped_rate and shipped_rate != self.default_sample_rate:
            logger.warning(
                "checkpoint outputs %s Hz but the model config declares %s",
                shipped_rate,
                self.default_sample_rate,
            )

    def unload(self) -> None:
        self._model = None
        self._latents.clear()
        self._empty_cache()
        self._loaded = False

    def _empty_cache(self) -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── sintesis ─────────────────────────────────────────────────────────────────────────────

    def reference_seconds(self, ref_wav: Path) -> float:
        """Seconds of reference the model will actually consume, from the start of the clip."""
        return min(read_duration(ref_wav), max(self.gpt_cond_len, self.max_ref_length))

    def _conditioning(self, ref_wav: Path) -> tuple[Any, Any]:
        """Speaker latents for one reference: computed once, cached, and outside the clock.

        The cost is logged separately rather than charged to whichever synthesis ran first
        (SPEC 6.3).
        """
        key = (str(ref_wav), self.gpt_cond_len, self.max_ref_length)
        if key not in self._latents:
            started = time.perf_counter()
            self._latents[key] = self._model.get_conditioning_latents(
                audio_path=[str(ref_wav)],
                gpt_cond_len=self.gpt_cond_len,
                gpt_cond_chunk_len=self.gpt_cond_chunk_len,
                max_ref_length=self.max_ref_length,
                librosa_trim_db=self.librosa_trim_db,
                sound_norm_refs=self.sound_norm_refs,
            )
            logger.info(
                "conditioned reference",
                extra={
                    "fields": {
                        "ref_wav": str(ref_wav),
                        "ref_seconds": round(self.reference_seconds(ref_wav), 2),
                        "conditioning_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    }
                },
            )
        return self._latents[key]

    def _generate_kwargs(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "do_sample": self.do_sample,
            "speed": self.speed,
            "enable_text_splitting": self.enable_text_splitting,
            **self.sampling,
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
        # XTTS-v2 clona SOLO desde audio: su `get_conditioning_latents` no acepta texto. El
        # parametro se ignora, y esa asimetria contra Qwen3-TTS —que si lo recibe— queda registrada
        # por fila en `ref_text_provided` y declarada en el README (SPEC 6.4).
        del ref_text, speaker_id
        if self._model is None:
            raise RuntimeError(f"{self.name}: load() was not called")

        ref_seconds = self.reference_seconds(ref_wav)
        gpt_cond_latent, speaker_embedding = self._conditioning(ref_wav)
        conditioning = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
        }

        try:
            if self.streaming:
                samples, ttfb_ms, total_ms = self._synthesize_streaming(text, conditioning)
            else:
                samples, ttfb_ms, total_ms = self._synthesize_whole(text, conditioning)
        except Exception as err:  # noqa: BLE001 — el OOM se reporta y la corrida sigue
            if not self._is_out_of_memory(err):
                raise
            self._empty_cache()
            return SynthesisResult(
                ref_seconds_used=ref_seconds,
                streaming=self.streaming,
                error=f"{type(err).__name__}: {err}",
            )

        if not samples:
            return SynthesisResult(
                ref_seconds_used=ref_seconds,
                streaming=self.streaming,
                error="model returned no audio",
            )

        # El wav se escribe con el reloj ya detenido: el I/O no es latencia del modelo.
        write_wav(out_path, samples, self.default_sample_rate)
        return SynthesisResult(
            sample_rate=self.default_sample_rate,
            duration_s=len(samples) / float(self.default_sample_rate),
            ttfb_ms=ttfb_ms,
            total_ms=total_ms,
            ref_seconds_used=ref_seconds,
            streaming=self.streaming,
        )

    def _synthesize_whole(
        self, text: str, conditioning: dict[str, Any]
    ) -> tuple[list[float], float, float]:
        started = time.perf_counter()
        out = self._model.inference(
            text=text,
            num_beams=self.num_beams,
            **conditioning,
            **self._generate_kwargs(),
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        # Sin streaming el primer byte de audio recien existe cuando termina todo, asi que TTFB
        # es el total. No es un placeholder: es el dato correcto para este modo.
        return as_float_list(out["wav"]), total_ms, total_ms

    def _synthesize_streaming(
        self, text: str, conditioning: dict[str, Any]
    ) -> tuple[list[float], float, float]:
        started = time.perf_counter()
        ttfb_ms: float | None = None
        samples: list[float] = []
        # inference_stream no acepta num_beams: el streaming es incompatible con beam search.
        for chunk in self._model.inference_stream(
            text=text,
            stream_chunk_size=self.stream_chunk_size,
            overlap_wav_len=self.overlap_wav_len,
            **conditioning,
            **self._generate_kwargs(),
        ):
            if ttfb_ms is None:
                ttfb_ms = (time.perf_counter() - started) * 1000.0
            samples.extend(as_float_list(chunk))
        total_ms = (time.perf_counter() - started) * 1000.0
        return samples, ttfb_ms if ttfb_ms is not None else total_ms, total_ms

    @staticmethod
    def _is_out_of_memory(err: BaseException) -> bool:
        return type(err).__name__ == "OutOfMemoryError" or "out of memory" in str(err).lower()

    # ── procedencia ──────────────────────────────────────────────────────────────────────────

    def describe(self) -> dict[str, str | None]:
        return {
            "model_revision": self._resolved_revision,
            "lib_version": self._lib_version,
            "torch_version": self._torch_version,
            "device": self._device,
            "gpu": self._gpu,
        }
