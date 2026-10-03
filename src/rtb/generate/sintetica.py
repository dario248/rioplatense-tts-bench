"""Sintética (fine-tuned propio), in-process y local.

FastPitch multi-hablante + HiFiGAN sobre el fork vendored de coqui-tts. No clona: la voz es un
`speaker_id` horneado en los pesos, así que el clip de referencia no entra a la síntesis y sólo
sirve como objetivo de la similitud (SPEC 6.5, ANALYSIS A15).

Los pesos vienen repartidos en un checkpoint por género. Sólo uno queda residente: la grilla
itera por hablante, así que la corrida entera cuesta un recambio, y tener los dos en memoria
duplicaría la RAM pico que mide 7.3.

Heavy imports live inside `load()`; `tests/test_core_isolation.py` fails if that stops being true.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rtb.config import ModelConfig
from rtb.generate.audio_io import as_float_list, write_wav
from rtb.generate.base import SynthesisResult, TTSAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Checkpoint:
    """One set of weights and the benchmark speakers it can voice."""

    name: str
    synth: Path
    config: Path
    speakers: Path
    stats: Path | None


class SinteticaAdapter(TTSAdapter):
    """Drives the Sintética checkpoints over the benchmark grid."""

    name = "sintetica"
    max_ref_seconds = 6.0
    supports_streaming = False
    default_sample_rate = 22050
    selects_voice_by_id = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        params = dict(config.params)

        # Los pesos viven fuera del repo: el config los declara con ${SINTETICA_TTS_ROOT} y la
        # variable resuelve al checkout del servicio en cada maquina.
        declared = str(params.pop("weights_root"))
        root = Path(os.path.expandvars(declared))
        if "$" in str(root):
            raise ValueError(
                f"{self.name}: weights_root={declared!r} has unexpanded variables: set "
                "SINTETICA_TTS_ROOT to your checkout of the service"
            )
        self.weights_root = root

        # Mapeo hablante del benchmark -> voz del modelo. Va en el config y no en el adapter
        # porque cambia el experimento: es qué voz se le pide a Sintética para cada fila.
        self.voice_of: dict[str, str] = dict(params.pop("voices"))

        self.checkpoints: dict[str, Checkpoint] = {}
        # Qué checkpoint tiene cada voz. Derivado de la declaración, no descubierto escaneando:
        # un archivo de más en el directorio no puede cambiar en silencio quién sintetizó qué.
        self.checkpoint_of: dict[str, str] = {}
        for entry in params.pop("checkpoints"):
            stats = entry.get("stats")
            checkpoint = Checkpoint(
                name=str(entry["name"]),
                synth=root / entry["synth"],
                config=root / entry["config"],
                speakers=root / entry["speakers"],
                stats=root / stats if stats else None,
            )
            self.checkpoints[checkpoint.name] = checkpoint
            for voice in entry["voices"]:
                self.checkpoint_of[str(voice)] = checkpoint.name

        vocoder = params.pop("vocoder")
        self.vocoder_weights = root / vocoder["weights"]
        self.vocoder_config = root / vocoder["config"]

        self.speed: float = float(params.pop("speed", 1.0))
        self._device: str = config.device

        self._tts: Any = None
        self._resident: str | None = None
        self._revision: str | None = None
        self._lib_version: str | None = None
        self._torch_version: str | None = None
        self._gpu: str | None = None

        if params:
            raise ValueError(f"{self.name}: unknown params {sorted(params)}")

        missing = sorted(set(self.voice_of.values()) - set(self.checkpoint_of))
        if missing:
            raise ValueError(f"{self.name}: no checkpoint declares the voices {missing}")

    # ── carga ────────────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        import torch  # noqa: PLC0415 — import pesado: va acá, no a nivel de modulo

        self._torch_version = torch.__version__
        self._lib_version = self._detect_lib_version()
        self._revision = self._checkpoint_digest()
        if self._device.startswith("cuda"):
            if torch.cuda.is_available():
                self._gpu = torch.cuda.get_device_name(0)
            else:
                # Correr en CPU no es un fallback silencioso: cambia la latencia por un factor de
                # decenas, y esos numeros no son comparables con los de GPU.
                logger.warning("cuda requested but not available; falling back to cpu")
                self._device = "cpu"
        self._loaded = True

    def _detect_lib_version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        for package in ("coqui-tts", "TTS"):
            try:
                return f"{package}=={version(package)}"
            except PackageNotFoundError:
                continue
        return "coqui-tts==unknown"

    def _checkpoint_digest(self) -> str:
        """`<checkpoint>=<sha256[:12]>` por cada uno, más el vocoder.

        No hay commit de HuggingFace que citar: son `.pth` locales. El hash del contenido cumple
        la misma función que pide SPEC 5.3 — decir exactamente qué pesos corrieron.
        """
        parts = []
        for name in sorted(self.checkpoints):
            parts.append(f"{name}={_sha256_short(self.checkpoints[name].synth)}")
        parts.append(f"vocoder={_sha256_short(self.vocoder_weights)}")
        return ",".join(parts)

    def _patched_config(self, path: Path, speakers: Path | None, stats: Path | None) -> str:
        """Los configs entrenados traen paths relativos al layout original; se reapuntan acá.

        Devuelve un temporal que el caller borra: la librería sólo acepta una ruta.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        if speakers is not None:
            absolute = str(speakers.resolve())
            data["speakers_file"] = absolute
            if isinstance(data.get("model_args"), dict):
                data["model_args"]["speakers_file"] = absolute
        if isinstance(data.get("audio"), dict) and data["audio"].get("stats_path"):
            data["audio"]["stats_path"] = str(stats.resolve()) if stats else None
        handle, tmp = tempfile.mkstemp(suffix=".json", prefix="rtb_sintetica_")
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out)
        return tmp

    def _build(self, checkpoint: Checkpoint) -> Any:
        """Sintetizador + vocoder. Se reconstruye entero al cambiar de checkpoint.

        El servicio intercambia sólo el sintetizador vía `Synthesizer._load_tts`, que es API
        privada del fork. Acá la grilla itera por hablante, así que la corrida entera cuesta UN
        recambio y no vale la pena depender de eso.
        """
        from TTS.api import TTS  # noqa: PLC0415

        synth_config = self._patched_config(
            checkpoint.config, checkpoint.speakers, checkpoint.stats
        )
        vocoder_config = self._patched_config(self.vocoder_config, None, checkpoint.stats)
        started = time.perf_counter()
        try:
            tts = TTS(
                model_path=str(checkpoint.synth),
                config_path=synth_config,
                vocoder_path=str(self.vocoder_weights),
                vocoder_config_path=vocoder_config,
                gpu=self._device.startswith("cuda"),
            )
        finally:
            for tmp in (synth_config, vocoder_config):
                Path(tmp).unlink(missing_ok=True)
        # Fuera del reloj, igual que el conditioning de XTTS: es costo de arranque y cobrárselo a
        # una fila la convierte en un outlier y deja a las otras midiendo otra cosa (SPEC 6.3).
        logger.info(
            "checkpoint resident",
            extra={
                "fields": {
                    "checkpoint": checkpoint.name,
                    "load_ms": round((time.perf_counter() - started) * 1000.0, 1),
                }
            },
        )
        return tts

    def _model_for(self, voice: str) -> Any:
        name = self.checkpoint_of[voice]
        if name != self._resident:
            self._tts = None
            self._empty_cache()
            self._tts = self._build(self.checkpoints[name])
            self._resident = name
        return self._tts

    def unload(self) -> None:
        self._tts = None
        self._resident = None
        self._empty_cache()
        self._loaded = False

    def _empty_cache(self) -> None:
        import gc  # noqa: PLC0415

        gc.collect()
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        # El clip de referencia no entra a la sintesis: es el objetivo del coseno, nada mas. Y la
        # seed no mueve nada porque FastPitch es deterministico — medido, ver SPEC 6.5.
        del ref_wav, seed, ref_text
        if not self._loaded:
            raise RuntimeError(f"{self.name}: load() before synthesize()")
        if not speaker_id:
            raise ValueError(f"{self.name} selects its voice by id, but the row has no speaker")
        try:
            voice = self.voice_of[speaker_id]
        except KeyError:
            raise ValueError(
                f"{self.name}: speaker {speaker_id!r} has no voice in params.voices"
            ) from None

        model = self._model_for(voice)
        model.synthesizer.tts_model.length_scale = 1.0 / self.speed

        started = time.perf_counter()
        wav = model.tts(text=text, speaker=voice)
        total_ms = (time.perf_counter() - started) * 1000.0

        samples = as_float_list(wav)
        sample_rate = int(model.synthesizer.vocoder_ap.sample_rate)
        write_wav(out_path, samples, sample_rate)
        return SynthesisResult(
            sample_rate=sample_rate,
            duration_s=len(samples) / sample_rate if sample_rate else None,
            # Sin streaming el primer byte existe cuando termina todo, igual que los otros (6.3).
            ttfb_ms=total_ms,
            total_ms=total_ms,
            # No consume referencia: el clip de la fila es el objetivo del coseno, no un prompt.
            ref_seconds_used=None,
            ref_text_provided=False,
            streaming=False,
        )

    def describe(self) -> dict[str, str | None]:
        return {
            "model_revision": self._revision,
            "lib_version": self._lib_version,
            "torch_version": self._torch_version,
            "device": self._device,
            "gpu": self._gpu,
            # Que la voz venga de un id y no de un clip cambia cómo se lee la fila entera.
            "voice_selection": "speaker_id",
        }


def _sha256_short(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]
