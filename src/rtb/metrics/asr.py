"""Transcription with Whisper — the expensive pass, run once (SPEC 7.1).

The ASR model and compute type are part of the metric's identity and travel in every row.
`faster_whisper` is imported inside `load()`, so this file stays importable without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rtb.schema import MetricRow

logger = logging.getLogger(__name__)

METRIC_NAME = "transcripts"


@dataclass
class AsrConfig:
    """Everything that changes a transcription, and therefore has to be recorded."""

    model: str = "large-v3"
    compute_type: str = "int8_float16"
    device: str = "cuda"
    language: str = "es"
    beam_size: int = 5
    word_timestamps: bool = True


@dataclass
class Transcription:
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)
    language_probability: float | None = None

    @property
    def last_word_end_s(self) -> float | None:
        """When the last transcribed word ends.

        Crossed against the energy envelope, this is what finds a hallucinated tail: sound after
        the last word Whisper heard is content the model invented (SPEC 7.5).
        """
        return self.words[-1]["end"] if self.words else None


def installed_version() -> str:
    """The faster-whisper version, readable without importing it.

    Needed before `load()`: the idempotency check has to know the metric version to decide what to
    skip, and loading a model to find out would defeat the point.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return f"faster-whisper=={version('faster-whisper')}"
    except PackageNotFoundError:
        return "faster-whisper==unknown"


class Transcriber:
    """Wraps one loaded Whisper model. Heavy import stays inside `load`."""

    def __init__(self, config: AsrConfig) -> None:
        self.config = config
        self._model: Any = None
        self._lib_version: str = installed_version()

    @property
    def metric_version(self) -> str:
        """Compact identity: change any of these and every WER downstream changes with it."""
        return f"{self.config.model}/{self.config.compute_type}/{self._lib_version}"

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "asr_model": self.config.model,
            "asr_compute_type": self.config.compute_type,
            "asr_device": self.config.device,
            "asr_lib_version": self._lib_version,
            "asr_language": self.config.language,
            "asr_beam_size": self.config.beam_size,
        }

    def load(self) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415 — dependencia pesada: va acá

        self._model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(self, wav_path: Path) -> Transcription:
        if self._model is None:
            raise RuntimeError("Transcriber.load() was not called")
        segments, info = self._model.transcribe(
            str(wav_path),
            language=self.config.language,
            beam_size=self.config.beam_size,
            word_timestamps=self.config.word_timestamps,
        )
        pieces: list[str] = []
        words: list[dict[str, Any]] = []
        for segment in segments:  # el generador recien acá hace el trabajo
            pieces.append(segment.text.strip())
            for word in segment.words or []:
                words.append(
                    {
                        "word": word.word.strip(),
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                    }
                )
        return Transcription(
            text=" ".join(piece for piece in pieces if piece),
            words=words,
            language_probability=getattr(info, "language_probability", None),
        )

    def transcript_row(self, generation_id: str, wav_path: Path) -> MetricRow:
        try:
            result = self.transcribe(wav_path)
        except Exception as err:  # noqa: BLE001 — un clip roto no aborta las otras 551
            logger.exception("transcription failed", extra={"fields": {"wav": str(wav_path)}})
            return MetricRow(
                generation_id=generation_id,
                metric=METRIC_NAME,
                metric_version=self.metric_version,
                error=f"{type(err).__name__}: {err}",
            )
        return MetricRow(
            generation_id=generation_id,
            metric=METRIC_NAME,
            metric_version=self.metric_version,
            values={
                # Texto CRUDO, tal como lo devolvio Whisper. La normalizacion se aplica al calcular
                # WER, no al guardar: si se guardara normalizado, cambiar el normalizador obligaria
                # a re-transcribir, que es justo lo que esta separacion evita.
                "text": result.text,
                "words": result.words,
                "last_word_end_s": result.last_word_end_s,
                "language_probability": result.language_probability,
                **self.provenance,
            },
        )
