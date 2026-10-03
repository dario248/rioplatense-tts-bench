"""Data models written to and read from disk by every stage of the benchmark.

This module is installed in all four environments (one per TTS model, plus evaluation), so it
must not import anything beyond the standard library and pydantic. Adding a heavy dependency
here breaks the architecture described in SPEC.md section 3, and `tests/test_core_isolation.py`
will fail.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

# Modelo reservado para las filas de audio real (control_clips de SLR61). Sin estas filas no
# hay forma de saber si un WER de 6 % es bueno o malo — ver SPEC seccion 5.5.
GROUND_TRUTH_MODEL = "__ground_truth__"
# El nombre completo no sirve como token de id (arranca con "_" y contiene el separador), asi que
# los generation_id de audio real usan esta abreviatura.
GROUND_TRUTH_PREFIX = "gt"

ID_SEPARATOR = "__"

# Los tokens de un generation_id no pueden contener el separador, o el id deja de ser parseable.
_ID_TOKEN_RE = re.compile(r"^(?!.*__)[A-Za-z0-9][A-Za-z0-9_.\-]*$")

# Un token cuenta como palabra si contiene al menos un caracter alfanumerico: asi "$15.750",
# "km/h" y "4567-8900" cuentan como una palabra cada uno, y "..." o "—" no cuentan como ninguna.
_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)


class PhraseCategory(str, Enum):
    """The ten active phrase categories of the benchmark (SPEC 5.1)."""

    YEISMO_REHILADO = "yeismo_rehilado"
    ROTICA = "rotica"
    VOSEO = "voseo"
    ENUNCIADOS_CORTOS = "enunciados_cortos"
    LEXICO_LOCAL = "lexico_local"
    TOPONIMIA = "toponimia"
    NORMALIZACION = "normalizacion"
    ANGLICISMOS = "anglicismos"
    PROSODIA = "prosodia"
    FRASE_LARGA = "frase_larga"


class Phenomenon(str, Enum):
    """A dialect feature measurable on a word-initial segment.

    Both are measured the same way — spectral features over the segment that a WhisperX word
    timestamp opens — which is why they share a target list instead of one field per feature.
    """

    # /ʎ/ y /ʝ/ realizados como fricativa postalveolar, [ʒ] sonora o [ʃ] sorda.
    YEISMO = "yeismo"
    # Vibrante multiple: numero de oclusiones, asibilacion y duracion.
    RHOTIC = "rhotic"


class Gender(str, Enum):
    FEMALE = "F"
    MALE = "M"


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp, the only kind allowed in a row."""
    return datetime.now(timezone.utc)


def count_words(text: str) -> int:
    """Count whitespace-separated tokens that carry at least one alphanumeric character.

    This is the WRITTEN length; the duration heuristics divide by `Phrase.n_spoken_words`
    (SPEC 5.1).
    """
    return sum(1 for token in text.split() if _ALNUM_RE.search(token))


def validate_id_token(token: str, *, field: str) -> str:
    """Reject tokens that would make a generation_id ambiguous."""
    if not _ID_TOKEN_RE.match(token):
        raise ValueError(
            f"{field}={token!r} is not a valid id token: use [A-Za-z0-9_.-], no "
            f"{ID_SEPARATOR!r} inside, and do not start with a separator character"
        )
    return token


def make_generation_id(
    model: str,
    speaker_id: str,
    phrase_id: str,
    seed: int,
    ref_clip_idx: int,
) -> str:
    """Build the canonical id of one synthesis: ``model__speaker__phrase__s<seed>__ref<idx>``.

    The join key between the manifest and every metrics file, so it has to be derivable from the
    grid alone — never from a counter or a timestamp.
    """
    validate_id_token(model, field="model")
    validate_id_token(speaker_id, field="speaker_id")
    validate_id_token(phrase_id, field="phrase_id")
    if ref_clip_idx < 0:
        raise ValueError(f"ref_clip_idx must be >= 0, got {ref_clip_idx}")
    return ID_SEPARATOR.join([model, speaker_id, phrase_id, f"s{seed}", f"ref{ref_clip_idx}"])


def make_warmup_id(model: str, index: int) -> str:
    """Id of a warm-up synthesis, which lives outside the grid (SPEC section 6.2)."""
    validate_id_token(model, field="model")
    return f"{model}{ID_SEPARATOR}warmup{index}"


def make_ground_truth_id(speaker_id: str, clip_name: str) -> str:
    """Id of a real-audio row, prefixed `gt` because `__ground_truth__` is not a legal id token
    (SPEC 5.5)."""
    validate_id_token(speaker_id, field="speaker_id")
    validate_id_token(clip_name, field="clip_name")
    return ID_SEPARATOR.join([GROUND_TRUTH_PREFIX, speaker_id, clip_name])


class GenerationRow(BaseModel):
    """One line of `data/manifests/*.jsonl`: one synthesis attempt, successful or not.

    Failures are written too, with `wav_path: null` and `error` populated. A failure is a
    result, not an exception to hide (SPEC section 5.3).
    """

    # protected_namespaces=() porque los campos model_* chocan con el namespace reservado de
    # pydantic; extra="forbid" para que CI detecte campos viejos o mal tipeados en los JSONL.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    generation_id: str
    model: str

    # Identidad exacta de lo que corrio. Obligatorios salvo en filas de ground truth:
    # sin el commit hash de HuggingFace el benchmark deja de ser reproducible.
    model_revision: str | None = None
    lib_version: str | None = None
    torch_version: str | None = None
    device: str | None = None
    gpu: str | None = None

    speaker_id: str
    # Nulo solo en filas de ground truth, que no se condicionaron con ninguna referencia. Poner un
    # 0 ahi seria un dato inventado que despues alguien agrupa.
    ref_clip_idx: int | None = Field(default=None, ge=0)
    ref_seconds_used: float | None = Field(default=None, gt=0)
    # Condicion experimental, no detalle de implementacion: dos modelos que clonan desde cosas
    # distintas no se comparan sin declararlo (SPEC 6.4). Default None y no False, para que una
    # fila escrita antes de que existiera el campo no afirme algo que nadie midio.
    ref_text_provided: bool | None = None

    phrase_id: str
    seed: int | None = None

    # Texto de referencia para las filas de ground truth, que no salen de phrases.yaml.
    # En las filas de modelo queda en null: el texto se toma de phrases.yaml por phrase_id.
    reference_text: str | None = None

    wav_path: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    duration_s: float | None = Field(default=None, gt=0)

    ttfb_ms: float | None = Field(default=None, ge=0)
    total_ms: float | None = Field(default=None, ge=0)
    rtf: float | None = Field(default=None, ge=0)
    streaming: bool = False

    # Las primeras sintesis de cada sesion no sirven para medir latencia: se generan pero
    # quedan marcadas y las metricas de latencia las excluyen.
    warmup: bool = False

    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        """Always serialize as UTC with a trailing Z, so string sorting equals time sorting."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @property
    def is_ground_truth(self) -> bool:
        return self.model == GROUND_TRUTH_MODEL

    @property
    def ok(self) -> bool:
        return self.error is None

    @model_validator(mode="after")
    def _check_row(self) -> GenerationRow:
        if self.error is not None:
            if self.wav_path is not None:
                raise ValueError("a row with an error must have wav_path=null")
            return self

        # Fila exitosa: el audio y sus datos basicos son obligatorios, o la etapa de metricas
        # no tiene con qué trabajar.
        missing = [
            name
            for name in ("wav_path", "sample_rate", "duration_s")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"a successful row requires {', '.join(missing)}")

        if not self.is_ground_truth:
            required = [
                name
                for name in ("model_revision", "lib_version", "device")
                if not getattr(self, name)
            ]
            if required:
                raise ValueError(
                    f"a synthesis row requires {', '.join(required)}; only "
                    f"{GROUND_TRUTH_MODEL} rows may omit them"
                )
            if self.seed is None:
                raise ValueError("a synthesis row requires a seed")
            if self.ref_clip_idx is None:
                raise ValueError("a synthesis row requires a ref_clip_idx")
        return self


class MetricRow(BaseModel):
    """One line of `data/metrics/<metric>.jsonl`.

    One file per metric, so recomputing one does not invalidate the others; `metric_version` is
    what makes a recomputation detectable (SPEC 5.4).
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    generation_id: str
    metric: str
    metric_version: str
    values: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def _check_row(self) -> MetricRow:
        if self.error is None and not self.values:
            raise ValueError("a metric row needs either values or an error")
        return self
