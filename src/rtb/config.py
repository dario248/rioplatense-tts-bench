"""Loading and validation of the YAML configuration.

Every experimental decision lives here or in the runner, never inside an adapter (SPEC 12). The
validators are not decoration: a `target` in the wrong position invalidates the accent metric and
a colliding phrase id overwrites results, both silently. CI runs them on every push.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rtb.schema import Gender, Phenomenon, PhraseCategory, count_words, validate_id_token
from rtb.textnorm import is_normalized, normalize_text

DEFAULT_PHRASES_PATH = Path("configs/phrases.yaml")
DEFAULT_SPEAKERS_PATH = Path("configs/speakers.yaml")
DEFAULT_GRID_PATH = Path("configs/grid.yaml")
DEFAULT_THRESHOLDS_PATH = Path("configs/thresholds.yaml")

# Silencio que `scripts/build_reference_clips.py` inserta entre utterances concatenadas. Vive acá
# porque la etapa de metricas lo necesita para descontarlo al estimar la tasa de habla real: sin
# eso, cada gap cuenta como habla y la tasa sale inflada.
CONCATENATION_GAP_S = 0.25

# Ids retirados. NO se reutilizan: los manifests ya generados apuntan al texto viejo, asi que un
# id reciclado mezclaria dos frases distintas bajo la misma clave.
#   sim_*: la categoria s_implosiva se retiro de v0 (ver PhraseCategory).
DEPRECATED_PHRASE_IDS = frozenset({"sim_001", "sim_002", "sim_003", "sim_004"})

# Grafemas que en posicion inicial de palabra son siempre el fonema objetivo. La restriccion de
# SPEC 5.1 es esa posicion inicial: así el timestamp de palabra de WhisperX marca el inicio del
# segmento a medir y no hace falta montar un alineador forzado (MFA).
_YEISMO_ONSETS = ("ll", "y")
_RHOTIC_ONSET = "r"
# La conjuncion "y" aislada es la vocal [i], no la fricativa: nunca se marca.
_NEVER_A_TARGET = frozenset({"y"})

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*")


class ConfigError(Exception):
    """Raised when a config file is missing, unparseable, or internally inconsistent."""


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as err:
        raise ConfigError(f"{path}: invalid YAML: {err}") from err


def _expect_list(data: Any, path: Path) -> list[Any]:
    if not isinstance(data, list):
        raise ConfigError(f"{path}: expected a top-level YAML list, got {type(data).__name__}")
    return data


def words_of(text: str) -> list[str]:
    """Lowercased word-like tokens of a text, punctuation and digits stripped."""
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def has_onset_for(word: str, phenomenon: Phenomenon) -> bool:
    """Whether the word BEGINS with the phoneme of `phenomenon`.

    Word-initial position is what makes the realization measurable without forced alignment
    (SPEC 5.1).
    """
    lowered = word.lower()
    if lowered in _NEVER_A_TARGET:
        return False
    if phenomenon is Phenomenon.YEISMO:
        return lowered.startswith(_YEISMO_ONSETS)
    return lowered.startswith(_RHOTIC_ONSET)


class RefClip(BaseModel):
    """A reference, control or calibration clip on disk.

    `duration_s` is optional because it is measured when the clip is built, not authored by
    hand. Paths are relative to the repository root.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    duration_s: float | None = Field(default=None, gt=0)
    transcript: str | None = None
    # Ids del corpus que componen el clip, en orden: es lo que lo hace reconstruible por
    # cualquiera que tenga SLR61 (SPEC 5.2).
    source_utterances: list[str] = Field(default_factory=list)


class Speaker(BaseModel):
    """One SLR61 speaker: what is fed to the models, and what calibrates the scale."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str = "slr61"
    gender: Gender
    # Codigo real del hablante en el corpus origen (p. ej. "arf_02484"). Documenta la
    # trazabilidad sin acoplar los ids del benchmark a los del corpus.
    source_speaker: str | None = None
    notes: str | None = None

    ref_clips: list[RefClip] = Field(min_length=1)
    # Dos como minimo: el techo de similitud es la media entre DOS control_clips reales del
    # mismo hablante. Sin techo ni piso, un 0.72 no significa nada (SPEC 7.2).
    control_clips: list[RefClip] = Field(min_length=2)
    # Enunciados cortos naturales, para el umbral absoluto de duracion (SPEC 7.5). Aparte y NO en
    # el techo de similitud: promediar un clip de 0.8 s con los controles largos lo subestima.
    calibration_clips: list[RefClip] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_id_token(value, field="speaker.id")

    @property
    def all_clips(self) -> list[RefClip]:
        return [*self.ref_clips, *self.control_clips, *self.calibration_clips]

    @model_validator(mode="after")
    def _check_clips_disjoint(self) -> Speaker:
        # Un control que tambien es referencia mide el modelo contra su propio prompt: infla la
        # similitud y rompe el techo. Se chequea por ruta...
        seen_paths: set[Path] = set()
        for clip in self.all_clips:
            if clip.path in seen_paths:
                raise ValueError(
                    f"speaker {self.id}: clip path {clip.path} is used by more than one clip"
                )
            seen_paths.add(clip.path)

        # ...y por utterance de origen, que es lo que importa de verdad: dos clips con rutas
        # distintas pueden contener el mismo audio si se construyeron de la misma utterance.
        owner: dict[str, Path] = {}
        for clip in self.all_clips:
            for utterance in clip.source_utterances:
                if utterance in owner:
                    raise ValueError(
                        f"speaker {self.id}: source utterance {utterance} appears in both "
                        f"{owner[utterance]} and {clip.path}; clips must not share audio"
                    )
                owner[utterance] = clip.path
        return self

    def ref_clip(self, index: int) -> RefClip:
        try:
            return self.ref_clips[index]
        except IndexError as err:
            raise ConfigError(
                f"speaker {self.id} has {len(self.ref_clips)} ref_clips, index {index} requested"
            ) from err


class Target(BaseModel):
    """A word-initial segment to measure, and which phenomenon is measured on it."""

    model_config = ConfigDict(extra="forbid")

    word: str = Field(min_length=1)
    phenomenon: Phenomenon


class Phrase(BaseModel):
    """One benchmark phrase, its category, and the failure it is designed to expose."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=1)
    category: PhraseCategory
    failure_hypothesis: str = Field(min_length=1)

    # Segmentos medibles automaticamente. Vacio cuando la frase solo se evalua a oido.
    targets: list[Target] = Field(default_factory=list)

    # Forma hablada esperada, ya normalizada. Obligatoria en `normalizacion` y en cualquier
    # frase con digitos: es la referencia del CER, y ninguna persona dice "$15.750".
    expected_verbalization: str | None = None
    # Verbalizaciones igualmente validas. El CER se calcula contra la de menor distancia.
    accepted_alternatives: list[str] = Field(default_factory=list)

    notes: str | None = None

    # Largo del texto escrito. Se puede declarar en el YAML: si esta, tiene que coincidir con el
    # conteo derivado, y funciona como checksum de que texto y metadata no se separaron.
    n_words: int | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_id_token(value, field="phrase.id")

    @property
    def spoken_reference(self) -> str:
        """The text a speaker actually utters: the verbalization when there is one."""
        return self.expected_verbalization or self.text

    @property
    def n_spoken_words(self) -> int:
        """Spoken length, which is what the truncation and loop ratios must divide by.

        Differs from `n_words` whenever the text carries digits (SPEC 5.1).
        """
        return count_words(self.spoken_reference)

    def targets_for(self, phenomenon: Phenomenon) -> list[Target]:
        return [target for target in self.targets if target.phenomenon is phenomenon]

    @model_validator(mode="after")
    def _check_phrase(self) -> Phrase:
        computed = count_words(self.text)
        if self.n_words is not None and self.n_words != computed:
            raise ValueError(
                f"phrase {self.id}: n_words={self.n_words} but the text has {computed} tokens"
            )
        self.n_words = computed

        self._check_targets()
        self._check_verbalization()
        return self

    def _check_targets(self) -> None:
        present = set(words_of(self.text))
        for target in self.targets:
            lowered = target.word.lower()
            if lowered not in present:
                raise ValueError(
                    f"phrase {self.id}: target word {target.word!r} does not appear as a whole "
                    "word in the text"
                )
            if not has_onset_for(lowered, target.phenomenon):
                expected = "ll/y" if target.phenomenon is Phenomenon.YEISMO else "r"
                raise ValueError(
                    f"phrase {self.id}: target {target.word!r} does not begin with {expected}, "
                    f"so it is not measurable as {target.phenomenon.value}. SPEC 5.1 requires the "
                    "phoneme to be word-initial; drop it from targets and note it for manual "
                    "listening instead"
                )

        if self.category is PhraseCategory.YEISMO_REHILADO and not self.targets_for(
            Phenomenon.YEISMO
        ):
            raise ValueError(
                f"phrase {self.id}: category {self.category.value} needs at least one yeismo "
                "target, they are what the accent metric measures"
            )

    def _check_verbalization(self) -> None:
        has_digits = any(char.isdigit() for char in self.text)
        if self.category is PhraseCategory.NORMALIZACION and not self.expected_verbalization:
            raise ValueError(
                f"phrase {self.id}: category {self.category.value} requires "
                "expected_verbalization; without it the category has no objective measurement"
            )
        if has_digits and not self.expected_verbalization:
            raise ValueError(
                f"phrase {self.id}: the text contains digits, so it requires "
                "expected_verbalization; the normalizer does not expand numbers into words"
            )
        if self.accepted_alternatives and not self.expected_verbalization:
            raise ValueError(
                f"phrase {self.id}: accepted_alternatives without expected_verbalization"
            )

        # Ambos lados de la comparacion se normalizan con la misma funcion versionada; si el YAML
        # trae mayusculas, tildes o puntuacion, el CER de la categoria queda inflado en silencio.
        for field_name, value in [
            ("expected_verbalization", self.expected_verbalization),
            *[("accepted_alternatives", alt) for alt in self.accepted_alternatives],
        ]:
            if value is not None and not is_normalized(value):
                raise ValueError(
                    f"phrase {self.id}: {field_name} is not in normalized form; expected "
                    f"{normalize_text(value)!r}"
                )


class GridConfig(BaseModel):
    """The experimental grid, shared by every model.

    In its own file on purpose: per-model seed lists would silently rank different experiments.
    """

    model_config = ConfigDict(extra="forbid")

    seeds: list[int] = Field(min_length=1)
    ref_clip_indices: list[int] = Field(default_factory=lambda: [0])
    # Sintesis descartables antes de medir: la primera inferencia paga compilacion de kernels,
    # alocacion de VRAM y cache frio (SPEC 6.2).
    warmup_syntheses: int = Field(default=3, ge=0)
    warmup_text: str | None = None

    @field_validator("seeds", "ref_clip_indices")
    @classmethod
    def _check_unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError(f"duplicate entries in {value}")
        return value

    @field_validator("ref_clip_indices")
    @classmethod
    def _check_non_negative(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("ref_clip_indices must be >= 0")
        if not value:
            raise ValueError("ref_clip_indices must not be empty")
        return value


class DurationThresholds(BaseModel):
    """Bounds for the duration-based failure checks (SPEC 7.5)."""

    model_config = ConfigDict(extra="forbid")

    too_short_ratio: float = Field(gt=0, lt=1)
    too_long_ratio: float = Field(gt=1)
    short_utterance_max_words: int = Field(ge=0)
    short_utterance_allowance_s: float = Field(gt=0)


class AudioShapeThresholds(BaseModel):
    """Where sound starts and stops, which is what separates speech from padding (SPEC 7.5)."""

    model_config = ConfigDict(extra="forbid")

    frame_ms: int = Field(gt=0)
    silence_threshold_ratio: float = Field(gt=0, lt=1)
    silent_peak_ratio: float = Field(gt=0, lt=1)

    @property
    def version(self) -> str:
        """Las perillas que esta metrica usa, y ninguna mas.

        Versionar con el `version` global del archivo hacia que mover un umbral de OTRA metrica
        invalidara estas filas: el pase re-lee 2172 wav para recalcular lo mismo, o se deja el
        archivo con dos versiones que describen el mismo computo. Paso las dos cosas.
        """
        return (
            f"frame{self.frame_ms}"
            f"/silence{self.silence_threshold_ratio}"
            f"/peak{self.silent_peak_ratio}"
        )


class FailureThresholds(BaseModel):
    """The few numbers the failure taxonomy still needs: the rest is counted from the alignment
    (SPEC 7.5)."""

    model_config = ConfigDict(extra="forbid")

    insertions_min: int = Field(ge=1)
    trailing_deletions_min: int = Field(ge=1)
    ngram_repeats_min: int = Field(ge=2)
    tail_after_last_word_s: float = Field(gt=0)
    boundary_max_glued_rate: float = Field(gt=0)
    loop_min_length_ratio: float = Field(gt=1.0)


class Thresholds(BaseModel):
    """Policy numbers for the metrics stage, versioned because they change published results."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    duration: DurationThresholds
    audio_shape: AudioShapeThresholds
    failures: FailureThresholds


class ModelConfig(BaseModel):
    """Everything the runner needs to drive one model, and nothing about how it works."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    # "module.path:ClassName". Se resuelve con importlib recien en tiempo de ejecucion, para que
    # el core no importe nunca una libreria de TTS.
    adapter: str
    hf_repo: str | None = None
    # Commit hash de HuggingFace, no el nombre del repo: los pesos cambian sin aviso.
    model_revision: str | None = None

    max_ref_seconds: float = Field(gt=0)
    supports_streaming: bool = False
    default_sample_rate: int = Field(gt=0)
    device: str = "cuda"
    # True para los modelos que clonan desde (audio, texto). Cuando esta puesto, la carga del
    # config verifica que todos los ref_clips tengan transcripcion y aborta antes de sintetizar:
    # una transcripcion vacia clona peor y no da error.
    requires_ref_text: bool = False

    audio_root: Path = Path("audio")
    manifest_path: Path

    # Sintetica (fine-tuned propio) entra como fila de referencia, no compite en el ranking.
    reference_only: bool = False
    # False cuando la latencia incluye red o servidor HTTP y no es comparable in-process.
    comparable_latency: bool = True

    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        return validate_id_token(value, field="model.name")

    @field_validator("adapter")
    @classmethod
    def _check_adapter(cls, value: str) -> str:
        if value.count(":") != 1 or not all(part.strip() for part in value.split(":")):
            raise ValueError(f"adapter must be 'module.path:ClassName', got {value!r}")
        return value


class BenchConfig(BaseModel):
    """The four config files resolved together, plus the root they are relative to."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    root: Path
    model: ModelConfig
    grid: GridConfig
    phrases: list[Phrase]
    speakers: list[Speaker]

    def resolve(self, path: Path | str) -> Path:
        """Resolve a config-relative path against the repository root."""
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def phrase_by_id(self, phrase_id: str) -> Phrase:
        for phrase in self.phrases:
            if phrase.id == phrase_id:
                return phrase
        raise ConfigError(f"unknown phrase id: {phrase_id}")

    def speaker_by_id(self, speaker_id: str) -> Speaker:
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                return speaker
        raise ConfigError(f"unknown speaker id: {speaker_id}")


def _check_unique_ids(items: Sequence[Any], kind: str, path: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item.id in seen:
            duplicates.append(item.id)
        seen.add(item.id)
    if duplicates:
        raise ConfigError(f"{path}: duplicate {kind} ids: {', '.join(sorted(set(duplicates)))}")


def _validate_items(data: list[Any], model: type[BaseModel], path: Path) -> list[Any]:
    items = []
    for index, raw in enumerate(data):
        try:
            items.append(model.model_validate(raw))
        except Exception as err:  # noqa: BLE001 — se reetiqueta con archivo e indice
            raise ConfigError(f"{path}: entry #{index}: {err}") from err
    return items


def load_phrases(path: Path | str) -> list[Phrase]:
    path = Path(path)
    data = _expect_list(_load_yaml(path), path)
    phrases: list[Phrase] = _validate_items(data, Phrase, path)
    if not phrases:
        raise ConfigError(f"{path}: no phrases defined")
    _check_unique_ids(phrases, "phrase", path)

    recycled = sorted({p.id for p in phrases} & DEPRECATED_PHRASE_IDS)
    if recycled:
        raise ConfigError(
            f"{path}: deprecated phrase ids reused: {', '.join(recycled)}. Ids are permanent: "
            "already generated manifests point at the old text. Use a new id instead"
        )
    return phrases


def load_speakers(
    path: Path | str,
    *,
    root: Path | None = None,
    check_paths: bool = False,
) -> list[Speaker]:
    """Load speakers; with `check_paths=True` also assert every clip exists on disk.

    Path checking is opt-in because the configs are authored before the corpus is downloaded,
    and CI has no audio. The runner turns it on.
    """
    path = Path(path)
    data = _expect_list(_load_yaml(path), path)
    speakers: list[Speaker] = _validate_items(data, Speaker, path)
    if not speakers:
        raise ConfigError(f"{path}: no speakers defined")
    _check_unique_ids(speakers, "speaker", path)
    if check_paths:
        base = root or Path.cwd()
        missing = [
            str(clip.path)
            for speaker in speakers
            for clip in speaker.all_clips
            if not (clip.path if clip.path.is_absolute() else base / clip.path).exists()
        ]
        if missing:
            raise ConfigError(f"{path}: clip files not found: {', '.join(missing)}")
    return speakers


def load_grid(path: Path | str) -> GridConfig:
    path = Path(path)
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    try:
        return GridConfig.model_validate(data)
    except Exception as err:  # noqa: BLE001
        raise ConfigError(f"{path}: {err}") from err


def load_thresholds(path: Path | str = DEFAULT_THRESHOLDS_PATH) -> Thresholds:
    path = Path(path)
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    try:
        return Thresholds.model_validate(data)
    except Exception as err:  # noqa: BLE001
        raise ConfigError(f"{path}: {err}") from err


def load_model_config(path: Path | str) -> ModelConfig:
    path = Path(path)
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    try:
        return ModelConfig.model_validate(data)
    except Exception as err:  # noqa: BLE001
        raise ConfigError(f"{path}: {err}") from err


def load_bench_config(
    model_config: Path | str,
    *,
    root: Path | str = ".",
    phrases: Path | str = DEFAULT_PHRASES_PATH,
    speakers: Path | str = DEFAULT_SPEAKERS_PATH,
    grid: Path | str = DEFAULT_GRID_PATH,
    check_paths: bool = False,
) -> BenchConfig:
    """Load the four config files, resolving relative paths against `root`."""
    root = Path(root).resolve()

    def resolve(path: Path | str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else root / path

    config = BenchConfig(
        root=root,
        model=load_model_config(resolve(model_config)),
        grid=load_grid(resolve(grid)),
        phrases=load_phrases(resolve(phrases)),
        speakers=load_speakers(resolve(speakers), root=root, check_paths=check_paths),
    )
    if config.model.requires_ref_text:
        check_ref_texts(config)
    return config


def check_ref_texts(config: BenchConfig) -> None:
    """Every reference clip the grid will use has a non-empty transcript.

    Fail-fast at config load, not a rescue: the model raises on its own (SPEC 6.4). The silent
    case is a transcript that is WRONG, and that one is caught by
    `build_reference_clips.py --check`.
    """
    missing: list[str] = []
    for speaker in config.speakers:
        for index in sorted(config.grid.ref_clip_indices):
            if index >= len(speaker.ref_clips):
                continue  # el runner ya reporta esto por su cuenta
            clip = speaker.ref_clips[index]
            if not (clip.transcript or "").strip():
                missing.append(f"{speaker.id}/{clip.path.name}")
    if missing:
        raise ConfigError(
            f"model {config.model.name!r} sets requires_ref_text, but these reference clips have "
            f"no transcript: {', '.join(missing)}. Regenerate them with "
            "`scripts/build_reference_clips.py --check --emit-yaml`, which derives the text from "
            "`source_utterances`. Refusing to run now rather than after the checkpoint downloads"
        )


def categories_covered(phrases: Iterable[Phrase]) -> dict[str, int]:
    """Phrase count per category, for the dry-run summary and the README table."""
    counts: dict[str, int] = {category.value: 0 for category in PhraseCategory}
    for phrase in phrases:
        counts[phrase.category.value] += 1
    return counts
