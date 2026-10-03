"""Speaker similarity: ECAPA-TDNN cosine, with the ceiling and floor that make it mean something.

    .venv-simil/bin/python -m rtb.metrics.speaker_sim \\
        --manifest data/manifests/xtts_v2.jsonl

`--unseen` measures the same syntheses against the OTHER reference clip of the same speaker,
into its own file: a synthesis that scores as high against a clip it never heard copied identity,
not the prompt.

Writes the per-synthesis cosine and, separately, the scale it has to be read against (SPEC 7.2).
The rows store the RAW cosine: normalizing at report time means a recomputed scale never
invalidates 540 rows. Runs on CPU; `speechbrain` and `torch` are imported inside `Embedder.load`.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rtb.config import DEFAULT_SPEAKERS_PATH, ConfigError, Speaker, load_speakers
from rtb.logconf import event, setup_logging
from rtb.manifest import JsonlWriter, index_by_generation_id, iter_rows, read_rows
from rtb.metrics.audio_utils import TARGET_SAMPLE_RATE
from rtb.metrics.latency import percentile
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, MetricRow

logger = logging.getLogger("rtb.metrics.speaker_sim")

METRIC_NAME = "speaker_sim"
UNSEEN_METRIC_NAME = "speaker_sim_unseen"
SCALE_METRIC_NAME = "speaker_scale"
DEFAULT_OUT_DIR = Path("data/metrics")
DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

CEILING = "ceiling"
FLOOR = "floor"
SHORT_CEILING = "short_ceiling"
REF_VS_CONTROL = "ref_vs_control"
REF_VS_REF = "ref_vs_ref"
# El clip de contenido compartido, el unico sobre el que el piso es interpretable.
SHARED_CONTENT_INDEX = 0

PAIR_DESCRIPTIONS = {
    CEILING: "same speaker, different content",
    FLOOR: "different speaker, identical content",
    SHORT_CEILING: "same speaker, one clip under 1.2 s",
    REF_VS_CONTROL: "concatenated reference vs real control",
    REF_VS_REF: "the two concatenated references",
}

# Duracion maxima de los clips de calibracion de speakers.yaml (el mas largo mide 1.17 s). Un bin
# de duracion que entra entero acá tiene un techo corto medido con el que compararse.
SHORT_CLIP_MAX_S = 1.2

# Bandas de duracion de habla, en segundos, para leer la similitud sin el sesgo de la duracion.
# La ultima es abierta: `frase_larga` produce 10 s de habla.
DURATION_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 6.0),
    (6.0, math.inf),
)


class SpeakerSimError(Exception):
    pass


# ── el coseno ────────────────────────────────────────────────────────────────────────────────────


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, in the standard library so the tests that pin it run anywhere."""
    if len(left) != len(right):
        raise SpeakerSimError(f"embeddings of different size: {len(left)} vs {len(right)}")
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        raise SpeakerSimError("cannot compare a zero embedding")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (norm_left * norm_right)


# ── los pares que definen la escala ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pair:
    """Two clips whose cosine calibrates the metric rather than measuring a model."""

    kind: str
    label: str
    left: Path
    right: Path

    @property
    def row_id(self) -> str:
        return f"scale__{self.kind}__{self.label}"


def check_shared_content(speakers: Sequence[Speaker]) -> None:
    """The floor is only interpretable if `ctrl_00` really is the same text for everyone.

    Checked and not assumed: getting it wrong fails silently and in the direction that flatters
    every model (SPEC 7.2).
    """
    transcripts: dict[str, str] = {}
    for speaker in speakers:
        clip = speaker.control_clips[SHARED_CONTENT_INDEX]
        if not clip.transcript:
            raise SpeakerSimError(
                f"speaker {speaker.id}: {clip.path} has no transcript, so it cannot be verified "
                "as the shared-content clip the floor needs"
            )
        transcripts[speaker.id] = clip.transcript.strip()
    distinct = set(transcripts.values())
    if len(distinct) > 1:
        listing = "; ".join(f"{key}: {value!r}" for key, value in sorted(transcripts.items()))
        raise SpeakerSimError(
            "the first control clip must be the same text for every speaker — the floor of "
            f"SPEC 7.2 compares identical content across speakers, but got {listing}"
        )


def scale_pairs(speakers: Sequence[Speaker]) -> list[Pair]:
    """Every pair whose cosine calibrates the metric, in a fixed order.

    `calibration_clips` stay out of the published ceiling and get their own `short_ceiling`
    instead, because clip duration confounds the cosine (SPEC 7.2).
    """
    check_shared_content(speakers)
    pairs: list[Pair] = []

    for speaker in speakers:
        # Techo: contenido distinto, mismo hablante. Con dos controles es un par por hablante;
        # el itertools esta para que agregar un tercer control no requiera tocar esto.
        for left, right in itertools.combinations(speaker.control_clips, 2):
            pairs.append(
                Pair(
                    kind=CEILING,
                    label=f"{speaker.id}__{left.path.stem}__{right.path.stem}",
                    left=left.path,
                    right=right.path,
                )
            )

    # Piso: contenido identico, hablantes distintos. Todos los pares, no solo cruzados por genero:
    # un piso calculado solo entre voces de distinto genero seria mas bajo que el real y volveria
    # a inflar la escala.
    for left_speaker, right_speaker in itertools.combinations(speakers, 2):
        pairs.append(
            Pair(
                kind=FLOOR,
                label=f"{left_speaker.id}__{right_speaker.id}",
                left=left_speaker.control_clips[SHARED_CONTENT_INDEX].path,
                right=right_speaker.control_clips[SHARED_CONTENT_INDEX].path,
            )
        )

    for speaker in speakers:
        # Techo corto (SPEC 7.2). Contra ctrl_00 y no contra el otro control, para que el par
        # tenga la misma asimetria de duracion que la medicion real.
        for calibration in speaker.calibration_clips:
            pairs.append(
                Pair(
                    kind=SHORT_CEILING,
                    label=f"{speaker.id}__{calibration.path.stem}__"
                    f"{speaker.control_clips[SHARED_CONTENT_INDEX].path.stem}",
                    left=calibration.path,
                    right=speaker.control_clips[SHARED_CONTENT_INDEX].path,
                )
            )

    for speaker in speakers:
        # Diagnostico de la construccion de los clips, no parte de la escala publicada.
        reference = speaker.ref_clips[0]
        pairs.append(
            Pair(
                kind=REF_VS_CONTROL,
                label=f"{speaker.id}__{reference.path.stem}__"
                f"{speaker.control_clips[SHARED_CONTENT_INDEX].path.stem}",
                left=reference.path,
                right=speaker.control_clips[SHARED_CONTENT_INDEX].path,
            )
        )
        if len(speaker.ref_clips) > 1:
            pairs.append(
                Pair(
                    kind=REF_VS_REF,
                    label=f"{speaker.id}__{reference.path.stem}__{speaker.ref_clips[1].path.stem}",
                    left=reference.path,
                    right=speaker.ref_clips[1].path,
                )
            )

    return pairs


# ── la escala ────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class End:
    """One reference point of the scale: the pairs behind it, not just their mean."""

    n: int = 0
    mean: float | None = None
    low: float | None = None
    high: float | None = None

    @classmethod
    def of(cls, values: Sequence[float]) -> End:
        if not values:
            return cls()
        # El rango va junto a la media SIEMPRE: cuatro pares es poca muestra, y un techo de 0.83
        # que en realidad va de 0.61 a 0.95 no es un techo, es un promedio de dos cosas distintas.
        return cls(n=len(values), mean=statistics.mean(values), low=min(values), high=max(values))


@dataclass
class Scale:
    """The reference points that make a cosine readable, keyed by pair kind."""

    ends: dict[str, End] = field(default_factory=dict)

    def end(self, kind: str) -> End:
        return self.ends.get(kind, End())

    @property
    def ceiling(self) -> float | None:
        return self.end(CEILING).mean

    @property
    def floor(self) -> float | None:
        return self.end(FLOOR).mean

    @property
    def short_ceiling(self) -> float | None:
        """What two real recordings of one speaker score when one of them is under 1.2 s."""
        return self.end(SHORT_CEILING).mean

    @property
    def usable(self) -> bool:
        ceiling, floor = self.ceiling, self.floor
        return ceiling is not None and floor is not None and ceiling > floor

    def normalized(self, similarity: float, ceiling_kind: str = CEILING) -> float | None:
        """Where a cosine sits between floor (0) and ceiling (1).

        Not clipped: above 1.0 means the synthesis matched its prompt more closely than two real
        recordings of the speaker match each other, and that is worth seeing. `--unseen` measured
        what it is — the far end of a prompt-specific gain every model has, not a mode of its own
        (SPEC 7.2). `ceiling_kind` picks which ceiling.
        """
        floor = self.floor
        ceiling = self.end(ceiling_kind).mean
        if floor is None or ceiling is None or ceiling <= floor:
            return None
        return (similarity - floor) / (ceiling - floor)

    def diagnostic_verdict(self) -> str:
        """Whether the concatenated reference clips preserved speaker identity (SPEC 7.2)."""
        concatenated = self.end(REF_VS_CONTROL)
        ceiling = self.end(CEILING)
        if concatenated.low is None or ceiling.low is None:
            return "no diagnostics"
        if concatenated.low >= ceiling.low:
            return (
                f"ok — the weakest concatenated reference scores {concatenated.low:.3f}, at or "
                f"above the weakest real same-speaker pair ({ceiling.low:.3f})"
            )
        return (
            f"CHECK THE CLIPS — a concatenated reference scores {concatenated.low:.3f}, below the "
            f"weakest real same-speaker pair ({ceiling.low:.3f})"
        )


def scale_from_rows(rows: Iterable[MetricRow]) -> Scale:
    """Rebuild the scale from `speaker_scale.jsonl`, so reporting needs no audio."""
    buckets: dict[str, list[float]] = {}
    for row in rows:
        kind = row.values.get("kind")
        similarity = row.values.get("cosine")
        if row.error or not isinstance(kind, str) or not isinstance(similarity, (int, float)):
            continue
        buckets.setdefault(kind, []).append(float(similarity))
    return Scale(ends={kind: End.of(values) for kind, values in buckets.items()})


# ── resumenes ────────────────────────────────────────────────────────────────────────────────────


@dataclass
class SimSummary:
    """Similarity for one slice, always alongside the reference point it is read against."""

    label: str
    n: int
    median: float | None = None
    p05: float | None = None
    normalized_median: float | None = None
    # Cual techo se uso: la columna normalizada no significa lo mismo contra uno que contra otro,
    # asi que viaja con el numero en vez de quedar implicita.
    ceiling_kind: str = CEILING


def summarize(
    rows: Iterable[MetricRow],
    scale: Scale,
    label: str = "all",
    ceiling_kind: str = CEILING,
) -> SimSummary:
    values = [
        float(row.values["cosine"])
        for row in rows
        if not row.error and isinstance(row.values.get("cosine"), (int, float))
    ]
    if not values:
        return SimSummary(label=label, n=0, ceiling_kind=ceiling_kind)
    median = statistics.median(values)
    return SimSummary(
        label=label,
        n=len(values),
        median=median,
        # El 5 % peor y no el p95: en similitud la cola mala es la baja, y es la que se escucha
        # como "esa voz no es la del prompt".
        p05=percentile(values, 0.05),
        normalized_median=scale.normalized(median, ceiling_kind),
        ceiling_kind=ceiling_kind,
    )


def summarize_by(
    rows: Iterable[MetricRow],
    scale: Scale,
    key: str,
) -> list[SimSummary]:
    """Group by a field stored in the rows (`category`, `speaker_id`, `ref_clip_idx`)."""
    groups: dict[str, list[MetricRow]] = {}
    for row in rows:
        value = row.values.get(key)
        if value is None:
            continue
        groups.setdefault(f"{key}={value}", []).append(row)
    return [summarize(group, scale, label) for label, group in sorted(groups.items())]


def _speech_seconds(row: MetricRow, shapes: dict[str, MetricRow]) -> float | None:
    shape = shapes.get(row.generation_id)
    if shape is None or shape.error:
        return None
    speech = shape.values.get("speech_s")
    return float(speech) if isinstance(speech, (int, float)) else None


def duration_correlation(
    rows: Iterable[MetricRow], shapes: dict[str, MetricRow]
) -> float | None:
    """Pearson between seconds of speech and cosine, measured per model and never quoted.

    It is not the same number for every model, and one model's figure under another's table turns
    a caveat into a wrong claim.
    """
    pairs = [
        (speech, float(row.values["cosine"]))
        for row in rows
        if not row.error
        and isinstance(row.values.get("cosine"), (int, float))
        and (speech := _speech_seconds(row, shapes)) is not None
    ]
    if len(pairs) < 3:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    spread = sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    return numerator / math.sqrt(spread) if spread > 0 else None


@dataclass
class BandSpread:
    """How far apart the categories sit once duration is held roughly fixed."""

    low: float
    high: float
    categories: int
    lowest: float
    highest: float

    @property
    def spread(self) -> float:
        return self.highest - self.lowest


def category_spread_in_band(
    rows: Iterable[MetricRow],
    shapes: dict[str, MetricRow],
    low: float = 2.0,
    high: float = 4.0,
    min_rows: int = 8,
) -> BandSpread | None:
    """Median cosine per category among clips of similar length (SPEC 7.2).

    Makes the category table readable: if the categories collapse together
    once duration is fixed, their apparent ordering was clip length. Categories with fewer than
    `min_rows` in the band are dropped — a median over three clips is not a position.
    """
    groups: dict[str, list[float]] = {}
    for row in rows:
        speech = _speech_seconds(row, shapes)
        category = row.values.get("category")
        if speech is None or not isinstance(category, str) or row.error:
            continue
        if low <= speech < high and isinstance(row.values.get("cosine"), (int, float)):
            groups.setdefault(category, []).append(float(row.values["cosine"]))

    medians = [statistics.median(v) for v in groups.values() if len(v) >= min_rows]
    if len(medians) < 2:
        return None
    return BandSpread(low, high, len(medians), min(medians), max(medians))


def summarize_by_duration(
    rows: Iterable[MetricRow],
    shapes: dict[str, MetricRow],
    scale: Scale,
    bins: tuple[tuple[float, float], ...] = DURATION_BINS,
) -> list[SimSummary]:
    """Group by how much speech the clip contains, joining `audio_shape.jsonl`.

    Has to be read BEFORE the category slice, because duration is the confound behind it
    (SPEC 7.2). Joined here and not stored in the row: one file per metric (SPEC 5.4).
    """
    groups: dict[tuple[float, float], list[MetricRow]] = {}
    for row in rows:
        shape = shapes.get(row.generation_id)
        if shape is None or shape.error:
            continue
        speech = shape.values.get("speech_s")
        if not isinstance(speech, (int, float)):
            continue
        for low, high in bins:
            if low <= float(speech) < high:
                groups.setdefault((low, high), []).append(row)
                break

    summaries = []
    for (low, high), group in sorted(groups.items()):
        label = f"speech {low:g}+ s" if high == math.inf else f"speech {low:g}-{high:g} s"
        # Los bins que caben enteros dentro de la duracion de los clips de calibracion se leen
        # contra el techo corto; los demas contra el largo, que es el unico alcanzable.
        kind = SHORT_CEILING if high <= SHORT_CLIP_MAX_S else CEILING
        summaries.append(summarize(group, scale, label, ceiling_kind=kind))
    return summaries


# ── el modelo de embeddings ──────────────────────────────────────────────────────────────────────


@dataclass
class EcapaConfig:
    """Everything that changes an embedding, and therefore has to be recorded."""

    model: str = DEFAULT_MODEL
    revision: str | None = None
    device: str = "cpu"
    sample_rate: int = TARGET_SAMPLE_RATE


def installed_version() -> str:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return f"speechbrain=={version('speechbrain')}"
    except PackageNotFoundError:
        return "speechbrain==unknown"


class Embedder:
    """One loaded ECAPA model plus an embedding cache. Heavy imports stay inside `load`."""

    def __init__(self, config: EcapaConfig, cache_dir: Path | None = None) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self._model: Any = None
        self._lib_version = installed_version()
        self._revision: str | None = config.revision
        self._embeddings: dict[Path, list[float]] = {}

    @property
    def revision(self) -> str:
        """The resolved checkpoint commit, or `unresolved` before `load`.

        A cosine traced to a moving Hub tag is not reproducible — same argument as
        `model_revision`.
        """
        return self._revision or "unresolved"

    @property
    def metric_version(self) -> str:
        model = self.config.model.rsplit("/", 1)[-1]
        return f"{model}/{self.revision}/{self.config.sample_rate}/{self._lib_version}"

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "sim_model": self.config.model,
            "sim_model_revision": self.revision,
            "sim_device": self.config.device,
            "sim_lib_version": self._lib_version,
            "sim_sample_rate": self.config.sample_rate,
        }

    def load(self) -> None:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415

        # Se baja el snapshot primero para quedarse con el commit: el nombre del directorio ES el
        # sha, asi que la procedencia sale del disco y no de una llamada extra a la API.
        local = Path(snapshot_download(self.config.model, revision=self.config.revision))
        self._revision = local.name
        self._model = EncoderClassifier.from_hparams(
            source=str(local),
            savedir=str(self.cache_dir) if self.cache_dir else None,
            run_opts={"device": self.config.device},
        )

    def embed(self, wav_path: Path) -> list[float]:
        """The 192-dim embedding of one clip, cached by path: 540 syntheses share 8 references."""
        resolved = wav_path.resolve()
        cached = self._embeddings.get(resolved)
        if cached is not None:
            return cached
        if self._model is None:
            raise SpeakerSimError("Embedder.load() was not called")

        import torch  # noqa: PLC0415

        from rtb.metrics.audio_utils import load_resampled  # noqa: PLC0415

        audio = load_resampled(resolved, self.config.sample_rate)
        with torch.no_grad():
            embedding = self._model.encode_batch(audio.unsqueeze(0)).squeeze()
        values = [float(value) for value in embedding]
        self._embeddings[resolved] = values
        return values

    def similarity(self, left: Path, right: Path) -> float:
        return cosine(self.embed(left), self.embed(right))


# ── filas ────────────────────────────────────────────────────────────────────────────────────────


def unseen_index(speaker: Speaker, used: int) -> int:
    """The reference clip of the same speaker that the synthesis never heard.

    Only defined for exactly two clips: with three, "the other one" is a choice and it has to be
    made explicitly rather than by this function.
    """
    if len(speaker.ref_clips) != 2:
        raise SpeakerSimError(
            f"speaker {speaker.id} has {len(speaker.ref_clips)} ref_clips: "
            "the unseen clip is only defined for two"
        )
    return 1 - used


def sim_row(
    row: GenerationRow,
    ref_path: Path,
    similarity: float,
    metric_version: str,
    *,
    category: str | None = None,
    provenance: dict[str, Any] | None = None,
    metric: str = METRIC_NAME,
    target_clip_idx: int | None = None,
) -> MetricRow:
    """One synthesis against a reference clip: the one it was conditioned on, or the other one.

    Carries the slicing keys — speaker, reference index, category — so the report can group without
    re-reading the manifest, and the reference path so a suspicious number can be listened to.
    `ref_clip_idx` is always the clip the model heard; `target_clip_idx` appears only when the
    cosine was taken against a different one.
    """
    return MetricRow(
        generation_id=row.generation_id,
        metric=metric,
        metric_version=metric_version,
        values={
            "cosine": round(similarity, 5),
            "ref_clip": str(ref_path),
            "speaker_id": row.speaker_id,
            "ref_clip_idx": row.ref_clip_idx,
            "category": category,
            **({} if target_clip_idx is None else {"target_clip_idx": target_clip_idx}),
            **(provenance or {}),
        },
    )


def scale_row(pair: Pair, similarity: float, metric_version: str) -> MetricRow:
    return MetricRow(
        generation_id=pair.row_id,
        metric=SCALE_METRIC_NAME,
        metric_version=metric_version,
        values={
            "kind": pair.kind,
            "label": pair.label,
            "cosine": round(similarity, 5),
            "left": str(pair.left),
            "right": str(pair.right),
        },
    )


def error_row(
    generation_id: str, metric: str, metric_version: str, err: BaseException
) -> MetricRow:
    return MetricRow(
        generation_id=generation_id,
        metric=metric,
        metric_version=metric_version,
        error=f"{type(err).__name__}: {err}",
    )


# ── el pase ──────────────────────────────────────────────────────────────────────────────────────


def measurable(rows: Iterable[GenerationRow]) -> list[GenerationRow]:
    """Rows this metric applies to: no errors, no warm-ups, and no ground truth (SPEC 7.2)."""
    return [
        row
        for row in rows
        if row.wav_path
        and not row.warmup
        and row.error is None
        and row.model != GROUND_TRUTH_MODEL
        and row.ref_clip_idx is not None
    ]


def already_done(path: Path, version: str) -> set[str]:
    """Ids already measured with this version. Lenient: a torn tail must not block a resume."""
    return {
        row.generation_id
        for row in iter_rows(path, MetricRow, strict=False)
        if row.metric_version == version
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtb.metrics.speaker_sim",
        description="ECAPA-TDNN cosine per synthesis, plus the ceiling and floor of the scale.",
    )
    parser.add_argument("--manifest", required=True, type=Path, action="append")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--speakers", type=Path, default=DEFAULT_SPEAKERS_PATH)
    parser.add_argument("--phrases", type=Path, default=None, help="to slice by category")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sim-model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None, help="pin the checkpoint commit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--unseen",
        action="store_true",
        help="measure against the other reference clip of the speaker, into its own file",
    )
    parser.add_argument("--limit", type=int, help="stop after N rows, for a smoke test")
    parser.add_argument("--dry-run", action="store_true", help="report what would run and exit")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--plain-logs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level, json_lines=not args.plain_logs)
    root = Path(args.root).resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        speakers = load_speakers(resolve(args.speakers))
        pairs = scale_pairs(speakers)
    except (ConfigError, SpeakerSimError) as err:
        event(logger, "speaker_sim_aborted", level=logging.CRITICAL, error=str(err))
        return 1

    categories: dict[str, str] = {}
    if args.phrases:
        from rtb.config import load_phrases  # noqa: PLC0415 — opcional: solo para el slice

        try:
            categories = {
                phrase.id: phrase.category.value for phrase in load_phrases(resolve(args.phrases))
            }
        except ConfigError as err:
            event(logger, "speaker_sim_aborted", level=logging.CRITICAL, error=str(err))
            return 1

    by_speaker = {speaker.id: speaker for speaker in speakers}
    rows: list[GenerationRow] = []
    for manifest_path in args.manifest:
        rows.extend(index_by_generation_id(read_rows(resolve(manifest_path))).values())

    embedder = Embedder(
        EcapaConfig(model=args.sim_model, revision=args.revision, device=args.device)
    )
    metric_name = UNSEEN_METRIC_NAME if args.unseen else METRIC_NAME
    sims_path = resolve(args.out) / f"{metric_name}.jsonl"
    scale_path = resolve(args.out) / f"{SCALE_METRIC_NAME}.jsonl"

    todo = measurable(rows)
    event(
        logger,
        "speaker_sim_plan",
        metric=metric_name,
        rows=len(rows),
        measurable=len(todo),
        scale_pairs=0 if args.unseen else len(pairs),
        model=args.sim_model,
        device=args.device,
        # La version definitiva incluye el commit, que se resuelve al cargar: acá va sin resolver.
        metric_version=embedder.metric_version,
    )
    if args.dry_run:
        return 0

    event(logger, "loading_sim_model", model=args.sim_model, device=args.device)
    try:
        embedder.load()
    except ImportError as err:
        event(
            logger,
            "speaker_sim_aborted",
            level=logging.CRITICAL,
            error=f"{err}. Speaker similarity has its own environment — see envs/speaker_sim.txt",
        )
        return 1

    version = embedder.metric_version
    done = already_done(sims_path, version)
    todo = [row for row in todo if row.generation_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    event(
        logger,
        "speaker_sim_resolved",
        metric_version=version,
        pending=len(todo),
        skipped=len(done),
    )

    failed = 0
    sims_path.parent.mkdir(parents=True, exist_ok=True)
    # El pase del clip no visto mide las mismas filas contra otro objetivo: la escala es la misma
    # y se lee contra la ya publicada, asi que no se toca.
    if not args.unseen:
        # La escala se reescribe entera: es funcion determinista de speakers.yaml y del
        # checkpoint, y dos valores para el mismo par serian ambiguos.
        scale_rows: list[MetricRow] = []
        for pair in pairs:
            try:
                similarity = embedder.similarity(resolve(pair.left), resolve(pair.right))
            except Exception as err:  # noqa: BLE001 — un clip roto no aborta la escala entera
                logger.exception("scale pair failed", extra={"fields": {"pair": pair.row_id}})
                scale_rows.append(error_row(pair.row_id, SCALE_METRIC_NAME, version, err))
                failed += 1
                continue
            scale_rows.append(scale_row(pair, similarity, version))
            event(
                logger,
                "scale_pair",
                level=logging.DEBUG,
                kind=pair.kind,
                label=pair.label,
                cosine=round(similarity, 4),
            )

        scale_path.unlink(missing_ok=True)
        with JsonlWriter(scale_path) as writer:
            for metric_row in scale_rows:
                writer.write(metric_row)

        scale = scale_from_rows(scale_rows)
        event(
            logger,
            "scale_measured",
            ends={
                kind: {
                    "n": end.n,
                    "mean": None if end.mean is None else round(end.mean, 4),
                    "low": None if end.low is None else round(end.low, 4),
                    "high": None if end.high is None else round(end.high, 4),
                }
                for kind, end in sorted(scale.ends.items())
            },
            verdict=scale.diagnostic_verdict(),
        )
        if not scale.usable:
            # Sin escala las filas se pueden calcular igual, pero no se pueden leer. Se avisa
            # fuerte.
            event(logger, "scale_unusable", level=logging.ERROR, ceiling_n=scale.end(CEILING).n)

    with JsonlWriter(sims_path) as writer:
        for position, row in enumerate(todo, start=1):
            speaker = by_speaker.get(row.speaker_id or "")
            if speaker is None or row.ref_clip_idx is None:
                writer.write(
                    error_row(
                        row.generation_id,
                        metric_name,
                        version,
                        SpeakerSimError(f"unknown speaker {row.speaker_id!r}"),
                    )
                )
                failed += 1
                continue
            try:
                target_idx = (
                    unseen_index(speaker, row.ref_clip_idx) if args.unseen else row.ref_clip_idx
                )
                ref_path = speaker.ref_clip(target_idx).path
                similarity = embedder.similarity(
                    resolve(Path(row.wav_path or "")), resolve(ref_path)
                )
            except Exception as err:  # noqa: BLE001 — un clip roto no aborta las otras 539
                logger.exception(
                    "similarity failed", extra={"fields": {"generation_id": row.generation_id}}
                )
                writer.write(error_row(row.generation_id, metric_name, version, err))
                failed += 1
                continue
            writer.write(
                sim_row(
                    row,
                    ref_path,
                    similarity,
                    version,
                    category=categories.get(row.phrase_id or ""),
                    provenance=embedder.provenance,
                    metric=metric_name,
                    target_clip_idx=target_idx if args.unseen else None,
                )
            )
            event(
                logger,
                "measured",
                level=logging.DEBUG,
                generation_id=row.generation_id,
                progress=f"{position}/{len(todo)}",
                cosine=round(similarity, 4),
            )

    event(
        logger,
        "speaker_sim_finished",
        metric=metric_name,
        written=len(todo),
        scale_pairs=0 if args.unseen else len(pairs),
        failed=failed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
