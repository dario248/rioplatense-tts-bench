"""The YAML a video is: a list of segments, rendered in order and concatenated.

Every text field has an `_es` twin, so the same storyboard renders in either language. Unknown
keys are an error: a typo in `sublabel` would otherwise render a video with a missing line and
no complaint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LANGUAGES = ("en", "es")


class StoryboardError(Exception):
    """The storyboard says something the renderer cannot do."""


@dataclass(frozen=True)
class Text:
    """One string per language; `es` falls back to `en` so a partial translation still renders."""

    en: str
    es: str | None = None

    def get(self, lang: str) -> str:
        return (self.es or self.en) if lang == "es" else self.en


@dataclass(frozen=True)
class Trim:
    word: str
    pad: float = 0.3


@dataclass(frozen=True)
class Card:
    duration: float
    title: Text
    lines: tuple[Text, ...] = ()
    stats: tuple[Text, ...] = ()
    note: Text | None = None

    type = "card"


@dataclass(frozen=True)
class ClipSegment:
    source: str  # generation | reference
    label: Text
    generation_id: str | None = None
    speaker: str | None = None
    clip: str = "ref_00"
    sublabel: Text | None = None
    caption: Text | None = None
    view: str = "spectrogram"
    mark_last_word: bool = False
    highlight_words: bool = False
    # La palabra que hay que escuchar: queda en color en el pie y su tramo marcado en el
    # espectrograma, para que se vea venir en vez de pasar.
    focus: str | None = None
    trim: Trim | None = None
    hold: float = 0.6

    type = "clip"


Segment = Card | ClipSegment


@dataclass(frozen=True)
class Storyboard:
    name: str
    path: Path
    lang: str
    segments: tuple[Segment, ...]
    gap: float = 0.4
    title: Text | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _text(data: dict[str, Any], key: str, where: str, *, required: bool = False) -> Text | None:
    value = data.pop(key, None)
    spanish = data.pop(f"{key}_es", None)
    if value is None:
        if required:
            raise StoryboardError(f"{where}: `{key}` is required")
        if spanish is not None:
            raise StoryboardError(f"{where}: `{key}_es` with no `{key}`")
        return None
    if not isinstance(value, str):
        raise StoryboardError(f"{where}: `{key}` must be a string")
    if spanish is not None and not isinstance(spanish, str):
        raise StoryboardError(f"{where}: `{key}_es` must be a string")
    return Text(value, spanish)


def _texts(data: dict[str, Any], key: str, where: str) -> tuple[Text, ...]:
    value = data.pop(key, None) or []
    spanish = data.pop(f"{key}_es", None) or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise StoryboardError(f"{where}: `{key}` must be a list of strings")
    if not isinstance(spanish, list) or not all(isinstance(v, str) for v in spanish):
        raise StoryboardError(f"{where}: `{key}_es` must be a list of strings")
    if spanish and len(spanish) != len(value):
        raise StoryboardError(
            f"{where}: `{key}_es` has {len(spanish)} items, `{key}` has {len(value)}"
        )
    return tuple(Text(v, spanish[i] if spanish else None) for i, v in enumerate(value))


def _done(data: dict[str, Any], where: str) -> None:
    if data:
        raise StoryboardError(f"{where}: unknown keys {', '.join(sorted(data))}")


def _card(data: dict[str, Any], where: str) -> Card:
    duration = data.pop("duration", None)
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise StoryboardError(f"{where}: a card needs a positive `duration`")
    title = _text(data, "title", where, required=True)
    assert title is not None
    card = Card(
        duration=float(duration),
        title=title,
        lines=_texts(data, "lines", where),
        stats=_texts(data, "stats", where),
        note=_text(data, "note", where),
    )
    _done(data, where)
    return card


def _clip(data: dict[str, Any], where: str) -> ClipSegment:
    source = data.pop("source", "generation")
    if source not in ("generation", "reference"):
        raise StoryboardError(f"{where}: `source` must be generation or reference")

    generation_id = data.pop("generation_id", None)
    speaker = data.pop("speaker", None)
    if source == "generation" and not generation_id:
        raise StoryboardError(f"{where}: a generation clip needs a `generation_id`")
    if source == "reference" and not speaker:
        raise StoryboardError(f"{where}: a reference clip needs a `speaker`")

    view = data.pop("view", "spectrogram")
    if view not in ("spectrogram", "waveform"):
        raise StoryboardError(f"{where}: `view` must be spectrogram or waveform")

    raw_trim = data.pop("trim", None)
    trim = None
    if raw_trim is not None:
        if not isinstance(raw_trim, dict) or "word" not in raw_trim:
            raise StoryboardError(f"{where}: `trim` must be a mapping with a `word`")
        extra = set(raw_trim) - {"word", "pad"}
        if extra:
            raise StoryboardError(f"{where}: unknown keys in `trim`: {', '.join(sorted(extra))}")
        trim = Trim(str(raw_trim["word"]), float(raw_trim.get("pad", 0.3)))

    label = _text(data, "label", where, required=True)
    assert label is not None
    clip = ClipSegment(
        source=source,
        label=label,
        generation_id=generation_id,
        speaker=speaker,
        clip=data.pop("clip", "ref_00"),
        sublabel=_text(data, "sublabel", where),
        caption=_text(data, "caption", where),
        view=view,
        mark_last_word=bool(data.pop("mark_last_word", False)),
        highlight_words=bool(data.pop("highlight_words", False)),
        focus=data.pop("focus", None),
        trim=trim,
        hold=float(data.pop("hold", 0.6)),
    )
    _done(data, where)
    if clip.source == "reference" and clip.mark_last_word:
        raise StoryboardError(f"{where}: `mark_last_word` needs a generation, not a reference")
    return clip


def load(path: Path, lang: str | None = None) -> Storyboard:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise StoryboardError(f"{path}: the storyboard must be a mapping")

    resolved = lang or data.pop("lang", "en")
    data.pop("lang", None)
    if resolved not in LANGUAGES:
        raise StoryboardError(f"{path}: lang must be one of {', '.join(LANGUAGES)}")

    raw_segments = data.pop("segments", None)
    if not isinstance(raw_segments, list) or not raw_segments:
        raise StoryboardError(f"{path}: `segments` must be a non-empty list")

    segments: list[Segment] = []
    for index, raw in enumerate(raw_segments):
        where = f"{path.name} segment {index}"
        if not isinstance(raw, dict):
            raise StoryboardError(f"{where}: must be a mapping")
        item = dict(raw)
        kind = item.pop("type", None)
        if kind == "card":
            segments.append(_card(item, where))
        elif kind == "clip":
            segments.append(_clip(item, where))
        else:
            raise StoryboardError(f"{where}: `type` must be card or clip, not {kind!r}")

    title = _text(data, "title", str(path))
    gap = float(data.pop("gap", 0.4))
    _done(data, str(path))

    return Storyboard(
        name=path.stem,
        path=path,
        lang=resolved,
        segments=tuple(segments),
        gap=gap,
        title=title,
    )
