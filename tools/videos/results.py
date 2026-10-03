"""Everything a video shows, looked up by `generation_id`.

No metric is typed into a storyboard: the renderer asks for a value and this module reads it from
the same JSONL the README's tables come from, or raises. A clip whose number is missing stops the
render instead of rendering a blank.

Audio comes from `data/samples/`, which is committed, so a clean clone renders the three videos
without regenerating anything. `data/refs/` is gitignored (SLR61, CC BY-SA) and is only reached
for a reference clip the listening set does not carry.
"""

from __future__ import annotations

import json
import re
import unicodedata
import wave
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from rtb.config import load_phrases, load_speakers
from rtb.manifest import read_rows
from rtb.metrics.speaker_sim import Scale, scale_from_rows
from rtb.schema import MetricRow

MANIFESTS = ("xtts_v2", "qwen3_tts", "chatterbox_v3", "sintetica")

# Como se escriben los modelos en pantalla. Tiene que coincidir con el README: un video que
# escribe "XTTS v2" y una tabla que escribe "XTTS-v2" ya son dos fuentes.
DISPLAY_NAMES = {
    "xtts_v2": "XTTS-v2",
    "qwen3_tts": "Qwen3-TTS",
    "chatterbox_v3": "Chatterbox v3",
    "sintetica": "Sintética",
}


class ResultsError(Exception):
    """A value a storyboard asked for is not in the data."""


def fold(word: str) -> str:
    """Compare words the way the ASR writes them, not the way the phrase set does.

    Whisper returns `callao` lowercase and `Neuquén,` with the comma attached, and the same
    model writes `Bájate` where another writes `Bajate`. Matching a `trim` target by raw string
    would miss all three.
    """
    stripped = unicodedata.normalize("NFD", word.strip())
    stripped = "".join(ch for ch in stripped if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^\w]", "", stripped, flags=re.UNICODE).casefold()


@dataclass(frozen=True)
class Word:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class Clip:
    """One synthesis, with the numbers that make it evidence of something."""

    generation_id: str
    model_id: str
    speaker_id: str
    phrase_id: str
    seed: int
    text: str
    wav: Path
    wer: float
    cer: float
    cosine: float
    sim_scaled: float
    tail_s: float
    last_word_end_s: float
    speech_end_s: float
    total_s: float
    words: tuple[Word, ...]

    @property
    def model(self) -> str:
        return DISPLAY_NAMES.get(self.model_id, self.model_id)

    def placeholders(self) -> dict[str, object]:
        return {
            "model": self.model,
            "model_id": self.model_id,
            "speaker": self.speaker_id,
            "phrase_id": self.phrase_id,
            "seed": self.seed,
            "text": self.text,
            "wer": self.wer,
            "cer": self.cer,
            "cosine": self.cosine,
            "sim_scaled": self.sim_scaled,
            "tail_s": self.tail_s,
            "total_s": self.total_s,
        }

    def find_word(self, target: str) -> Word | None:
        wanted = fold(target)
        for word in self.words:
            if fold(word.word) == wanted:
                return word
        return None


@dataclass(frozen=True)
class Reference:
    """A real SLR61 recording, the only thing that makes "sounds similar" mean anything."""

    speaker_id: str
    clip_id: str
    wav: Path
    text: str
    seconds: float

    @property
    def model(self) -> str:
        return "Real recording"

    def placeholders(self) -> dict[str, object]:
        return {
            "model": self.model,
            "speaker": self.speaker_id,
            "clip": self.clip_id,
            "text": self.text,
            "seconds": self.seconds,
        }


class Results:
    """The index the renderer resolves every on-screen value against."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._samples = {path.stem: path for path in root.glob("data/samples/*/*.wav")}

    @cached_property
    def manifest(self) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        for name in MANIFESTS:
            path = self.root / "data/manifests" / f"{name}.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        rows[row["generation_id"]] = row
        return rows

    @cached_property
    def phrases(self) -> dict[str, str]:
        return {p.id: p.text for p in load_phrases(self.root / "configs/phrases.yaml")}

    @cached_property
    def speakers(self) -> dict:
        speakers = load_speakers(self.root / "configs/speakers.yaml")
        return {s.id: s for s in speakers}

    @cached_property
    def scale(self) -> Scale:
        rows = read_rows(self.root / "data/metrics/speaker_scale.jsonl", MetricRow)
        scale = scale_from_rows(rows)
        if not scale.usable:
            raise ResultsError("speaker_scale.jsonl does not define a usable floor and ceiling")
        return scale

    def _metric(self, name: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        path = self.root / "data/metrics" / f"{name}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("error"):
                    out[row["generation_id"]] = row.get("values") or {}
        return out

    @cached_property
    def failures(self) -> dict[str, dict]:
        return self._metric("failures")

    @cached_property
    def transcripts(self) -> dict[str, dict]:
        return self._metric("transcripts")

    @cached_property
    def sims(self) -> dict[str, dict]:
        return self._metric("speaker_sim")

    @cached_property
    def shapes(self) -> dict[str, dict]:
        return self._metric("audio_shape")

    def _need(self, table: dict[str, dict], gid: str, key: str, where: str) -> float:
        values = table.get(gid)
        if values is None:
            raise ResultsError(f"{gid}: no row in {where}")
        value = values.get(key)
        if not isinstance(value, (int, float)):
            raise ResultsError(f"{gid}: {where} has no numeric `{key}`")
        return float(value)

    def wav_for(self, gid: str) -> Path:
        """The committed copy first; the generated tree only for a clip outside the samples."""
        sample = self._samples.get(gid)
        if sample is not None:
            return sample
        row = self.manifest.get(gid)
        if row is None or not row.get("wav_path"):
            raise ResultsError(f"{gid}: not in data/samples/ and no wav_path in any manifest")
        path = self.root / str(row["wav_path"])
        if not path.exists():
            raise ResultsError(
                f"{gid}: not in data/samples/ and {row['wav_path']} is not on disk — "
                "regenerate it from the manifest or curate it into the listening set"
            )
        return path

    @cached_property
    def _clips(self) -> dict[str, Clip]:
        return {}

    def clip(self, gid: str) -> Clip:
        cached = self._clips.get(gid)
        if cached is not None:
            return cached

        row = self.manifest.get(gid)
        if row is None:
            raise ResultsError(f"{gid}: not in any manifest")
        if row.get("error"):
            raise ResultsError(f"{gid}: the synthesis itself failed ({row['error']})")

        phrase_id = str(row["phrase_id"])
        if phrase_id not in self.phrases:
            raise ResultsError(f"{gid}: phrase `{phrase_id}` is not in configs/phrases.yaml")

        cosine = self._need(self.sims, gid, "cosine", "speaker_sim.jsonl")
        scaled = self.scale.normalized(cosine)
        if scaled is None:
            raise ResultsError(f"{gid}: the similarity scale could not place cosine {cosine}")

        words = tuple(
            Word(w["word"], float(w["start"]), float(w["end"]))
            for w in self.transcripts.get(gid, {}).get("words", [])
        )
        if not words:
            raise ResultsError(f"{gid}: transcripts.jsonl has no word timestamps")

        clip = Clip(
            generation_id=gid,
            model_id=str(row["model"]),
            speaker_id=str(row["speaker_id"]),
            phrase_id=phrase_id,
            seed=int(row["seed"]),
            text=self.phrases[phrase_id],
            wav=self.wav_for(gid),
            wer=self._need(self.failures, gid, "wer", "failures.jsonl"),
            cer=self._need(self.failures, gid, "cer", "failures.jsonl"),
            cosine=cosine,
            sim_scaled=scaled,
            tail_s=self._need(self.failures, gid, "trailing_audio_s", "failures.jsonl"),
            last_word_end_s=self._need(
                self.transcripts, gid, "last_word_end_s", "transcripts.jsonl"
            ),
            speech_end_s=self._need(self.shapes, gid, "speech_end_s", "audio_shape.jsonl"),
            total_s=self._need(self.shapes, gid, "total_s", "audio_shape.jsonl"),
            words=words,
        )
        self._clips[gid] = clip
        return clip

    def reference(self, speaker_id: str, clip_id: str = "ref_00") -> Reference:
        speaker = self.speakers.get(speaker_id)
        if speaker is None:
            raise ResultsError(f"{speaker_id}: not in configs/speakers.yaml")
        declared = {Path(c.path).stem: c for c in speaker.all_clips}
        if clip_id not in declared:
            raise ResultsError(
                f"{speaker_id}: no reference clip `{clip_id}` "
                f"(has {', '.join(sorted(declared))})"
            )

        # El set de escucha commitea ref_00 de cada hablante; data/refs/ no esta en git.
        wav = self._samples.get(f"{speaker_id}__reference")
        if wav is None or clip_id != "ref_00":
            wav = self.root / str(declared[clip_id].path)
        if not wav.exists():
            raise ResultsError(f"{speaker_id}/{clip_id}: {wav} is not on disk")

        with wave.open(str(wav)) as handle:
            seconds = handle.getnframes() / handle.getframerate()

        return Reference(
            speaker_id=speaker_id,
            clip_id=clip_id,
            wav=wav,
            text=declared[clip_id].transcript or "",
            seconds=seconds,
        )
